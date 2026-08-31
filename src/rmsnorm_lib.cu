#include "rmsnorm.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void cuda_check(cudaError_t status, const char* expression, const char* file,
                int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error at ") + file + ':' +
                             std::to_string(line) + " for " + expression +
                             ": " + cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(call) cuda_check((call), #call, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t elements) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_),
                          elements * sizeof(T)));
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) cudaFree(data_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* data() { return data_; }
  const T* data() const { return data_; }

 private:
  T* data_ = nullptr;
};

class Stream {
 public:
  Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
  ~Stream() {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class Event {
 public:
  Event() { CUDA_CHECK(cudaEventCreate(&event_)); }
  ~Event() {
    if (event_ != nullptr) cudaEventDestroy(event_);
  }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

__inline__ __device__ float warp_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
  }
  return value;
}

__inline__ __device__ float block_sum(float value) {
  __shared__ float warp_sums[32];
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  value = warp_sum(value);
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();
  const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
  value = threadIdx.x < warp_count ? warp_sums[lane] : 0.0f;
  if (warp == 0) value = warp_sum(value);
  return value;
}

__global__ void row_inverse_rms_kernel(const float* input, float* row_scales,
                                       std::size_t hidden_size,
                                       float epsilon) {
  const std::size_t row = blockIdx.x;
  const float* row_input = input + row * hidden_size;
  float sum = 0.0f;
  for (std::size_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    const float value = row_input[column];
    sum = fmaf(value, value, sum);
  }
  sum = block_sum(sum);
  if (threadIdx.x == 0) {
    row_scales[row] =
        rsqrtf(sum / static_cast<float>(hidden_size) + epsilon);
  }
}

__global__ void apply_rmsnorm_kernel(const float* input, const float* weights,
                                     const float* row_scales, float* output,
                                     std::size_t elements,
                                     std::size_t hidden_size) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
  if (index < elements) {
    const std::size_t row = index / hidden_size;
    const std::size_t column = index - row * hidden_size;
    output[index] = input[index] * row_scales[row] * weights[column];
  }
}

__global__ void fused_rmsnorm_scalar_kernel(
    const float* input, const float* weights, float* output,
    std::size_t hidden_size, float epsilon) {
  const std::size_t row = blockIdx.x;
  const float* row_input = input + row * hidden_size;
  float* row_output = output + row * hidden_size;
  float sum = 0.0f;
  for (std::size_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    const float value = row_input[column];
    sum = fmaf(value, value, sum);
  }
  sum = block_sum(sum);
  __shared__ float inverse_rms;
  if (threadIdx.x == 0) {
    inverse_rms =
        rsqrtf(sum / static_cast<float>(hidden_size) + epsilon);
  }
  __syncthreads();
  for (std::size_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    row_output[column] = row_input[column] * inverse_rms * weights[column];
  }
}

__global__ void fused_rmsnorm_float4_kernel(
    const float4* input, const float4* weights, float4* output,
    std::size_t vectors_per_row, std::size_t hidden_size, float epsilon) {
  const std::size_t row = blockIdx.x;
  const float4* row_input = input + row * vectors_per_row;
  float4* row_output = output + row * vectors_per_row;
  float sum = 0.0f;
  for (std::size_t column = threadIdx.x; column < vectors_per_row;
       column += blockDim.x) {
    const float4 value = row_input[column];
    sum = fmaf(value.x, value.x, sum);
    sum = fmaf(value.y, value.y, sum);
    sum = fmaf(value.z, value.z, sum);
    sum = fmaf(value.w, value.w, sum);
  }
  sum = block_sum(sum);
  __shared__ float inverse_rms;
  if (threadIdx.x == 0) {
    inverse_rms =
        rsqrtf(sum / static_cast<float>(hidden_size) + epsilon);
  }
  __syncthreads();
  for (std::size_t column = threadIdx.x; column < vectors_per_row;
       column += blockDim.x) {
    const float4 value = row_input[column];
    const float4 weight = weights[column];
    row_output[column] =
        make_float4(value.x * inverse_rms * weight.x,
                    value.y * inverse_rms * weight.y,
                    value.z * inverse_rms * weight.z,
                    value.w * inverse_rms * weight.w);
  }
}

__global__ void fused_rmsnorm_half2_kernel(
    const __half2* input, const __half2* weights, __half2* output,
    std::size_t pairs_per_row, std::size_t hidden_size, float epsilon) {
  const std::size_t row = blockIdx.x;
  const __half2* row_input = input + row * pairs_per_row;
  __half2* row_output = output + row * pairs_per_row;
  float sum = 0.0f;
  for (std::size_t column = threadIdx.x; column < pairs_per_row;
       column += blockDim.x) {
    const float2 value = __half22float2(row_input[column]);
    sum = fmaf(value.x, value.x, sum);
    sum = fmaf(value.y, value.y, sum);
  }
  sum = block_sum(sum);
  __shared__ float inverse_rms;
  if (threadIdx.x == 0) {
    inverse_rms =
        rsqrtf(sum / static_cast<float>(hidden_size) + epsilon);
  }
  __syncthreads();
  for (std::size_t column = threadIdx.x; column < pairs_per_row;
       column += blockDim.x) {
    const float2 value = __half22float2(row_input[column]);
    const float2 weight = __half22float2(weights[column]);
    row_output[column] = __floats2half2_rn(
        value.x * inverse_rms * weight.x,
        value.y * inverse_rms * weight.y);
  }
}

void validate_options(const inference_kernels::RmsNormOptions& options,
                      inference_kernels::RmsNormVariant variant) {
  if (options.rows == 0 || options.hidden_size == 0 ||
      options.block_size < 32 || options.block_size > 1024 ||
      options.block_size % 32 != 0 || options.warmup_iterations < 0 ||
      options.measured_iterations <= 0 || !std::isfinite(options.epsilon) ||
      options.epsilon <= 0.0f) {
    throw std::invalid_argument(
        "rows and hidden size must be positive; block size must be a multiple "
        "of 32 in [32, 1024]; warmup must be non-negative; measured "
        "iterations and epsilon must be positive");
  }
  if (options.rows > std::numeric_limits<unsigned int>::max() ||
      options.rows > std::numeric_limits<std::size_t>::max() /
                         options.hidden_size) {
    throw std::overflow_error("RMSNorm shape exceeds the supported grid or size");
  }
  const std::size_t elements = options.rows * options.hidden_size;
  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error("RMSNorm tensor byte size overflows size_t");
  }
  const std::size_t apply_blocks =
      (elements + static_cast<std::size_t>(options.block_size) - 1) /
      static_cast<std::size_t>(options.block_size);
  if (apply_blocks > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error("RMSNorm element grid exceeds CUDA grid.x");
  }
  if (variant == inference_kernels::RmsNormVariant::kFusedVectorized &&
      options.hidden_size % 4 != 0) {
    throw std::invalid_argument(
        "the float4 RMSNorm variant requires hidden size divisible by four");
  }
  if (variant == inference_kernels::RmsNormVariant::kFusedHalf2 &&
      options.hidden_size % 2 != 0) {
    throw std::invalid_argument(
        "the half2 RMSNorm variant requires hidden size divisible by two");
  }
}

void fill_inputs(std::vector<float>& input, std::vector<float>& weights) {
  std::mt19937 generator(2026);
  std::uniform_real_distribution<float> input_distribution(-1.0f, 1.0f);
  std::uniform_real_distribution<float> weight_distribution(0.5f, 1.5f);
  for (float& value : input) value = input_distribution(generator);
  for (float& value : weights) value = weight_distribution(generator);
}

std::vector<__half> quantize_to_half(std::vector<float>& values) {
  std::vector<__half> quantized(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    quantized[index] = __float2half(values[index]);
    values[index] = __half2float(quantized[index]);
  }
  return quantized;
}

std::vector<float> convert_from_half(const std::vector<__half>& values) {
  std::vector<float> converted(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    converted[index] = __half2float(values[index]);
  }
  return converted;
}

void launch_rmsnorm(inference_kernels::RmsNormVariant variant,
                    const inference_kernels::RmsNormOptions& options,
                    const float* input, const float* weights, float* row_scales,
                    float* output, cudaStream_t stream) {
  const dim3 row_grid(static_cast<unsigned int>(options.rows));
  const dim3 block(static_cast<unsigned int>(options.block_size));
  const std::size_t elements = options.rows * options.hidden_size;
  if (variant == inference_kernels::RmsNormVariant::kTwoPass) {
    row_inverse_rms_kernel<<<row_grid, block, 0, stream>>>(
        input, row_scales, options.hidden_size, options.epsilon);
    CUDA_CHECK(cudaGetLastError());
    const unsigned int apply_blocks = static_cast<unsigned int>(
        (elements + options.block_size - 1) / options.block_size);
    apply_rmsnorm_kernel<<<apply_blocks, block, 0, stream>>>(
        input, weights, row_scales, output, elements, options.hidden_size);
    CUDA_CHECK(cudaGetLastError());
  } else if (variant == inference_kernels::RmsNormVariant::kFusedScalar) {
    fused_rmsnorm_scalar_kernel<<<row_grid, block, 0, stream>>>(
        input, weights, output, options.hidden_size, options.epsilon);
    CUDA_CHECK(cudaGetLastError());
  } else {
    fused_rmsnorm_float4_kernel<<<row_grid, block, 0, stream>>>(
        reinterpret_cast<const float4*>(input),
        reinterpret_cast<const float4*>(weights),
        reinterpret_cast<float4*>(output), options.hidden_size / 4,
        options.hidden_size, options.epsilon);
    CUDA_CHECK(cudaGetLastError());
  }
}

void launch_rmsnorm_half2(const inference_kernels::RmsNormOptions& options,
                          const __half* input, const __half* weights,
                          __half* output, cudaStream_t stream) {
  const dim3 row_grid(static_cast<unsigned int>(options.rows));
  const dim3 block(static_cast<unsigned int>(options.block_size));
  fused_rmsnorm_half2_kernel<<<row_grid, block, 0, stream>>>(
      reinterpret_cast<const __half2*>(input),
      reinterpret_cast<const __half2*>(weights),
      reinterpret_cast<__half2*>(output), options.hidden_size / 2,
      options.hidden_size, options.epsilon);
  CUDA_CHECK(cudaGetLastError());
}

double reference_inverse_rms(const std::vector<float>& input,
                             std::size_t row, std::size_t hidden_size,
                             float epsilon) {
  double sum = 0.0;
  const std::size_t offset = row * hidden_size;
  for (std::size_t column = 0; column < hidden_size; ++column) {
    const double value = input[offset + column];
    sum += value * value;
  }
  return 1.0 / std::sqrt(sum / static_cast<double>(hidden_size) + epsilon);
}

struct VerificationResult {
  double max_abs_error = 0.0;
  std::size_t checked_elements = 0;
};

VerificationResult verify_output(const std::vector<float>& input,
                                 const std::vector<float>& weights,
                                 const std::vector<float>& output,
                                 const inference_kernels::RmsNormOptions& options) {
  const std::size_t elements = input.size();
  const std::size_t samples = options.verification_samples == 0
                                  ? elements
                                  : std::min(elements,
                                             options.verification_samples);
  double max_error = 0.0;
  if (samples == elements) {
    for (std::size_t row = 0; row < options.rows; ++row) {
      const double inverse_rms = reference_inverse_rms(
          input, row, options.hidden_size, options.epsilon);
      const std::size_t offset = row * options.hidden_size;
      for (std::size_t column = 0; column < options.hidden_size; ++column) {
        const double expected = input[offset + column] * inverse_rms *
                                weights[column];
        max_error = std::max(
            max_error, std::abs(expected - output[offset + column]));
      }
    }
    return {max_error, samples};
  }

  std::unordered_map<std::size_t, double> inverse_rms_by_row;
  for (std::size_t sample = 0; sample < samples; ++sample) {
    const std::size_t index = samples == 1
                                  ? 0
                                  : static_cast<std::size_t>(
                                        static_cast<long double>(sample) *
                                        (elements - 1) / (samples - 1));
    const std::size_t row = index / options.hidden_size;
    const std::size_t column = index - row * options.hidden_size;
    auto [position, inserted] = inverse_rms_by_row.emplace(row, 0.0);
    if (inserted) {
      position->second = reference_inverse_rms(
          input, row, options.hidden_size, options.epsilon);
    }
    const double expected =
        input[index] * position->second * weights[column];
    max_error = std::max(max_error, std::abs(expected - output[index]));
  }
  return {max_error, samples};
}

double nearest_rank(std::vector<float> values, double quantile) {
  std::sort(values.begin(), values.end());
  const std::size_t rank = static_cast<std::size_t>(
      std::ceil(quantile * static_cast<double>(values.size())));
  return values[std::max<std::size_t>(1, rank) - 1];
}

double median_value(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 != 0) return values[middle];
  return (static_cast<double>(values[middle - 1]) + values[middle]) / 2.0;
}

}  // namespace

namespace inference_kernels {

const char* variant_name(RmsNormVariant variant) {
  switch (variant) {
    case RmsNormVariant::kTwoPass:
      return "two_pass";
    case RmsNormVariant::kFusedScalar:
      return "fused_scalar";
    case RmsNormVariant::kFusedVectorized:
      return "fused_float4";
    case RmsNormVariant::kFusedHalf2:
      return "fused_half2";
  }
  return "unknown";
}

BenchmarkResult benchmark_rmsnorm(const RmsNormOptions& options,
                                  RmsNormVariant variant) {
  validate_options(options, variant);
  const std::size_t elements = options.rows * options.hidden_size;
  std::vector<float> host_input(elements);
  std::vector<float> host_weights(options.hidden_size);
  std::vector<float> host_output(elements);
  fill_inputs(host_input, host_weights);

  if (variant == RmsNormVariant::kFusedHalf2) {
    const std::vector<__half> host_input_half = quantize_to_half(host_input);
    const std::vector<__half> host_weights_half = quantize_to_half(host_weights);
    std::vector<__half> host_output_half(elements);
    DeviceBuffer<__half> device_input(elements);
    DeviceBuffer<__half> device_weights(options.hidden_size);
    DeviceBuffer<__half> device_output(elements);
    CUDA_CHECK(cudaMemcpy(device_input.data(), host_input_half.data(),
                          elements * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_weights.data(), host_weights_half.data(),
                          options.hidden_size * sizeof(__half),
                          cudaMemcpyHostToDevice));
    Stream stream;
    Event start;
    Event stop;

    for (int iteration = 0; iteration < options.warmup_iterations;
         ++iteration) {
      launch_rmsnorm_half2(options, device_input.data(), device_weights.data(),
                           device_output.data(), stream.get());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    std::vector<float> timings;
    timings.reserve(options.measured_iterations);
    for (int iteration = 0; iteration < options.measured_iterations;
         ++iteration) {
      CUDA_CHECK(cudaEventRecord(start.get(), stream.get()));
      launch_rmsnorm_half2(options, device_input.data(), device_weights.data(),
                           device_output.data(), stream.get());
      CUDA_CHECK(cudaEventRecord(stop.get(), stream.get()));
      CUDA_CHECK(cudaEventSynchronize(stop.get()));
      float milliseconds = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()));
      timings.push_back(milliseconds);
    }

    CUDA_CHECK(cudaMemcpy(host_output_half.data(), device_output.data(),
                          elements * sizeof(__half), cudaMemcpyDeviceToHost));
    host_output = convert_from_half(host_output_half);
    const VerificationResult verification =
        verify_output(host_input, host_weights, host_output, options);
    const double mean =
        std::accumulate(timings.begin(), timings.end(), 0.0) /
        static_cast<double>(timings.size());
    const double p50 = median_value(timings);
    const double p95 = nearest_rank(timings, 0.95);
    const double logical_bytes =
        static_cast<double>(elements) * sizeof(__half) * 3.0;
    return {mean,
            p50,
            p95,
            logical_bytes / (p50 * 1.0e6),
            verification.max_abs_error,
            verification.checked_elements};
  }

  DeviceBuffer<float> device_input(elements);
  DeviceBuffer<float> device_weights(options.hidden_size);
  DeviceBuffer<float> device_output(elements);
  DeviceBuffer<float> device_row_scales(options.rows);
  CUDA_CHECK(cudaMemcpy(device_input.data(), host_input.data(),
                        elements * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_weights.data(), host_weights.data(),
                        options.hidden_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  Stream stream;
  Event start;
  Event stop;

  for (int iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    launch_rmsnorm(variant, options, device_input.data(), device_weights.data(),
                   device_row_scales.data(), device_output.data(), stream.get());
  }
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  std::vector<float> timings;
  timings.reserve(options.measured_iterations);
  for (int iteration = 0; iteration < options.measured_iterations; ++iteration) {
    CUDA_CHECK(cudaEventRecord(start.get(), stream.get()));
    launch_rmsnorm(variant, options, device_input.data(), device_weights.data(),
                   device_row_scales.data(), device_output.data(), stream.get());
    CUDA_CHECK(cudaEventRecord(stop.get(), stream.get()));
    CUDA_CHECK(cudaEventSynchronize(stop.get()));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()));
    timings.push_back(milliseconds);
  }

  CUDA_CHECK(cudaMemcpy(host_output.data(), device_output.data(),
                        elements * sizeof(float), cudaMemcpyDeviceToHost));
  const VerificationResult verification =
      verify_output(host_input, host_weights, host_output, options);
  const double mean = std::accumulate(timings.begin(), timings.end(), 0.0) /
                      static_cast<double>(timings.size());
  const double p50 = median_value(timings);
  const double p95 = nearest_rank(timings, 0.95);
  const double logical_bytes = static_cast<double>(elements) * sizeof(float) * 3.0;
  return {mean,
          p50,
          p95,
          logical_bytes / (p50 * 1.0e6),
          verification.max_abs_error,
          verification.checked_elements};
}

}  // namespace inference_kernels
