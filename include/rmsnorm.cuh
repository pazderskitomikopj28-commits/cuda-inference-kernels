#pragma once

#include <cstddef>

namespace inference_kernels {

enum class RmsNormVariant {
  kTwoPass,
  kFusedScalar,
  kFusedVectorized,
  kFusedHalf2,
};

struct RmsNormOptions {
  std::size_t rows = 4096;
  std::size_t hidden_size = 4096;
  int block_size = 128;
  int warmup_iterations = 10;
  int measured_iterations = 100;
  float epsilon = 1.0e-5f;
  std::size_t verification_samples = 8192;
};

struct BenchmarkResult {
  double mean_milliseconds = 0.0;
  double p50_milliseconds = 0.0;
  double p95_milliseconds = 0.0;
  double logical_gb_per_second = 0.0;
  double max_abs_error = 0.0;
  std::size_t verified_elements = 0;
};

const char* variant_name(RmsNormVariant variant);
BenchmarkResult benchmark_rmsnorm(const RmsNormOptions& options,
                                  RmsNormVariant variant);

}  // namespace inference_kernels
