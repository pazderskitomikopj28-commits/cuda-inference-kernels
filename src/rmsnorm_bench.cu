#include "rmsnorm.cuh"

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
  inference_kernels::RmsNormOptions benchmark;
  std::string mode = "all";
};

CliOptions parse_options(int argc, char** argv) {
  CliOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    auto require_value = [&]() -> const char* {
      if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for " + argument);
      }
      return argv[++i];
    };
    if (argument == "--rows") {
      options.benchmark.rows = std::stoull(require_value());
    } else if (argument == "--hidden") {
      options.benchmark.hidden_size = std::stoull(require_value());
    } else if (argument == "--block-size") {
      options.benchmark.block_size = std::stoi(require_value());
    } else if (argument == "--warmup") {
      options.benchmark.warmup_iterations = std::stoi(require_value());
    } else if (argument == "--iters") {
      options.benchmark.measured_iterations = std::stoi(require_value());
    } else if (argument == "--epsilon") {
      options.benchmark.epsilon = std::stof(require_value());
    } else if (argument == "--verify-samples") {
      options.benchmark.verification_samples = std::stoull(require_value());
    } else if (argument == "--mode") {
      options.mode = require_value();
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  if (options.mode != "all" && options.mode != "two-pass" &&
      options.mode != "fused-scalar" && options.mode != "fused-float4") {
    throw std::invalid_argument(
        "--mode must be all, two-pass, fused-scalar or fused-float4");
  }
  return options;
}

void print_device() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return;
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) return;
  std::cout << "device=" << properties.name
            << " compute_capability=" << properties.major << '.'
            << properties.minor
            << " memory_bus_width_bits=" << properties.memoryBusWidth
            << " l2_bytes=" << properties.l2CacheSize << '\n';
}

void print_result(inference_kernels::RmsNormVariant variant,
                  const inference_kernels::BenchmarkResult& result) {
  std::cout << std::fixed << std::setprecision(6)
            << inference_kernels::variant_name(variant)
            << " mean_ms=" << result.mean_milliseconds
            << " p50_ms=" << result.p50_milliseconds
            << " p95_ms=" << result.p95_milliseconds
            << " logical_GBps=" << result.logical_gb_per_second
            << " max_abs_error=" << std::scientific << std::setprecision(3)
            << result.max_abs_error << std::fixed << std::setprecision(6)
            << " verified_elements=" << result.verified_elements << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions options = parse_options(argc, argv);
    print_device();
    std::cout << "rows=" << options.benchmark.rows
              << " hidden_size=" << options.benchmark.hidden_size
              << " block_size=" << options.benchmark.block_size
              << " warmup_iterations=" << options.benchmark.warmup_iterations
              << " measured_iterations="
              << options.benchmark.measured_iterations
              << " epsilon=" << options.benchmark.epsilon << '\n';

    std::vector<inference_kernels::RmsNormVariant> variants;
    if (options.mode == "all" || options.mode == "two-pass") {
      variants.push_back(inference_kernels::RmsNormVariant::kTwoPass);
    }
    if (options.mode == "all" || options.mode == "fused-scalar") {
      variants.push_back(inference_kernels::RmsNormVariant::kFusedScalar);
    }
    if (options.mode == "all" || options.mode == "fused-float4") {
      variants.push_back(inference_kernels::RmsNormVariant::kFusedVectorized);
    }

    bool valid = true;
    double reference_p50 = 0.0;
    for (const auto variant : variants) {
      const auto result =
          inference_kernels::benchmark_rmsnorm(options.benchmark, variant);
      print_result(variant, result);
      valid = valid && result.max_abs_error < 1.0e-4;
      if (variant == inference_kernels::RmsNormVariant::kTwoPass) {
        reference_p50 = result.p50_milliseconds;
      } else if (reference_p50 > 0.0) {
        std::cout << inference_kernels::variant_name(variant)
                  << "_vs_two_pass_speedup="
                  << reference_p50 / result.p50_milliseconds << '\n';
      }
    }
    return valid ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
