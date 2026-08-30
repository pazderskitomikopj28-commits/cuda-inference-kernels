#include "rmsnorm.cuh"

#include <exception>
#include <iostream>
#include <stdexcept>

int main() {
  try {
    inference_kernels::RmsNormOptions options;
    options.rows = 5;
    options.hidden_size = 1024;
    options.block_size = 128;
    options.warmup_iterations = 1;
    options.measured_iterations = 2;
    options.verification_samples = 0;
    for (const auto variant : {
             inference_kernels::RmsNormVariant::kTwoPass,
             inference_kernels::RmsNormVariant::kFusedScalar,
             inference_kernels::RmsNormVariant::kFusedVectorized}) {
      const auto result = inference_kernels::benchmark_rmsnorm(options, variant);
      if (result.max_abs_error >= 1.0e-4 ||
          result.verified_elements != options.rows * options.hidden_size) {
        std::cerr << "RMSNorm correctness failed for "
                  << inference_kernels::variant_name(variant) << '\n';
        return 1;
      }
    }

    options.rows = 7;
    options.hidden_size = 1003;
    const auto uneven = inference_kernels::benchmark_rmsnorm(
        options, inference_kernels::RmsNormVariant::kFusedScalar);
    if (uneven.max_abs_error >= 1.0e-4) {
      std::cerr << "non-vectorized tail shape failed\n";
      return 1;
    }

    bool rejected = false;
    try {
      (void)inference_kernels::benchmark_rmsnorm(
          options, inference_kernels::RmsNormVariant::kFusedVectorized);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected) {
      std::cerr << "float4 variant accepted an incompatible hidden size\n";
      return 1;
    }

    rejected = false;
    try {
      options.hidden_size = 1024;
      options.block_size = 100;
      (void)inference_kernels::benchmark_rmsnorm(
          options, inference_kernels::RmsNormVariant::kFusedScalar);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected) {
      std::cerr << "invalid block size was accepted\n";
      return 1;
    }
    std::cout << "RMSNorm tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
