# FP16 `half2` RMSNorm results — RTX 4060 Laptop GPU

Benchmark source commit: `cadcb01` (`feat: add fp16 half2 RMSNorm path`).

## Environment and protocol

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9;
- driver 591.74; CUDA 12.4.131; MSVC 19.38.33145; Release `sm_89` build;
- shape: 4096 rows × 4096 hidden elements; epsilon: `1e-5`;
- FP16 path: FP16 input, weight and output; FP32 square-sum and inverse-RMS
  accumulation; hidden size must be divisible by two;
- every fresh process ran 10 warm-up and 100 measured invocations, timed with
  CUDA Events;
- five independent processes were run for each comparison, with a different
  variant order in every round. Process-level Mean/P50/P95 below are calculated
  from the five internal P50 values; P95 uses nearest rank.

Each run evenly checked 8192 outputs against a host FP32 reference formed from
the same FP16-quantized input and weight values. The FP16 acceptance threshold
is `2e-3`; the observed maximum error was `9.753e-4`.

```powershell
& D:\DevTools\Builds\cuda-inference-kernels-rtx4060\Release\rmsnorm_bench.exe `
  --mode all --rows 4096 --hidden 4096 --block-size 128 --warmup 10 --iters 100
```

## Variant comparison

| Variant | Process P50 ms (mean/P50/P95) | Logical GB/s at P50 | Speedup vs two-pass | Maximum error |
| --- | ---: | ---: | ---: | ---: |
| Two-pass FP32 | 0.865658 / 0.841360 / 0.962528 | 239.287 | 1.0000× | 3.605e-7 |
| Fused scalar FP32 | 0.573795 / 0.573440 / 0.574464 | 351.086 | 1.4672× | 3.605e-7 |
| Fused `float4` FP32 | 0.577283 / 0.577472 / 0.577536 | 348.634 | 1.4570× | 3.466e-7 |
| Fused `half2` FP16 | 0.291114 / 0.290816 / 0.291840 | 346.141 | 2.8931× | 9.753e-4 |

At this shape, the FP16 path is 1.9857× faster than the FP32 `float4` path by
process P50. `logical_GBps` uses 12 bytes per output element for FP32 and 6
bytes for FP16, so it is useful for like-precision comparisons but is not a
physical DRAM-throughput measurement.

One two-pass process reached 0.962528 ms while its other four P50 values fell
between 0.840704 and 0.842688 ms. The table keeps that run and exposes it in
the process P95 rather than discarding it.

The `half2` path does not use Tensor Cores: RMSNorm is an elementwise reduction
and scaling kernel. Its speedup here comes from halving input/weight/output
storage and moving two FP16 elements together, while retaining FP32 reduction
accuracy.

## Interleaved block-size sweep

The five block sizes were run in a different order in each round to reduce the
effect of clock state or adjacent runs. All values use the same shape and
`fused-half2` mode.

| Threads per block | Process P50 ms (mean/P50/P95) |
| ---: | ---: |
| 64 | 0.291021 / 0.290816 / 0.291840 |
| 128 | 0.289741 / 0.289728 / 0.289792 |
| 256 | 0.288432 / 0.289536 / 0.289792 |
| 512 | 0.288086 / 0.289600 / 0.289792 |
| 1024 | 0.395235 / 0.392192 / 0.406528 |

The 64–512 thread configurations are within 0.44% by process P50. The 128
thread setting remains the example default because it is stable and matches the
FP32 comparison protocol; this experiment does not establish a universal best
block size. The 1024-thread setting is clearly slower for this shape.

## Systems, SASS and sanitizer checks

`scripts/profile.ps1 -Mode fused-half2 -SkipNcu` produced a Systems report with
six `fused_rmsnorm_half2_kernel` instances: one warm-up plus five measured
launches. The profiler-reported median Kernel duration was 283.5915 µs. It is
close to, but not substituted for, the uninstrumented Event timing above.
This report uses Systems-only profiling and does not report Nsight Compute
occupancy, DRAM-throughput or warp-stall counters.

The `sm_89` SASS contains packed-half conversion instructions
(`HADD2.F32`, `F2FP.F16.F32.PACK_AB`) around FP32 arithmetic and paired global
loads/stores for the `__half2` data path. This confirms the compiled path keeps
the intended FP16 storage with FP32 computation.

Compute Sanitizer completed with zero memcheck errors and zero racecheck
hazards. The raw Systems artifacts are stored locally at:

```text
D:\DevTools\Profiles\cuda-inference-kernels\rmsnorm-fused-half2-r4096-h4096.nsys-rep
SHA256 56F5BFAF47E8FBFFEDAAA24050DA8AAD22A063A54AFFBB01A34F3AB555FFE761

D:\DevTools\Profiles\cuda-inference-kernels\rmsnorm-fused-half2-r4096-h4096.sqlite
SHA256 941FF2D16C28D9A2075615484B7C0EE66D7A8D7A8F4AA9A2DF7249C735EFDF4C
```
