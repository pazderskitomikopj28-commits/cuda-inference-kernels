# CUDA Inference Kernels

[![CUDA compile](https://github.com/pazderskitomikopj28-commits/cuda-inference-kernels/actions/workflows/cuda-build.yml/badge.svg)](https://github.com/pazderskitomikopj28-commits/cuda-inference-kernels/actions/workflows/cuda-build.yml)

一个以推理工作负载为背景的 CUDA 算子实验仓库。首个算子是 RMSNorm：从可验证的
两 Kernel FP32 参考路径开始，逐步比较融合标量、`float4` 向量化，以及 FP16
`half2` 融合实现。

## RMSNorm 定义

对每一行输入 `x` 和长度为 hidden size 的权重 `w`：

```text
y[i] = x[i] * rsqrt(mean(x²) + epsilon) * w[i]
```

四条实现使用同一随机种子、形状、block size 和参考公式：

- `two-pass`：先计算每行 inverse RMS，再用第二个 Kernel 应用归一化；
- `fused-scalar`：一个 block 处理一行，warp shuffle + shared memory 完成 block
  reduction，在同一 Kernel 内写回；
- `fused-float4`：保持融合 reduction，并对输入、权重和输出使用 128-bit 向量访问。
- `fused-half2`：输入、权重和输出存为 FP16；每个线程按 `half2` 读取两个元素，
  但平方和和 inverse RMS 均保留为 FP32。

`float4` 路径要求 hidden size 能被 4 整除，`half2` 路径要求能被 2 整除。FP16
路径以量化后的输入和权重作为参考，并使用 `2e-3` 的输出误差阈值；其余 FP32
路径使用 `1e-4`。大规模 benchmark 默认均匀验证 8192 个输出；单元测试对小型张量
全量验证，并覆盖非整除 hidden size 和非法 block 配置。

## 构建与运行

Windows PowerShell（工具链与产物均在 D 盘）：

```powershell
.\scripts\build_windows.ps1
& D:\DevTools\Builds\cuda-inference-kernels-rtx4060\Release\rmsnorm_bench.exe `
  --mode all --rows 4096 --hidden 4096 --block-size 128 `
  --warmup 10 --iters 100
```

Linux：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/rmsnorm_bench --mode all --rows 4096 --hidden 4096 --block-size 128 \
  --warmup 10 --iters 100
```

程序使用 CUDA Event 报告每次算子调用的 Mean/P50/P95。`logical_GBps` 按每个输出
元素一次输入读取、一次权重读取和一次输出写入计算：FP32 路径为 12 bytes，FP16
路径为 6 bytes。这是跨同精度实现一致的有效吞吐口径，不等于 profiler 观察到的物理
DRAM bytes。

## VSCode 开发

仓库包含 `.vscode/launch.json` 和 `tasks.json`。在 Windows 上打开仓库后，Run and
Debug 中选择 `Debug RMSNorm benchmark` 或 `Debug RMSNorm unit tests`，按 F5 会调用
`scripts/build_windows.ps1`，在
`D:\DevTools\Builds\cuda-inference-kernels-vscode-debug` 构建 Debug 目标并运行 CTest。

`.devcontainer/devcontainer.json` 提供 CUDA 12.4 GPU 容器配置，并在容器中安装
Nsight VSCode Edition、C/C++ 和 CMake Tools。需要 Docker Desktop 的 WSL2 GPU 后端。

## Profiling 与安全检查

```powershell
$env:Path = 'D:\DevTools\NVIDIA\Nsight-Systems-2026.4.1\target-windows-x64;' + `
  'D:\DevTools\NVIDIA\Nsight-Compute-2026.2.1\target\windows-desktop-win7-x64;' + `
  $env:Path
.\scripts\profile.ps1 -Mode fused-float4
$env:CUDA_PATH = 'D:\DevTools\NVIDIA\CUDA\v12.4'
.\scripts\sanitize.ps1
```

Nsight Systems 用于确认 launch 数量和时间线，Nsight Compute 用于检查内存吞吐、
occupancy 和指令行为。Profiler 会改变绝对耗时，因此性能表使用未插桩的 Event 结果，
profile 只作为机制证据。

如果目标机的 Nsight Compute 无法初始化，可用 `-SkipNcu` 只生成 Systems 报告；
应在结果文档中明确记录缺失的 hardware counters，不能用 Event 或 SASS 代替这些指标。

RTX 4060 Laptop GPU 的五进程对照、block-size sweep、Systems 时间线和 SASS
检查见 [`docs/results/rmsnorm-rtx4060-2026-08-30.md`](docs/results/rmsnorm-rtx4060-2026-08-30.md)。

## 后续实验

- BF16 输入、FP32 累加与向量化转换；
- 支持未对齐 hidden size 的安全尾部处理；
- 针对小 hidden size 的一 warp 一行实现；
- 将 RMSNorm 与后续量化或门控算子融合；
- 在不同 GPU 上重新 sweep block size，不把 RTX 4060 的最优配置当作通用结论。

## 许可证

MIT
