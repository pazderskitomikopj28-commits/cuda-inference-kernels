[CmdletBinding()]
param(
  [ValidateSet('two-pass', 'fused-scalar', 'fused-float4')]
  [string]$Mode = 'fused-float4',
  [int]$Rows = 4096,
  [int]$Hidden = 4096,
  [switch]$SkipNcu,
  [string]$Executable = 'D:\DevTools\Builds\cuda-inference-kernels-rtx4060\Release\rmsnorm_bench.exe',
  [string]$ReportDir = 'D:\DevTools\Profiles\cuda-inference-kernels'
)

$ErrorActionPreference = 'Stop'
$Executable = (Resolve-Path $Executable).Path
$ReportDir = [IO.Path]::GetFullPath($ReportDir)
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
$tempDir = Join-Path (Split-Path $ReportDir -Parent) '_temp'
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
$env:TEMP = $tempDir
$env:TMP = $tempDir
if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH 'bin'))) {
  $env:Path = (Join-Path $env:CUDA_PATH 'bin') + ';' + $env:Path
}
$trace = if ($env:OS -eq 'Windows_NT') { 'cuda,nvtx' } else { 'cuda,nvtx,osrt' }
$platformOptions = if ($env:OS -eq 'Windows_NT') {
  @('--sample=none', '--cpuctxsw=none')
} else { @() }
$name = "rmsnorm-$Mode-r${Rows}-h${Hidden}"
$launchSkip = if ($Mode -eq 'two-pass') { 2 } else { 1 }
$launchCount = if ($Mode -eq 'two-pass') { 2 } else { 1 }
Push-Location (Split-Path $Executable -Parent)
try {
  nsys profile --force-overwrite=true --trace=$trace --stats=true `
    @platformOptions -o (Join-Path $ReportDir $name) $Executable `
    --mode $Mode --rows $Rows --hidden $Hidden --warmup 1 --iters 5
  if ($LASTEXITCODE -ne 0) { throw "nsys failed: $LASTEXITCODE" }
  if (-not $SkipNcu) {
    ncu --force-overwrite --set basic --page raw --kernel-name-base demangled `
      --launch-skip $launchSkip --launch-count $launchCount --csv `
      --log-file (Join-Path $ReportDir "$name-ncu.csv") $Executable `
      --mode $Mode --rows $Rows --hidden $Hidden --warmup 1 --iters 1
    if ($LASTEXITCODE -ne 0) { throw "ncu failed: $LASTEXITCODE" }
  }
} finally {
  Pop-Location
}
