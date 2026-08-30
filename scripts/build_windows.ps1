[CmdletBinding()]
param(
  [string]$BuildDir = 'D:\DevTools\Builds\cuda-inference-kernels-rtx4060',
  [string]$CudaRoot = '',
  [string]$CMake = '',
  [string]$MsvcVersion = '14.38',
  [string]$Architecture = '89',
  [ValidateSet('Debug', 'Release')]
  [string]$Configuration = 'Release'
)

$ErrorActionPreference = 'Stop'
$projectName = 'cuda-inference-kernels'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $CudaRoot) { $CudaRoot = $env:CUDA_PATH }
if (-not $CudaRoot -and (Test-Path 'D:\DevTools\NVIDIA\CUDA\v12.4')) {
  $CudaRoot = 'D:\DevTools\NVIDIA\CUDA\v12.4'
}
if (-not $CudaRoot -or -not (Test-Path (Join-Path $CudaRoot 'bin\nvcc.exe'))) {
  throw 'CUDA Toolkit not found. Pass -CudaRoot or set CUDA_PATH.'
}
if (-not $CMake -and (Test-Path 'D:\DevTools\Kitware\CMake-4.4.3\bin\cmake.exe')) {
  $CMake = 'D:\DevTools\Kitware\CMake-4.4.3\bin\cmake.exe'
}
if (-not $CMake) {
  $command = Get-Command cmake.exe -ErrorAction SilentlyContinue
  if ($command) { $CMake = $command.Source }
}
if (-not $CMake -or -not (Test-Path $CMake)) {
  throw 'CMake not found. Pass -CMake or add it to PATH.'
}

$BuildDir = [IO.Path]::GetFullPath($BuildDir)
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$sourceDir = $projectRoot
if ($projectRoot -match '[^\x00-\x7F]') {
  $linkRoot = Join-Path (Split-Path $BuildDir -Parent) '_source_links'
  $sourceDir = Join-Path $linkRoot $projectName
  New-Item -ItemType Directory -Force -Path $linkRoot | Out-Null
  if (Test-Path -LiteralPath $sourceDir) {
    $link = Get-Item -LiteralPath $sourceDir -Force
    if ($link.LinkType -ne 'Junction' -or
        -not ($link.Target -contains $projectRoot)) {
      throw "Unexpected source-link target: $sourceDir"
    }
  } else {
    New-Item -ItemType Junction -Path $sourceDir -Target $projectRoot | Out-Null
  }
}

$tempDir = Join-Path (Split-Path $BuildDir -Parent) '_temp'
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
$env:CUDA_PATH = $CudaRoot
$env:CUDA_PATH_V12_4 = $CudaRoot
$env:TEMP = $tempDir
$env:TMP = $tempDir
$env:Path = (Join-Path $CudaRoot 'bin') + ';' + (Split-Path $CMake -Parent) +
  ';' + $env:Path

& $CMake -S $sourceDir -B $BuildDir -G 'Visual Studio 17 2022' -A x64 `
  -T "version=$MsvcVersion" "-DCMAKE_CUDA_ARCHITECTURES=$Architecture"
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed: $LASTEXITCODE" }
& $CMake --build $BuildDir --config $Configuration --parallel
if ($LASTEXITCODE -ne 0) { throw "Build failed: $LASTEXITCODE" }
& $CMake --build $BuildDir --config $Configuration --target RUN_TESTS
if ($LASTEXITCODE -ne 0) { throw "Tests failed: $LASTEXITCODE" }
