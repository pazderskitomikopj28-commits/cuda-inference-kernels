[CmdletBinding()]
param(
  [string]$Executable = 'D:\DevTools\Builds\cuda-inference-kernels-rtx4060\Release\rmsnorm_bench.exe',
  [int]$Rows = 4096,
  [int]$Hidden = 4096,
  [int]$BlockSize = 128,
  [int]$Warmup = 20,
  [int]$Iterations = 300,
  [int]$VerifySamples = 16384,
  [ValidateRange(1, 100)]
  [int]$Runs = 7,
  [string]$OutputCsv = ''
)

$ErrorActionPreference = 'Stop'
$Executable = (Resolve-Path -LiteralPath $Executable).Path

$orders = @(
  @('two-pass', 'fused-scalar', 'fused-float4', 'fused-half2'),
  @('fused-half2', 'fused-float4', 'two-pass', 'fused-scalar'),
  @('fused-scalar', 'two-pass', 'fused-half2', 'fused-float4'),
  @('fused-float4', 'fused-half2', 'fused-scalar', 'two-pass'),
  @('two-pass', 'fused-float4', 'fused-half2', 'fused-scalar'),
  @('fused-scalar', 'fused-half2', 'fused-float4', 'two-pass'),
  @('fused-half2', 'two-pass', 'fused-scalar', 'fused-float4')
)

$pattern = '^(two_pass|fused_scalar|fused_float4|fused_half2) mean_ms=([0-9.]+) p50_ms=([0-9.]+) p95_ms=([0-9.]+) logical_GBps=([0-9.]+) max_abs_error=([0-9.e+-]+) verified_elements=([0-9]+)'
$results = @()

for ($round = 0; $round -lt $Runs; $round++) {
  $order = $orders[$round % $orders.Count]
  foreach ($mode in $order) {
    $arguments = @(
      '--mode', $mode,
      '--rows', $Rows,
      '--hidden', $Hidden,
      '--block-size', $BlockSize,
      '--warmup', $Warmup,
      '--iters', $Iterations,
      '--verify-samples', $VerifySamples
    )
    $output = @(& $Executable @arguments 2>&1)
    if ($LASTEXITCODE -ne 0) {
      throw "Benchmark failed in round $($round + 1), mode $mode (exit code $LASTEXITCODE)."
    }
    $resultLine = $output | Where-Object { $_.ToString() -match $pattern } | Select-Object -First 1
    if (-not $resultLine) {
      throw "Benchmark output did not contain a result line in round $($round + 1), mode $mode."
    }
    $line = $resultLine.ToString()
    if ($line -notmatch $pattern) {
      throw "Could not parse benchmark output: $line"
    }
    $results += [pscustomobject]@{
      round = $round + 1
      mode = $Matches[1]
      mean_ms = [double]$Matches[2]
      p50_ms = [double]$Matches[3]
      p95_ms = [double]$Matches[4]
      logical_gbps = [double]$Matches[5]
      max_abs_error = [double]$Matches[6]
      verified_elements = [int]$Matches[7]
    }
  }
}

if ($OutputCsv) {
  $OutputCsv = [IO.Path]::GetFullPath($OutputCsv)
  $parent = Split-Path -Parent $OutputCsv
  if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
  }
  $results | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation -Encoding UTF8
  Write-Host "Wrote $OutputCsv"
}

$results | Format-Table round,mode,mean_ms,p50_ms,p95_ms,logical_gbps,max_abs_error,verified_elements -AutoSize
