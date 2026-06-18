param(
    [string]$Root = "out_sd3_v3",
    [string]$Prompt = "a photo",
    [string]$OutDir = "",
    [string]$Formats = "png,svg",
    [int]$Dpi = 300,
    [string]$LossImage = "suzume",
    [string]$LegacyLossRoot = "",
    [switch]$Clip,
    [switch]$Lpips,
    [switch]$ForceMetrics
)

$ErrorActionPreference = "Stop"

if (-not $OutDir) {
    $OutDir = Join-Path $Root "figures"
}

$Metrics = Join-Path $Root "full_metrics.csv"
$AnalysisDir = Join-Path $Root "analysis"

if ($ForceMetrics -or -not (Test-Path $Metrics)) {
    $computeArgs = @("code/metrics/compute_sd3_metrics.py", "--root", $Root, "--prompt", $Prompt, "--out", $Metrics)
    if ($Clip) { $computeArgs += "--clip" }
    if ($Lpips) { $computeArgs += "--lpips" }
    python @computeArgs
}

$hasRows = $false
if (Test-Path $Metrics) {
    $rows = Import-Csv $Metrics
    $hasRows = ($rows.Count -gt 0)
}

if ($hasRows) {
    python code/analysis/analyze_sd3_v3_results.py --metrics $Metrics --out-dir $AnalysisDir
} else {
    Write-Host "Skip analysis tables: metrics file is missing or has no data rows: $Metrics"
}

$drawArgs = @(
    "-m", "draw.figures.make_all_figures",
    "--root", $Root,
    "--metrics", $Metrics,
    "--analysis-dir", $AnalysisDir,
    "--out-dir", $OutDir,
    "--loss-image", $LossImage,
    "--formats", $Formats,
    "--dpi", $Dpi
)
if ($LegacyLossRoot) {
    $drawArgs += @("--legacy-loss-root", $LegacyLossRoot)
}
python @drawArgs

Write-Host "SD3 v3.1 figures written to $OutDir"
