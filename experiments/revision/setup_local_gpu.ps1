<# Create an isolated GPU environment and run the short actual-panel benchmark. #>
param(
    [string]$BasePython='E:\anaconda3\envs\common\python.exe',
    [switch]$SkipInstall
)
$ErrorActionPreference='Stop'
$repoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$venvPath=Join-Path $repoRoot '.venv-local-gpu'
$gpuPython=Join-Path $venvPath 'Scripts\python.exe'
function Invoke-Checked {
    param([string]$Executable,[string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable exited with $LASTEXITCODE" }
}
if (-not (Test-Path -LiteralPath $gpuPython -PathType Leaf)) {
    if (Test-Path -LiteralPath $venvPath) { throw 'Incomplete environment already exists; inspect it before retrying.' }
    Invoke-Checked $BasePython @('-m','venv','--system-site-packages',$venvPath)
}
# Use the official CUDA 12.8 wheel, with a Python 3.11-compatible ABI.
# The existing common environment is not changed.
if (-not $SkipInstall) {
    Invoke-Checked $gpuPython @('-m','pip','--disable-pip-version-check','install','torch==2.7.1',
        '--index-url','https://download.pytorch.org/whl/cu128')
}
# File-based entry points avoid Windows PowerShell 5.1 native argument quoting.
Invoke-Checked $gpuPython @((Join-Path $PSScriptRoot 'train_controlled.py'),'--check-environment')
Invoke-Checked $gpuPython @((Join-Path $PSScriptRoot 'train_controlled.py'),'--smoke','--device','cuda:0')
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$output=Join-Path $repoRoot ('artifacts\revision_20261114\local_gpu_benchmark_'+$stamp)
Invoke-Checked $gpuPython @((Join-Path $PSScriptRoot 'benchmark_local_gpu.py'),'--out',$output)
Write-Host "Benchmark completed: $output"
