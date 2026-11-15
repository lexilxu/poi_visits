<#
Run from the user's normal PowerShell session. This script does not alter network,
security, SSH, or Git settings. It checks the connection before transferring data,
uses a fresh versioned remote directory, and downloads results after the run.
The default budget covers one recipe: three LR trials plus four additional seeds.
#>
param(
    [ValidatePattern('^[A-Za-z0-9._-]+$')][string]$SshHost = 'autodl4',
    [string]$Python = 'python',
    [ValidateRange(0.1,24)][double]$MaxHours = 2,
    [ValidateRange(1,357)][int]$MaxNewFits = 7
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$artifactRoot = Join-Path $repoRoot 'artifacts\revision_20261114'
$panelPath = [IO.Path]::GetFullPath((Join-Path $repoRoot '..\paper_figures\mechanism_figures\formal_2020q4\approved_panel.zip'))
$sourcePath = Join-Path $artifactRoot 'training_source.zip'
$panelDigest = '12d6414990eeb5703b1179152b5993518dcfa68361e39e09ae621919950695ec'

function Invoke-Checked {
    param([string]$Executable,[string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable exited with code $LASTEXITCODE. No later step was started."
    }
}

if (-not (Test-Path -LiteralPath $panelPath -PathType Leaf)) { throw "Missing approved panel: $panelPath" }
if ((Get-FileHash -LiteralPath $panelPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $panelDigest) {
    throw 'Approved-panel checksum mismatch.'
}

Write-Host 'Checking the existing SSH connection and GPU. No settings will be changed.'
Invoke-Checked 'ssh' @('-o','ConnectTimeout=15',$SshHost,
    'hostname && nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader && python -c "import numpy,pandas,scipy,torch; print(torch.__version__); assert torch.cuda.is_available()"')
Invoke-Checked $Python @((Join-Path $PSScriptRoot 'package_training.py'),'--out',$sourcePath)
$sourceDigest = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
$remoteRoot = '/root/autodl-tmp/poi_revision_' + $sourceDigest.Substring(0,12)
$localResults = Join-Path $artifactRoot ('private_remote_runs\' + $sourceDigest.Substring(0,12))
New-Item -ItemType Directory -Force -Path $localResults | Out-Null

Invoke-Checked 'ssh' @($SshHost,('mkdir -p ' + $remoteRoot))
Invoke-Checked 'scp' @($sourcePath,($SshHost + ':' + $remoteRoot + '/training_source.zip'))
Invoke-Checked 'scp' @($panelPath,($SshHost + ':' + $remoteRoot + '/approved_panel.zip'))

$remoteScript = @'
set -euo pipefail
cd "$1"
python - "$2" "$3" <<'PY'
from pathlib import Path
from hashlib import sha256
import sys
for name,expected in zip(('training_source.zip','approved_panel.zip'),sys.argv[1:]):
    actual=sha256(Path(name).read_bytes()).hexdigest()
    if actual!=expected:
        raise SystemExit('Checksum mismatch: '+name)
print('Both uploaded archives verified',flush=True)
PY
python -m zipfile -e training_source.zip .
python experiments/revision/train_controlled.py --check-environment
python experiments/revision/training_queue.py --stage core --execute \
  --bundle approved_panel.zip --out private_runs/core --max-hours "$4" \
  --max-new-fits "$5" 2>&1 | tee queue_console.log
'@
$hoursText = $MaxHours.ToString([Globalization.CultureInfo]::InvariantCulture)
Write-Host "Starting the bounded training queue in $remoteRoot"
$remoteScript.Replace("`r`n","`n") | & ssh $SshHost bash -s -- $remoteRoot $sourceDigest $panelDigest $hoursText $MaxNewFits
$trainingExitCode = $LASTEXITCODE

Write-Host 'Downloading any completed checkpoints, predictions, and queue logs.'
& scp -r ($SshHost + ':' + $remoteRoot + '/private_runs/core') $localResults
$resultsExitCode = $LASTEXITCODE
& scp ($SshHost + ':' + $remoteRoot + '/queue_console.log') $localResults
$logExitCode = $LASTEXITCODE
$record = [ordered]@{
    ssh_host=$SshHost; remote_directory=$remoteRoot; source_sha256=$sourceDigest;
    panel_sha256=$panelDigest; training_exit_code=$trainingExitCode;
    result_download_exit_code=$resultsExitCode; log_download_exit_code=$logExitCode;
    max_hours=$MaxHours; max_new_fits=$MaxNewFits;
    instance_power_state='Not changed by this script'
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $localResults 'launcher_status.json') -Encoding UTF8
Write-Host "Results and status: $localResults"
Write-Host 'Queue stopped. This script leaves the instance power state unchanged.'
if ($trainingExitCode -ne 0) {
    Write-Warning 'The queue stopped at its budget or an error. Read queue_status.json and queue_console.log before another run.'
}
if ($resultsExitCode -ne 0 -or $logExitCode -ne 0) {
    throw "Some downloads failed. Remote files remain in $remoteRoot."
}
