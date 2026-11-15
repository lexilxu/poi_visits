$ErrorActionPreference = 'Stop'
$target = 'root@connect.cqa1.seetacloud.com'
$remote = '/root/autodl-tmp/poi_continuation_20261114'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$destination = Join-Path $repo "artifacts\revision_20261114\private_autodl_continuation_20261114\snapshot_$stamp"
New-Item -ItemType Directory -Path $destination | Out-Null
& scp -o BatchMode=yes -P 10150 (Join-Path $PSScriptRoot 'autodl_status.py') "${target}:$remote/autodl_status.py"
if ($LASTEXITCODE -ne 0) { throw 'Status script upload failed' }
$report = & ssh -o BatchMode=yes -p 10150 $target "/root/miniconda3/bin/python $remote/autodl_status.py"
if ($LASTEXITCODE -ne 0) { throw 'Status read failed' }
[IO.File]::WriteAllLines((Join-Path $destination 'status.json'),$report,[Text.UTF8Encoding]::new($false))
$report
& ssh -o BatchMode=yes -p 10150 $target "tail -6 $remote/analysis_v2.log; tail -4 $remote/january_v2.log; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader"
& scp -r -o BatchMode=yes -P 10150 "${target}:$remote/new_december_results" $destination
& scp -r -o BatchMode=yes -P 10150 "${target}:$remote/new_december_results_v2" $destination
& scp -o BatchMode=yes -P 10150 "${target}:$remote/training_completion_audit.json" "${target}:$remote/density_evaluation_correction.json" $destination
Write-Output "Snapshot: $destination"
