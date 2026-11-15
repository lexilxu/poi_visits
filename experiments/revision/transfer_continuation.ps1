$ErrorActionPreference = 'Stop'
$target = 'root@connect.cqa1.seetacloud.com'
$remote = '/root/autodl-tmp/poi_continuation_20261114'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$payload = Join-Path $repo 'artifacts\revision_20261114\private_autodl_continuation_20261114'
& ssh -o BatchMode=yes -p 10150 $target "mkdir $remote"
if ($LASTEXITCODE -ne 0) { throw 'Fresh destination creation failed' }
& scp -o BatchMode=yes -P 10150 (Join-Path $payload 'transfer_manifest.json') (Join-Path $PSScriptRoot 'continue_autodl.py') "${target}:$remote/"
if ($LASTEXITCODE -ne 0) { throw 'Metadata transfer failed' }
& scp -o BatchMode=yes -P 10150 (Join-Path $payload 'continuation.tar') "${target}:$remote/"
if ($LASTEXITCODE -ne 0) { throw 'Archive transfer failed' }
& ssh -o BatchMode=yes -p 10150 $target "/root/miniconda3/bin/python $remote/continue_autodl.py prepare"
if ($LASTEXITCODE -eq 10) { Write-Output 'Full archive verified; the active remote queue continues.'; return }
if ($LASTEXITCODE -ne 0) { throw 'Remote verification failed' }
& ssh -o BatchMode=yes -p 10150 $target "cd $remote && nohup /root/miniconda3/bin/python -u continue_autodl.py run > continuation.log 2>&1 < /dev/null &"
if ($LASTEXITCODE -ne 0) { throw 'Launch failed' }
