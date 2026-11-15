$ErrorActionPreference = 'Stop'
$target = 'root@connect.cqa1.seetacloud.com'
$remote = '/root/autodl-tmp/poi_continuation_20261114'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$payload = Join-Path $repo 'artifacts\revision_20261114\private_autodl_continuation_20261114'
& scp -o BatchMode=yes -P 10150 (Join-Path $payload 'bootstrap.tar') (Join-Path $PSScriptRoot 'continue_autodl.py') "${target}:$remote/"
if ($LASTEXITCODE -ne 0) { throw 'Bootstrap transfer failed' }
& ssh -o BatchMode=yes -p 10150 $target "/root/miniconda3/bin/python $remote/continue_autodl.py bootstrap"
if ($LASTEXITCODE -ne 0) { throw 'Bootstrap verification failed' }
& ssh -o BatchMode=yes -p 10150 $target "cd $remote && nohup /root/miniconda3/bin/python -u continue_autodl.py run > continuation.log 2>&1 < /dev/null &"
