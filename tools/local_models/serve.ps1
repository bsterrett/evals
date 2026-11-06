param(
    [Parameter(Mandatory = $true)][ValidateSet("huginn", "ouro", "ouro-1.4b-thinking", "ouro-2.6b-thinking", "dory")][string]$Profile,
    [Parameter(Mandatory = $true)][int]$Depth,
    [int]$Port = 8010
)
$ErrorActionPreference = "Stop"
$repoPath = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$windowsRepo = $repoPath.Replace('\', '/')
$linuxRepo = [string](& wsl -d Ubuntu-LLM -- wslpath -a $windowsRepo)
if ($LASTEXITCODE -ne 0) { throw "Could not resolve repository in Ubuntu-LLM" }
$linuxRepo = $linuxRepo.Trim()
$runtimeName = if ($Profile -in @("ouro-1.4b-thinking", "ouro-2.6b-thinking")) { "ouro" } else { $Profile }
& wsl -d Ubuntu-LLM -- "/home/ben/llm/recurrent/envs/$runtimeName/bin/python" -u -B "$linuxRepo/tools/local_models/serve.py" --profile $Profile --depth $Depth --port $Port
exit $LASTEXITCODE
