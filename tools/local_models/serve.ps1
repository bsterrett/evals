param(
    [Parameter(Mandatory = $true)][ValidateSet("huginn", "ouro", "ouro-1.4b-thinking", "ouro-2.6b-thinking", "dory", "qwen3-1.7b", "qwen3-4b-instruct", "qwen3-1.7b-base", "qwen3-1.7b-loopus")][string]$Profile,
    [Parameter(Mandatory = $true)][int]$Depth,
    [int]$Port = 8010,
    # Retrofit profiles only: the design to install, for example 'block[9:18]',
    # 'block[15:19]/euler,anchor=0.5' or 'recirculate[14to7]/convex,alpha=0.07'.
    # Omit to serve the plain parent at depth 1. The LoopUS checkpoint always
    # takes its own design, 'gated[2:27]'.
    [ValidatePattern('^((block|layer|gated)\[\d+:\d+\](/[a-z]+(,[a-z]+=[A-Za-z0-9.]+)*)?|recirculate\[\d+to\d+\]/[a-z]+,alpha=[0-9.]+)$')][string]$Method
)
$ErrorActionPreference = "Stop"
$repoPath = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$windowsRepo = $repoPath.Replace('\', '/')
$linuxRepo = [string](& wsl -d Ubuntu-LLM -- wslpath -a $windowsRepo)
if ($LASTEXITCODE -ne 0) { throw "Could not resolve repository in Ubuntu-LLM" }
$linuxRepo = $linuxRepo.Trim()
$isRetrofitParent = $Profile -like "qwen3-*"
if ($Method -and -not $isRetrofitParent) { throw "-Method applies to retrofit parent profiles only" }
$runtimeName = if ($Profile -in @("ouro-1.4b-thinking", "ouro-2.6b-thinking")) { "ouro" } elseif ($isRetrofitParent) { "retrofit" } else { $Profile }
$extra = @()
# The design is passed as --method=<design> with the brackets escaped, because the
# WSL launcher hands the command line to a shell that would treat them as a glob.
if ($Method) { $extra += "--method=" + $Method.Replace('[', '\[').Replace(']', '\]') }
& wsl -d Ubuntu-LLM -- "/home/ben/llm/recurrent/envs/$runtimeName/bin/python" -u -B "$linuxRepo/tools/local_models/serve.py" --profile $Profile --depth $Depth --port $Port @extra
exit $LASTEXITCODE
