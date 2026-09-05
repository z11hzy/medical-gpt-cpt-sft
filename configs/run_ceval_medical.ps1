param(
  [string]$Adapter = "",
  [string]$Name = "base"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
. "$PSScriptRoot\set_env.ps1"
$python = Join-Path $root ".venv\Scripts\python.exe"
$arguments = @(
  "$root\eval\ceval_medical.py",
  "--model", "$root\models\Qwen2.5-0.5B",
  "--dataset-root", "$root\data\eval\ceval",
  "--output", "$root\reports\results\ceval-$Name.json"
)
if ($Adapter) { $arguments += @("--adapter", $Adapter) }
& $python @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
