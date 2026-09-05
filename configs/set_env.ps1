$ErrorActionPreference = "Stop"

$root = (Resolve-Path "$PSScriptRoot\..").Path
$env:HF_HOME = Join-Path $root ".cache\huggingface"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:HF_DATASETS_CACHE = Join-Path $env:HF_HOME "datasets"
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME "transformers"
$env:TORCH_HOME = Join-Path $root ".cache\torch"
$env:PIP_CACHE_DIR = Join-Path $root ".cache\pip"
$env:TEMP = Join-Path $root ".tmp"
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = "1"
$env:TOKENIZERS_PARALLELISM = "false"

@(
  $env:HF_HOME,
  $env:HF_HUB_CACHE,
  $env:HF_DATASETS_CACHE,
  $env:TRANSFORMERS_CACHE,
  $env:TORCH_HOME,
  $env:PIP_CACHE_DIR,
  $env:TEMP
) | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

Write-Host "Caches and temporary files are configured under $root"
