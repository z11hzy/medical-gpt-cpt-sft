param([switch]$PrepareOnly)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\set_env.ps1"
$python = Join-Path $root ".venv\Scripts\python.exe"
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUNBUFFERED = "1"
$clean = Join-Path $root "data\processed\pretrain-clean-20k-v2\manifest.json"
if (!(Test-Path $clean)) {
    & $python "$root\data_pipeline\clean_cpt_20k.py"
    if ($LASTEXITCODE -ne 0) { throw "数据准备失败" }
}
foreach ($phase in @("base", "cpt")) {
    & $python "$root\tools\run_cpt_stage.py" $phase --check-only
    if ($LASTEXITCODE -ne 0) { throw "参数校验失败：$phase" }
}
if ($PrepareOnly) { return }
foreach ($phase in @("base", "cpt")) {
    Write-Host "Starting $phase (完整验证集，不截断训练数据)"
    & $python "$root\tools\run_cpt_stage.py" $phase
    if ($LASTEXITCODE -ne 0) { throw "实验失败：$phase；旧结果和检查点保留" }
}
& $python "$root\tools\show_cpt_results.py" --save
if ($LASTEXITCODE -ne 0) { throw "结果对比失败" }
