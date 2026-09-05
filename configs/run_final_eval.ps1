$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
. "$PSScriptRoot\set_env.ps1"
$python = Join-Path $root ".venv\Scripts\python.exe"
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$model = Join-Path $root "models\Qwen2.5-0.5B"
$adapter = Join-Path $root "outputs\pt-full20k-0.5b-v2"
$test = Join-Path $root "data\raw\shibing624-medical\pretrain\test_encyclopedia.json"

& $python "$root\eval\test_lm.py" --model $model --test-file $test --output "$root\reports\results\test-lm-base.json"
if ($LASTEXITCODE -ne 0) { throw "Base test evaluation failed" }
& $python "$root\eval\test_lm.py" --model $model --adapter $adapter --test-file $test --output "$root\reports\results\test-lm-cpt-full20k.json"
if ($LASTEXITCODE -ne 0) { throw "CPT test evaluation failed" }
& "$PSScriptRoot\run_ceval_medical.ps1" -Name "base-final"
if ($LASTEXITCODE -ne 0) { throw "Base C-Eval failed" }
& "$PSScriptRoot\run_ceval_medical.ps1" -Adapter $adapter -Name "cpt-full20k"
if ($LASTEXITCODE -ne 0) { throw "CPT C-Eval failed" }
& $python "$root\tools\show_final_eval_results.py" --save
if ($LASTEXITCODE -ne 0) { throw "Result summary failed" }
