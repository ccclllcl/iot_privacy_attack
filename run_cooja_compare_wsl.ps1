param(
  [string]$WslDistro = "Ubuntu-22.04",
  [string]$WslDir = "\\wsl$\Ubuntu-22.04\home\linchen\iot-privacy-project",
  [string]$BaselineRadio = "Radiomsg.txt",
  [string]$BaselineApp = "loglistener.txt",
  [string]$DefenseRadio = "Radiomsg_defense.txt",
  [string]$DefenseApp = "loglistener_defense.txt",
  [string]$OutDir = "outputs/cooja_compare",
  [string]$Seeds = "42,123,2026",
  [double]$WindowS = 8.0,
  [double]$StepS = 3.0,
  [int]$MinRequests = 2,
  [double]$DominanceThreshold = 0.2
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$baselineRadioPath = Join-Path $WslDir $BaselineRadio
$baselineAppPath = Join-Path $WslDir $BaselineApp
$defenseRadioPath = Join-Path $WslDir $DefenseRadio
$defenseAppPath = Join-Path $WslDir $DefenseApp

foreach ($p in @($baselineRadioPath, $baselineAppPath, $defenseRadioPath, $defenseAppPath)) {
  if (-not (Test-Path $p)) {
    throw "Missing file: $p"
  }
}

py -3 run_cooja_compare.py `
  --baseline_radio_log "$baselineRadioPath" `
  --baseline_app_log "$baselineAppPath" `
  --defense_radio_log "$defenseRadioPath" `
  --defense_app_log "$defenseAppPath" `
  --out_dir "$OutDir" `
  --window_s $WindowS `
  --step_s $StepS `
  --min_requests $MinRequests `
  --dominance_threshold $DominanceThreshold `
  --seeds "$Seeds"

Write-Host ""
Write-Host "Done. Results at: $OutDir"
Write-Host "- compare_report.json"
Write-Host "- accuracy_f1_by_seed.png"
