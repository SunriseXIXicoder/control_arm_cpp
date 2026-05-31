$ErrorActionPreference = "Stop"
$host.UI.RawUI.WindowTitle = "EMsFEM ANN 5x5x5 load-tuned 40x20x8 250"

Set-Location -LiteralPath $PSScriptRoot

$prefix = "result/ems_ann_5x5x5_loadtuned_40_20_8_np2_250"
$wslRoot = "/mnt/c/piml_petsc_work/control_arm_cpp"

$bashCommand = "cd `"$wslRoot`" && mkdir -p result && LOG=result/ems_ann_5x5x5_loadtuned_40_20_8_np2_250_launcher.log && : > `"`$LOG`" && exec > >(tee -a `"`$LOG`") 2>&1 && EMS_SUB_N=5 ANN_DIR=../input_5 NELX=40 NELY=20 NELZ=8 ITERS=250 NP=2 WRITE_VTK=true PREFIX=result/ems_ann_5x5x5_loadtuned_40_20_8_np2_250 sh run_wsl_ems_ann_optimize_40_20_8.sh"

Write-Host "Starting local load-tuned EMsFEM/ANN 5x5x5 optimization..."
Write-Host "Coarse grid: 40 x 20 x 8, iterations: 250, MPI ranks: 2"
Write-Host "Result prefix: $prefix"
Write-Host ""

wsl -e bash -lc $bashCommand

Write-Host ""
Write-Host "Run finished or stopped."
Write-Host "Final VTK: $prefix`_final.vtk"
Write-Host "History CSV: $prefix`_history.csv"
Write-Host "Summary: $prefix`_opt_summary.txt"
Read-Host "Press Enter to close this window"
