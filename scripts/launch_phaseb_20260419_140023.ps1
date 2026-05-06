$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'C:\Users\30727\isaac-sim'
& .\\python.bat my_projects\\train_phaseb_hybrid.py 
  --resume-from "C:\Users\30727\isaac-sim\my_projects\models\my_underwater_robot_policy_phaseb_from566k_alpha80_ladder2p.zip" 
  --timesteps 1000000 
  --learning-rate 1e-4 
  --tb-run-name "fish_vla_phaseb_from566k_20260419_140023" 
  --model-output-path "my_projects\models\my_underwater_robot_policy_phaseb_from566k_20260419_140023" 
  --monitor-log-path "my_projects\monitor_logs\monitor_phaseb_from566k_20260419_140023.monitor.csv" 
  --alpha-start 0.8 
  --progress-bar 
  --renderer HydraStorm 2>&1 | Tee-Object -FilePath "my_projects\start_train_20260419_140023.console.log"
