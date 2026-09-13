# Multi-AMR Hybrid Communications Simulation Launcher (PowerShell)
Clear-Host
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "        DECENTRALIZED MULTI-AMR HYBRID COMMS SIMULATION" -ForegroundColor White
Write-Host "  Dual Transport: High-Speed Wi-Fi (10 Hz) + Wi-SUN Sub-GHz Mesh (2 Hz)" -ForegroundColor Yellow
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Select presentation mode:"
Write-Host "  [1] Launch Live Interactive GUI Visualizer (Warehouse Map + Mission HUD)" -ForegroundColor Green
Write-Host "  [2] Run Terminal Simulation with Network KPI Telemetry Logs" -ForegroundColor Yellow
Write-Host "  [3] Generate Fresh Presentation Screenshot (PNG)" -ForegroundColor Magenta
Write-Host ""

$choice = Read-Host "Enter option (1, 2, or 3) [default: 1]"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

$pythonExe = ".\.venv\Scripts\python.exe"

switch ($choice) {
    "1" {
        Write-Host "`n[*] Starting Live Animated GUI Visualizer..." -ForegroundColor Green
        Write-Host "[*] Controls: Press [SPACE] to Pause/Resume, [R] to Restart.`n" -ForegroundColor DarkCyan
        & $pythonExe demo_visualizer.pyc
    }
    "2" {
        Write-Host "`n[*] Executing Multi-AMR Simulation Engine...`n" -ForegroundColor Yellow
        & $pythonExe Final_Simulation.py
    }
    "3" {
        Write-Host "`n[*] Generating high-resolution presentation screenshot...`n" -ForegroundColor Magenta
        & $pythonExe demo_visualizer.pyc presentation_snapshot.png
        Write-Host "`n[+] Saved to presentation_snapshot.png" -ForegroundColor Green
    }
    Default {
        Write-Host "Invalid choice." -ForegroundColor Red
    }
}
