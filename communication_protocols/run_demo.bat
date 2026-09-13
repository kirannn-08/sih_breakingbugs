@echo off
title Multi-AMR Hybrid Communications Simulation
cls
echo ============================================================================
echo         DECENTRALIZED MULTI-AMR HYBRID COMMS SIMULATION
echo   Dual Transport: High-Speed Wi-Fi (10 Hz) + Wi-SUN Sub-GHz Mesh (2 Hz)
echo ============================================================================
echo.
echo Launching Live Interactive Web Visualizer...
echo Server starting at http://localhost:8000/frontend/index.html
echo.
start http://localhost:8000/frontend/index.html
.\.venv\Scripts\python server.py

pause
