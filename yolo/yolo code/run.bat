@echo off
REM ============================================================
REM  run.bat —— 用 yolo 环境运行项目脚本
REM
REM  用法 (在项目目录下):
REM    run.bat yolo_detect.py
REM    run.bat yolo_detect.py --source clips/ball.mp4
REM    run.bat extract_frames.py "iron ball video.mp4"
REM
REM  原理: 直接指定 yolo 环境的 Python 解释器,
REM        不依赖当前激活的是哪个环境, 永远不会选错。
REM ============================================================

set "ENVPY=C:\Users\asus\anaconda3\envs\yolo\python.exe"

if not exist "%ENVPY%" (
    echo [错误] 找不到 yolo 环境的 Python: %ENVPY%
    echo 请检查 Anaconda 安装位置是否正确。
    pause
    exit /b 1
)

if "%~1"=="" (
    echo [用法] run.bat 脚本名 [参数...]
    echo   例如: run.bat yolo_detect.py
    pause
    exit /b 1
)

"%ENVPY%" "%~dp0%~1" %2 %3 %4 %5 %6 %7 %8 %9
pause
