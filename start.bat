@echo off
echo ======================================
echo    视力检测后端服务 - 一键启动
echo ======================================
echo.

echo 正在启动虚拟环境...
call .venv\Scripts\activate.bat

echo.
echo 正在启动服务...
echo 服务地址：http://127.0.0.1:8090
echo.

python server_v12.py

pause