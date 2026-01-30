@echo off
:: PDF OCR工具启动脚本
:: 双击此文件启动GUI界面

cd /d "%~dp0"

:: 激活conda环境
call C:\Users\Kevin\anaconda3\Scripts\activate.bat deepseek-ocr2

:: 运行GUI
python ocr_gui.py

pause
