# 创建桌面快捷方式
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\PDF OCR工具.lnk")
$Shortcut.TargetPath = "powershell.exe"
$Shortcut.Arguments = "-NoExit -Command `"cd 'E:\frame analysis'; conda activate deepseek-ocr2; python ocr_gui_modern.py`""
$Shortcut.WorkingDirectory = "E:\frame analysis"
$Shortcut.Save()
Write-Host "桌面快捷方式已创建: PDF OCR工具.lnk" -ForegroundColor Green
