@echo off
rem Start the Hebrew TTS server automatically at Windows sign-in (a minimized window).
rem Undo with remove_autostart.bat.
powershell -NoProfile -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Startup') + '\Hebrew TTS.lnk'); $s.TargetPath = '%~dp0start.bat'; $s.WorkingDirectory = '%~dp0'; $s.WindowStyle = 7; $s.Save()"
echo The server will now start automatically when you sign in to Windows.
pause
