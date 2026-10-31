@echo off
rem Stop starting the Hebrew TTS server at Windows sign-in.
powershell -NoProfile -Command "Remove-Item -ErrorAction SilentlyContinue ([Environment]::GetFolderPath('Startup') + '\Hebrew TTS.lnk')"
echo Removed. The server will no longer start automatically.
pause
