@echo off
rem Starts the Hebrew TTS server: web UI at http://localhost:8010 (and http://<this-PC-IP>:8010
rem on the home network without a key; the key from api_key.txt is needed only from outside).
rem Extra arguments pass through, e.g.  start.bat --port 8011  or  start.bat --no-auth
rem Zonos-Hebrew (optional) is turned on in the web UI: Server status -> Zonos-Hebrew.
title Hebrew TTS server
cd /d "%~dp0"
"%USERPROFILE%\.conda\envs\voice\python.exe" server.py --host 0.0.0.0 %*
pause
