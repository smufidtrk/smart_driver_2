@echo off
rem Runs ShelfCapture with the user-local .NET 8 SDK (%USERPROFILE%\.dotnet).
rem The system "dotnet" on PATH has runtimes only and cannot build.
rem Usage:  run.cmd --epc <TEST_TAG_EPC> --session van1_grid   (same args as "dotnet run --")
set DOTNET_ROOT=%USERPROFILE%\.dotnet
set DOTNET_CLI_TELEMETRY_OPTOUT=1
set DOTNET_NOLOGO=1
pushd "%~dp0"
"%USERPROFILE%\.dotnet\dotnet.exe" run -- %*
popd
