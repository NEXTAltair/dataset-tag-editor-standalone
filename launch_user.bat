@echo off

set COMMANDLINE_ARGS=

:: venvのactivate
call ".\venv\Scripts\activate.bat"

:: パッケージ経由で実行
dataset-tag-editor %COMMANDLINE_ARGS%