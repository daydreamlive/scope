; Custom NSIS installer script for Daydream Scope
; This script downloads UV and runs uv sync during installation

!include "LogicLib.nsh"
!include "FileFunc.nsh"

; Define paths
!define UV_DOWNLOAD_URL "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"

; Helper macro to write log message to file
; Usage: !insertmacro WriteLog "message"
!macro WriteLog message
    FileOpen $R9 $R8 "a"
    FileWrite $R9 "${message}$\r$\n"
    FileClose $R9
!macroend

; Helper macro to write log message with timestamp
; Usage: !insertmacro WriteLogWithTime "message"
!macro WriteLogWithTime message
    ${GetTime} "" "L" $R0 $R1 $R2 $R3 $R4 $R5 $R6
    FileOpen $R9 $R8 "a"
    FileWrite $R9 "[$R2-$R1-$R0 $R4:$R5:$R6] ${message}$\r$\n"
    FileClose $R9
!macroend

; Called after installation is complete
!macro customInstall
    ; Get the AppData\Roaming path (equivalent to Electron's app.getPath('userData'))
    ReadEnvStr $0 APPDATA
    StrCpy $1 "$0\${PRODUCT_NAME}"

    ; Create directories
    CreateDirectory "$1"
    CreateDirectory "$1\uv"
    CreateDirectory "$1\python-project"
    CreateDirectory "$1\logs"

    ; Initialize log file
    StrCpy $R8 "$1\logs\installer.log"
    FileOpen $R9 $R8 "w"
    FileWrite $R9 "=== Daydream Scope Installer Log ===$\r$\n"
    FileClose $R9
    !insertmacro WriteLogWithTime "Installer started"

    ; Show progress
    DetailPrint "Setting up Python environment..."
    !insertmacro WriteLogWithTime "Setting up Python environment..."

    ; Download UV with retries
    DetailPrint "Downloading UV package manager..."
    !insertmacro WriteLogWithTime "Downloading UV package manager from ${UV_DOWNLOAD_URL}"

    ; Create temp directory for download
    GetTempFileName $2
    Delete $2
    CreateDirectory $2
    StrCpy $3 "$2\uv.zip"

    ; Retry download up to 3 times
    StrCpy $R0 0 ; retry counter
    StrCpy $R1 3 ; max retries
    download_retry_loop:
        IntOp $R0 $R0 + 1
        ${If} $R0 > 1
            DetailPrint "Retrying UV download (attempt $R0/$R1)..."
            !insertmacro WriteLogWithTime "Retrying UV download (attempt $R0/$R1)..."
            ; Wait before retry (exponential backoff: 2s, 4s)
            IntOp $R2 $R0 - 1
            IntOp $R2 $R2 * 2000
            Sleep $R2
            ; Clean up failed download if it exists
            IfFileExists "$3" 0 +2
                Delete "$3"
        ${EndIf}

        ; Use inetc plugin to download (bundled with electron-builder NSIS)
        DetailPrint "Downloading UV (attempt $R0/$R1)..."
        inetc::get /NOCANCEL /CAPTION "Downloading UV..." "${UV_DOWNLOAD_URL}" "$3" /END
        Pop $4

        ${If} $4 == "OK"
            ; Download successful, verify file exists
            IfFileExists "$3" download_verify 0
            ; File doesn't exist, check if we should retry
            ${If} $R0 < $R1
                !insertmacro WriteLogWithTime "Download file not found, retrying..."
                Goto download_retry_loop
            ${EndIf}
            DetailPrint "Warning: Downloaded UV file not found. UV will be downloaded on first app launch."
            !insertmacro WriteLogWithTime "ERROR: Downloaded UV file not found after $R0 attempts"
            Goto cleanup_download
        ${Else}
            ; Download failed, check if we should retry
            ${If} $R0 < $R1
                !insertmacro WriteLogWithTime "Download failed with status $4, retrying..."
                Goto download_retry_loop
            ${EndIf}
            DetailPrint "Warning: Failed to download UV ($4) after $R1 attempts. UV will be downloaded on first app launch."
            !insertmacro WriteLogWithTime "ERROR: Failed to download UV after $R1 attempts - Status: $4"
            Goto cleanup_download
        ${EndIf}

    download_verify:

    ; Verify download file exists and has size > 0
    IfFileExists "$3" 0 download_verify_failed
    FileOpen $R9 "$3" "r"
    FileSeek $R9 0 END $R0
    FileClose $R9
    ${If} $R0 == 0
        DetailPrint "Warning: Downloaded UV file is empty. UV will be downloaded on first app launch."
        !insertmacro WriteLogWithTime "ERROR: Downloaded UV file is empty (size: 0 bytes)"
        Goto cleanup_download
    ${EndIf}
    !insertmacro WriteLogWithTime "UV download successful - File size: $R0 bytes"
    Goto download_verified

    download_verify_failed:
    DetailPrint "Warning: Downloaded UV file not found. UV will be downloaded on first app launch."
    !insertmacro WriteLogWithTime "ERROR: Downloaded UV file not found"
    Goto cleanup_download

    download_verified:
    DetailPrint "Extracting UV..."
    !insertmacro WriteLogWithTime "Extracting UV archive..."

    ; Extract using PowerShell (more reliable than NSIS plugins)
    nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path \"$3\" -DestinationPath \"$2\extract\" -Force"'
    Pop $4

    ${If} $4 != 0
        DetailPrint "Warning: Failed to extract UV. UV will be downloaded on first app launch."
        !insertmacro WriteLogWithTime "ERROR: Failed to extract UV - Exit code: $4"
        Goto cleanup_download
    ${EndIf}
    !insertmacro WriteLogWithTime "UV archive extracted successfully"

    ; Find and copy uv.exe (it's in a subdirectory like uv-x86_64-pc-windows-msvc)
    FindFirst $5 $6 "$2\extract\*"
    ${DoWhile} $6 != ""
        ${If} $6 != "."
        ${AndIf} $6 != ".."
            ${If} ${FileExists} "$2\extract\$6\uv.exe"
                CopyFiles /SILENT "$2\extract\$6\uv.exe" "$1\uv\uv.exe"
                DetailPrint "UV installed successfully"
                !insertmacro WriteLogWithTime "UV installed successfully from subdirectory: $6"
                FindClose $5
                Goto found_uv
            ${EndIf}
        ${EndIf}
        FindNext $5 $6
    ${Loop}
    FindClose $5

    ; Check if uv.exe is directly in extract folder
    IfFileExists "$2\extract\uv.exe" 0 uv_not_found
        CopyFiles /SILENT "$2\extract\uv.exe" "$1\uv\uv.exe"
        DetailPrint "UV installed successfully"
        !insertmacro WriteLogWithTime "UV installed successfully from root extract folder"
        Goto found_uv

    uv_not_found:
    DetailPrint "Warning: Could not find uv.exe in archive. UV will be downloaded on first app launch."
    !insertmacro WriteLogWithTime "ERROR: Could not find uv.exe in archive"
    Goto cleanup_download

    found_uv:
    ; Verify uv.exe exists and is accessible
    IfFileExists "$1\uv\uv.exe" 0 uv_verify_failed
    ; Try to get file size to verify it's not corrupted
    FileOpen $R9 "$1\uv\uv.exe" "r"
    FileSeek $R9 0 END $R0
    FileClose $R9
    ${If} $R0 == 0
        DetailPrint "Warning: uv.exe file is empty or corrupted."
        !insertmacro WriteLogWithTime "ERROR: uv.exe file is empty (size: 0 bytes)"
        Goto cleanup_download
    ${EndIf}
    !insertmacro WriteLogWithTime "UV executable verified - Size: $R0 bytes, Path: $1\uv\uv.exe"
    Goto uv_verified

    uv_verify_failed:
    DetailPrint "Warning: uv.exe not found after installation."
    !insertmacro WriteLogWithTime "ERROR: uv.exe not found at expected path: $1\uv\uv.exe"
    Goto cleanup_download

    uv_verified:

    ; Copy Python project files from resources to userData
    DetailPrint "Copying Python project files..."
    !insertmacro WriteLogWithTime "Copying Python project files..."

    ; Source is in the installed resources folder
    StrCpy $7 "$INSTDIR\resources"

    ; Copy Python project files to userData
    IfFileExists "$7\pyproject.toml" 0 skip_python_copy
        CopyFiles /SILENT "$7\src" "$1\python-project\src"
        CopyFiles /SILENT "$7\pyproject.toml" "$1\python-project\pyproject.toml"
        CopyFiles /SILENT "$7\uv.lock" "$1\python-project\uv.lock"
        CopyFiles /SILENT "$7\.python-version" "$1\python-project\.python-version"
        CopyFiles /SILENT "$7\README.md" "$1\python-project\README.md"
        CopyFiles /SILENT "$7\LICENSE.md" "$1\python-project\LICENSE.md"
        IfFileExists "$7\frontend" 0 +2
            CopyFiles /SILENT "$7\frontend" "$1\python-project\frontend"
        DetailPrint "Python project files copied"
        !insertmacro WriteLogWithTime "Python project files copied successfully"
    skip_python_copy:

    ; Run uv sync to install Python dependencies
    DetailPrint "Installing Python dependencies (this may take a few minutes)..."
    !insertmacro WriteLogWithTime "Starting uv sync..."

    IfFileExists "$1\uv\uv.exe" 0 skip_uv_sync
    IfFileExists "$1\python-project\pyproject.toml" 0 skip_uv_sync

    ; Create log file for uv sync output
    StrCpy $R7 "$1\logs\uv-sync.log"
    FileOpen $R9 $R7 "w"
    FileWrite $R9 "=== UV Sync Execution Log ===$\r$\n"
    FileWrite $R9 "Command: $\"$1\uv\uv.exe$\" sync$\r$\n"
    FileWrite $R9 "Working Directory: $1\python-project$\r$\n"
    FileWrite $R9 "UV Executable Path: $1\uv\uv.exe$\r$\n"
    FileClose $R9

    ; Set working directory
    SetOutPath "$1\python-project"
    !insertmacro WriteLogWithTime "Working directory set to: $1\python-project"

    ; Verify uv.exe is accessible before running
    IfFileExists "$1\uv\uv.exe" 0 uv_not_accessible
    !insertmacro WriteLogWithTime "UV executable found at: $1\uv\uv.exe"
    Goto uv_accessible

    uv_not_accessible:
    DetailPrint "Warning: uv.exe not accessible. Skipping uv sync."
    !insertmacro WriteLogWithTime "ERROR: uv.exe not accessible at: $1\uv\uv.exe"
    Goto skip_uv_sync

    uv_accessible:
    ; Run uv sync with output redirected to log file
    ; Create a temporary batch file for reliable execution
    !insertmacro WriteLogWithTime "Executing uv sync command..."
    DetailPrint "Running uv sync (output will be logged to uv-sync.log)..."

    ; Create temporary batch file to execute uv sync
    GetTempFileName $R6
    Delete $R6
    StrCpy $R6 "$R6.bat"

    ; Write batch file content
    ; Use >> to append to the log file (headers already written)
    FileOpen $R9 $R6 "w"
    FileWrite $R9 "@echo off$\r$\n"
    FileWrite $R9 "cd /d $\"$1\python-project$\"$\r$\n"
    FileWrite $R9 "$\"$1\uv\uv.exe$\" sync >> $\"$R7$\" 2>&1$\r$\n"
    FileWrite $R9 "exit /b %ERRORLEVEL%$\r$\n"
    FileClose $R9

    !insertmacro WriteLog "Created batch file: $R6"
    !insertmacro WriteLog "UV executable: $1\uv\uv.exe"
    !insertmacro WriteLog "Working directory: $1\python-project"
    !insertmacro WriteLog "Log file: $R7"

    ; Execute batch file
    nsExec::ExecToLog '"$R6"'
    Pop $4

    ; Cleanup batch file
    Delete $R6
    !insertmacro WriteLog "Cleaned up batch file: $R6"

    ; Append exit code and timestamp to log
    FileOpen $R9 $R7 "a"
    FileWrite $R9 "$\r$\n=== Exit Code: $4 ===$\r$\n"
    ${GetTime} "" "L" $R0 $R1 $R2 $R3 $R4 $R5 $R6
    FileWrite $R9 "Completed at: $R2-$R1-$R0 $R4:$R5:$R6$\r$\n"
    FileClose $R9

    ; Also log to main installer log
    !insertmacro WriteLogWithTime "UV sync completed with exit code: $4"

    ; Read and append uv sync log to main installer log
    FileOpen $R9 $R7 "r"
    ${Do}
        FileRead $R9 $R0
        ${If} ${Errors}
            ${Break}
        ${EndIf}
        ; Remove trailing CRLF and append to main log
        StrCpy $R0 $R0 -2
        ${If} $R0 != ""
            !insertmacro WriteLog "[UV SYNC] $R0"
        ${EndIf}
    ${Loop}
    FileClose $R9

    ${If} $4 == 0
        DetailPrint "Python dependencies installed successfully"
        !insertmacro WriteLogWithTime "SUCCESS: Python dependencies installed successfully"
    ${Else}
        DetailPrint "Warning: uv sync failed (exit code: $4). Dependencies will be installed on first app launch."
        !insertmacro WriteLogWithTime "ERROR: uv sync failed with exit code: $4"
        !insertmacro WriteLog "Check $R7 for detailed uv sync output"
    ${EndIf}

    skip_uv_sync:

    cleanup_download:
    ; Cleanup temp files
    RMDir /r $2
    !insertmacro WriteLogWithTime "Cleaned up temporary download files"

    DetailPrint "Setup complete!"
    !insertmacro WriteLogWithTime "Installer completed successfully"
    !insertmacro WriteLog "Log files saved to: $1\logs\"
!macroend

; Called before uninstallation
!macro customUnInstall
    ; Optionally clean up UV and Python project on uninstall
    ; Note: We don't remove these by default to preserve user data
    ; Users can manually delete %APPDATA%\DaydreamScope if needed

    ; Show what will be kept
    ReadEnvStr $0 APPDATA
    DetailPrint "Note: Python environment in $0\${PRODUCT_NAME} will be preserved."
    DetailPrint "Delete this folder manually to fully remove all data."
!macroend
