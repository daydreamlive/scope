; Custom NSIS installer script for Daydream Scope
; This script downloads UV and runs uv sync during installation

!include "LogicLib.nsh"
!include "FileFunc.nsh"

; Define paths
!define UV_DOWNLOAD_URL "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"

; Called after installation is complete
!macro customInstall
    ; Get the AppData\Roaming path (equivalent to Electron's app.getPath('userData'))
    ReadEnvStr $0 APPDATA
    StrCpy $1 "$0\${PRODUCT_NAME}"

    ; Create directories
    CreateDirectory "$1"
    CreateDirectory "$1\uv"
    CreateDirectory "$1\python-project"

    ; Show progress
    DetailPrint "Setting up Python environment..."

    ; Download UV
    DetailPrint "Downloading UV package manager..."

    ; Create temp directory for download
    GetTempFileName $2
    Delete $2
    CreateDirectory $2
    StrCpy $3 "$2\uv.zip"

    ; Use inetc plugin to download (bundled with electron-builder NSIS)
    inetc::get /NOCANCEL /CAPTION "Downloading UV..." "${UV_DOWNLOAD_URL}" "$3" /END
    Pop $4

    ${If} $4 != "OK"
        DetailPrint "Warning: Failed to download UV ($4). UV will be downloaded on first app launch."
        Goto cleanup_download
    ${EndIf}

    DetailPrint "Extracting UV..."

    ; Extract using PowerShell (more reliable than NSIS plugins)
    nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path \"$3\" -DestinationPath \"$2\extract\" -Force"'
    Pop $4

    ${If} $4 != 0
        DetailPrint "Warning: Failed to extract UV. UV will be downloaded on first app launch."
        Goto cleanup_download
    ${EndIf}

    ; Find and copy uv.exe (it's in a subdirectory like uv-x86_64-pc-windows-msvc)
    FindFirst $5 $6 "$2\extract\*"
    ${DoWhile} $6 != ""
        ${If} $6 != "."
        ${AndIf} $6 != ".."
            IfFileExists "$2\extract\$6\uv.exe" 0 +3
                CopyFiles /SILENT "$2\extract\$6\uv.exe" "$1\uv\uv.exe"
                DetailPrint "UV installed successfully"
                Goto found_uv
        ${EndIf}
        FindNext $5 $6
    ${Loop}
    FindClose $5

    ; Check if uv.exe is directly in extract folder
    IfFileExists "$2\extract\uv.exe" 0 uv_not_found
        CopyFiles /SILENT "$2\extract\uv.exe" "$1\uv\uv.exe"
        DetailPrint "UV installed successfully"
        Goto found_uv

    uv_not_found:
    DetailPrint "Warning: Could not find uv.exe in archive. UV will be downloaded on first app launch."
    Goto cleanup_download

    found_uv:

    ; Copy Python project files from resources to userData
    DetailPrint "Copying Python project files..."

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
    skip_python_copy:

    ; Run uv sync to install Python dependencies
    DetailPrint "Installing Python dependencies (this may take a few minutes)..."

    IfFileExists "$1\uv\uv.exe" 0 skip_uv_sync
    IfFileExists "$1\python-project\pyproject.toml" 0 skip_uv_sync

    ; Run uv sync
    SetOutPath "$1\python-project"
    nsExec::ExecToLog '"$1\uv\uv.exe" sync'
    Pop $4

    ${If} $4 == 0
        DetailPrint "Python dependencies installed successfully"
    ${Else}
        DetailPrint "Warning: uv sync failed (exit code: $4). Dependencies will be installed on first app launch."
    ${EndIf}

    skip_uv_sync:

    cleanup_download:
    ; Cleanup temp files
    RMDir /r $2

    DetailPrint "Setup complete!"
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
