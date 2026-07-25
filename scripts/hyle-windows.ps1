param(
    [ValidateSet('Configure', 'Build', 'Test', 'Workflow', 'PackageSmoke')]
    [string]$Action = 'Configure',
    [string]$Preset = 'windows-ninja',
    [string]$SmokeRoot,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw 'vswhere.exe was not found. Install Visual Studio/Build Tools with the C++ workload.'
}

$installation = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if ($LASTEXITCODE -ne 0 -or -not $installation) {
    throw 'No Visual Studio installation with the x64 C++ toolchain was found.'
}

$vsDevCmd = Join-Path $installation 'Common7\Tools\VsDevCmd.bat'
if (-not (Test-Path -LiteralPath $vsDevCmd)) {
    throw "Visual Studio environment script was not found: $vsDevCmd"
}

# Import the selected Visual Studio environment into this PowerShell process so
# configure and Ninja builds see one coherent compiler/SDK contract. Nothing is
# written to the caller's persistent environment.
$environmentCommand = 'call "' + $vsDevCmd + '" -arch=x64 >nul && set'
$environmentLines = & $env:ComSpec /d /s /c $environmentCommand
if ($LASTEXITCODE -ne 0) {
    throw "Failed to initialize the Visual Studio environment from $vsDevCmd"
}
foreach ($line in $environmentLines) {
    $separator = $line.IndexOf('=')
    if ($separator -gt 0) {
        $name = $line.Substring(0, $separator)
        $value = $line.Substring($separator + 1)
        Set-Item -Path "Env:$name" -Value $value
    }
}

if ($Action -in @('Configure', 'Build', 'Workflow', 'PackageSmoke') -and
    -not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    throw 'The windows-ninja preset requires Ninja on PATH.'
}

switch ($Action) {
    'Configure' { & cmake --preset $Preset @ExtraArgs }
    'Build'     { & cmake --build --preset $Preset @ExtraArgs }
    'Test'      { & ctest --preset $Preset @ExtraArgs }
    'Workflow'  { & cmake --workflow --preset $Preset @ExtraArgs }
    'PackageSmoke' {
        if (-not $SmokeRoot) {
            throw 'Action PackageSmoke requires -SmokeRoot <scratch-directory>.'
        }
        $sourceDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
        $smokeScript = Join-Path $sourceDir 'cmake\HyleCiPackageSmoke.cmake'
        & cmake "-DHYLE_SOURCE_DIR=$sourceDir" "-DHYLE_SMOKE_ROOT=$SmokeRoot" `
            -DHYLE_SMOKE_GENERATOR=Ninja -P $smokeScript
    }
}
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
