[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet('Configure', 'Build', 'Test', 'Workflow', 'ComponentContract', 'SelectedComponentSmoke', 'PackageSmoke', 'ProductShapeSmoke', 'SourcePackageSmoke')]
    [string]$Action = 'Configure',
    [string]$Preset = 'windows-ninja',
    [string]$BinaryDir,
    [string]$Component,
    [string]$Factor,
    [string]$SmokeRoot,
    [string]$DependencySourceCache,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'

$sourceDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ($DependencySourceCache) {
    if (-not [System.IO.Path]::IsPathFullyQualified($DependencySourceCache)) {
        throw '-DependencySourceCache must be an absolute path.'
    }
    # Child CMake configure processes use the same documented precedence as a
    # direct invocation. This avoids PowerShell treating a raw -D value that
    # contains a Windows drive colon as another script parameter.
    $env:HYLE_DEP_SOURCE_CACHE = [System.IO.Path]::GetFullPath($DependencySourceCache)
}
$resolvedBinaryDir = $null
if ($BinaryDir) {
    if ($Action -notin @('Configure', 'Build', 'Test')) {
        throw '-BinaryDir is supported only by Configure, Build, and Test.'
    }
    if (-not [System.IO.Path]::IsPathFullyQualified($BinaryDir)) {
        throw '-BinaryDir must be an absolute, session-owned scratch path.'
    }
    $resolvedBinaryDir = [System.IO.Path]::GetFullPath($BinaryDir)
}
if ($Component) {
    if ([string]::IsNullOrWhiteSpace($Component)) {
        throw '-Component cannot be empty.'
    }
    if ($Action -eq 'Configure' -and -not $resolvedBinaryDir) {
        throw '-Component requires Configure together with an explicit -BinaryDir.'
    }
    if ($Action -notin @('Configure', 'SelectedComponentSmoke')) {
        throw '-Component is supported only by isolated Configure or SelectedComponentSmoke.'
    }
}
if ($Factor) {
    if ([string]::IsNullOrWhiteSpace($Factor)) {
        throw '-Factor cannot be empty.'
    }
    if ($Component) {
        throw '-Factor and -Component are mutually exclusive.'
    }
    if ($Action -ne 'Configure' -or -not $resolvedBinaryDir) {
        throw '-Factor is supported only by isolated Configure with an explicit -BinaryDir.'
    }
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw 'vswhere.exe was not found. Install Visual Studio/Build Tools with the C++ workload.'
}

function Get-LatestSupportedVs2022Toolset {
    param([string]$VswherePath)

    $vs2022 = & $VswherePath -latest -version '[17.0,18.0)' -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $vs2022) {
        throw 'The windows preset needs Visual Studio 2022 with the x64 C++ workload.'
    }

    $toolsetRoot = Join-Path $vs2022 'VC\Tools\MSVC'
    $toolsets = @(Get-ChildItem -LiteralPath $toolsetRoot -Directory -ErrorAction SilentlyContinue |
        ForEach-Object {
            try {
                [pscustomobject]@{ Name = $_.Name; Version = [version]$_.Name }
            } catch {
                # Ignore non-version directories owned by Visual Studio extensions.
            }
        } |
        Where-Object { $_.Version -ge [version]'14.43' } |
        Sort-Object Version -Descending)
    if ($toolsets.Count -eq 0) {
        throw "Visual Studio 2022 at '$vs2022' has no MSVC 14.43+ toolset."
    }

    $selected = $toolsets[0]
    [pscustomobject]@{
        Installation = $vs2022
        Version = "$($selected.Version.Major).$($selected.Version.Minor)"
    }
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
# configure and Ninja builds see one coherent compiler/SDK contract. Reuse an
# already initialized environment for the same installation/architecture:
# repeatedly calling VsDevCmd prepends duplicate PATH entries until cmd.exe's
# 8191-character input limit is exceeded (notably in multi-command CI steps).
$selectedInstallation = [System.IO.Path]::GetFullPath($installation).TrimEnd('\')
$activeInstallation = if ($env:VSINSTALLDIR) {
    [System.IO.Path]::GetFullPath($env:VSINSTALLDIR).TrimEnd('\')
} else {
    ''
}
$hasSelectedEnvironment = $env:VSCMD_VER -and
    $env:VCINSTALLDIR -and
    $env:WindowsSdkDir -and
    $env:VSCMD_ARG_TGT_ARCH -eq 'x64' -and
    $activeInstallation -eq $selectedInstallation
if (-not $hasSelectedEnvironment) {
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
            # Windows environment keys are case-insensitive, but a parent
            # process can still hand cmd.exe duplicate PATH/Path entries.
            # VsDevCmd emits its toolchain-prefixed PATH and may then echo the
            # stale mixed-case entry; importing both makes PowerShell let the
            # latter erase cl.exe again. For PATH, accept only the initialized
            # value that contains this Visual Studio installation.
            if ($name -ieq 'Path') {
                if ($value -like "*$installation\VC\Tools\MSVC\*") {
                    Set-Item -Path 'Env:Path' -Value $value
                }
            } else {
                Set-Item -Path "Env:$name" -Value $value
            }
        }
    }
}

$effectiveExtraArgs = @($ExtraArgs)
$vs2022Toolset = $null
if (($Action -eq 'Configure' -and $Preset -eq 'windows') -or
    ($Action -eq 'Workflow' -and $Preset -eq 'windows-check')) {
    $hasExplicitToolset = $effectiveExtraArgs | Where-Object {
        $_ -eq '-T' -or $_ -eq '--toolset' -or $_ -like '-T*' -or $_ -like '--toolset=*'
    }
    if (-not $hasExplicitToolset) {
        $vs2022Toolset = Get-LatestSupportedVs2022Toolset -VswherePath $vswhere
        $toolsetArgument = "version=$($vs2022Toolset.Version)"
        Write-Host "HYLE Visual Studio solution: VS 2022 toolset $toolsetArgument"

        # CMake forbids changing a generator toolset in-place. Refresh only its
        # generated cache when an older/default v143 cache is already present;
        # sources, dependencies, build outputs and user files remain untouched.
        $cachePath = if ($resolvedBinaryDir) {
            Join-Path $resolvedBinaryDir 'CMakeCache.txt'
        } else {
            Join-Path $sourceDir 'build-windows\CMakeCache.txt'
        }
        if (Test-Path -LiteralPath $cachePath) {
            $cachedToolset = Get-Content -LiteralPath $cachePath |
                Where-Object { $_ -like 'CMAKE_GENERATOR_TOOLSET:INTERNAL=*' } |
                Select-Object -First 1
            if ($cachedToolset -ne "CMAKE_GENERATOR_TOOLSET:INTERNAL=$toolsetArgument") {
                Write-Host 'Refreshing the Visual Studio CMake cache to select the supported toolset.'
                $effectiveExtraArgs = @('--fresh') + $effectiveExtraArgs
            }
        }
        $effectiveExtraArgs = @('-T', $toolsetArgument) + $effectiveExtraArgs
    }
}

$needsNinja = $Action -in @('PackageSmoke', 'ProductShapeSmoke', 'SourcePackageSmoke') -or
    $Preset -in @('windows-ninja', 'windows-ninja-check', 'windows-clangcl',
        'windows-clangcl-check', 'win', 'windows-cuda')
if ($needsNinja -and -not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    throw "The '$Preset' preset/action requires Ninja on PATH."
}
if ($Preset -in @('windows-clangcl', 'windows-clangcl-check') -and
    -not (Get-Command clang-cl -ErrorAction SilentlyContinue)) {
    throw "The '$Preset' preset requires clang-cl on PATH. Install the Visual Studio LLVM component or initialize an environment that provides clang-cl."
}

switch ($Action) {
    'Configure' {
        if ($resolvedBinaryDir) {
            $selectionArg = @()
            if ($Component) {
                $selectionArg = @("-DHYLE_COMPONENTS=$Component")
            } elseif ($Factor) {
                $selectionArg = @("-DHYLE_DEV_FACTORS=$Factor")
            }
            Write-Host "HYLE isolated configure: $resolvedBinaryDir"
            & cmake --preset $Preset -B $resolvedBinaryDir @selectionArg @effectiveExtraArgs
        } else {
            & cmake --preset $Preset @effectiveExtraArgs
        }
    }
    'Build' {
        if ($resolvedBinaryDir) {
            & cmake --build $resolvedBinaryDir @effectiveExtraArgs
        } else {
            & cmake --build --preset $Preset @effectiveExtraArgs
        }
    }
    'Test' {
        if ($resolvedBinaryDir) {
            & ctest --test-dir $resolvedBinaryDir @effectiveExtraArgs
        } else {
            & ctest --preset $Preset @effectiveExtraArgs
        }
    }
    'Workflow'  {
        if ($Preset -eq 'windows-check' -and $vs2022Toolset) {
            # A workflow preset cannot forward a dynamically discovered -T
            # argument to its configure step, so spell out the same three
            # preset-backed steps here.
            & cmake --preset windows @effectiveExtraArgs
            if ($LASTEXITCODE -eq 0) { & cmake --build --preset windows }
            if ($LASTEXITCODE -eq 0) { & ctest --preset windows }
        } else {
            & cmake --workflow --preset $Preset @effectiveExtraArgs
        }
    }
    'ComponentContract' {
        if (-not $SmokeRoot) {
            throw 'Action ComponentContract requires -SmokeRoot <scratch-directory>.'
        }
        $smokeScript = Join-Path $sourceDir 'cmake\HyleCiComponentContract.cmake'
        & cmake "-DHYLE_SOURCE_DIR=$sourceDir" "-DHYLE_SMOKE_ROOT=$SmokeRoot" `
            -DHYLE_SMOKE_GENERATOR=Ninja -P $smokeScript
    }
    'SelectedComponentSmoke' {
        if (-not $SmokeRoot -or -not $Component) {
            throw 'Action SelectedComponentSmoke requires -SmokeRoot and -Component.'
        }
        $smokeScript = Join-Path $sourceDir 'cmake\HyleCiSelectedComponentSmoke.cmake'
        & cmake "-DHYLE_SOURCE_DIR=$sourceDir" "-DHYLE_SMOKE_ROOT=$SmokeRoot" `
            "-DHYLE_SMOKE_COMPONENT=$Component" -DHYLE_SMOKE_GENERATOR=Ninja `
            -P $smokeScript
    }
    'PackageSmoke' {
        if (-not $SmokeRoot) {
            throw 'Action PackageSmoke requires -SmokeRoot <scratch-directory>.'
        }
        $smokeScript = Join-Path $sourceDir 'cmake\HyleCiPackageSmoke.cmake'
        & cmake "-DHYLE_SOURCE_DIR=$sourceDir" "-DHYLE_SMOKE_ROOT=$SmokeRoot" `
            -DHYLE_SMOKE_GENERATOR=Ninja -P $smokeScript
    }
    'ProductShapeSmoke' {
        if (-not $SmokeRoot) {
            throw 'Action ProductShapeSmoke requires -SmokeRoot <scratch-directory>.'
        }
        $smokeScript = Join-Path $sourceDir 'cmake\HyleCiProductShapeSmoke.cmake'
        & cmake "-DHYLE_SOURCE_DIR=$sourceDir" "-DHYLE_SMOKE_ROOT=$SmokeRoot" `
            -DHYLE_SMOKE_GENERATOR=Ninja @ExtraArgs -P $smokeScript
    }
    'SourcePackageSmoke' {
        if (-not $SmokeRoot) {
            throw 'Action SourcePackageSmoke requires -SmokeRoot <scratch-directory>.'
        }
        $packageBuild = Join-Path $SmokeRoot 'package-driver'
        $smokeScript = Join-Path $sourceDir 'cmake\HyleCiSourcePackageSmoke.cmake'
        & cmake -S $sourceDir -B $packageBuild -G Ninja `
            -DCMAKE_BUILD_TYPE=Release `
            -DHYLE_COMPONENTS=all `
            -DHYLE_BUILD_TESTS=OFF `
            -DHYLE_BUILD_BENCH=OFF `
            -DHYLE_IO_TLS=OFF `
            -DHYLE_IO_HTTP2=OFF `
            -DHYLE_COMPUTE_CUDA=OFF `
            -DHYLE_COMPUTE_GGML=OFF `
            -DHYLE_COMPUTE_VULKAN=OFF
        if ($LASTEXITCODE -eq 0) {
            & cmake "-DHYLE_SOURCE_DIR=$sourceDir" `
                "-DHYLE_PACKAGE_BUILD=$packageBuild" `
                "-DHYLE_SMOKE_ROOT=$SmokeRoot" `
                -DHYLE_SMOKE_GENERATOR=Ninja -P $smokeScript
        }
    }
}
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
