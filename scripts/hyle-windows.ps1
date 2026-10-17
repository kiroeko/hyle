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

# Workflow build/test presets retain their own binary directory. Redirecting
# only configure would build and test a different (possibly stale) tree.
if ($Action -eq 'Workflow') {
    foreach ($argument in $ExtraArgs) {
        if ($argument.StartsWith('-B') -or $argument -eq '--build-directory' -or
            $argument.StartsWith('--build-directory=')) {
            throw 'Workflow cannot redirect its binary directory with -B/--build-directory. Use Configure, Build, and Test with the same -BinaryDir, or define a workflow whose presets share that directory.'
        }
    }
}

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
    if ($Action -notin @('Configure', 'SelectedComponentSmoke', 'PackageSmoke', 'ProductShapeSmoke', 'SourcePackageSmoke')) {
        throw '-Component is supported only by isolated Configure or a component/package smoke action.'
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

function Get-HyleSmokeComponents {
    if ($Component) { return $Component }
    return 'all'
}

function Get-HyleCmakeArgument {
    param([string]$Definition, [string]$ShortOption, [string]$LongOption)
    $value = $null
    for ($i = 0; $i -lt $ExtraArgs.Count; ++$i) {
        $argument = $ExtraArgs[$i]
        if ($Definition) {
            if ($argument -eq '-D' -and $i + 1 -lt $ExtraArgs.Count) {
                $argument = '-D' + $ExtraArgs[++$i]
            }
            if ($argument -match ('^-D' + [regex]::Escape($Definition) + '(?::[^=]+)?=(.*)$')) {
                $value = $Matches[1]
            }
        } elseif ($argument -eq $ShortOption -or $argument -eq $LongOption) {
            if ($i + 1 -lt $ExtraArgs.Count) { $value = $ExtraArgs[++$i] }
        } elseif ($ShortOption -and $argument.StartsWith($ShortOption) -and $argument.Length -gt $ShortOption.Length) {
            $value = $argument.Substring($ShortOption.Length)
        } elseif ($LongOption -and $argument.StartsWith($LongOption + '=')) {
            $value = $argument.Substring($LongOption.Length + 1)
        }
    }
    return $value
}

# Resolve checked-in preset inheritance only for toolchain/cache observation.
# Unknown/user presets still go to CMake unchanged; explicit cache/toolchain
# facts below take precedence over preset defaults.
$presets = Get-Content -LiteralPath (Join-Path $sourceDir 'CMakePresets.json') -Raw | ConvertFrom-Json
function Get-HyleConfigurePreset {
    param([string]$Name)
    $entry = $presets.configurePresets | Where-Object name -eq $Name | Select-Object -First 1
    if (-not $entry) { return $null }
    $result = @{ environment=@{}; cacheVariables=@{} }
    $parents = @($entry.inherits)
    for ($i = $parents.Count - 1; $i -ge 0; --$i) {
        if (-not $parents[$i]) { continue }
        $parent = Get-HyleConfigurePreset $parents[$i]
        if ($parent) {
            foreach ($key in $parent.Keys) {
                if ($key -in @('environment', 'cacheVariables')) {
                    if (-not $result.ContainsKey($key)) { $result[$key] = @{} }
                    foreach ($name in $parent[$key].Keys) { $result[$key][$name] = $parent[$key][$name] }
                } else { $result[$key] = $parent[$key] }
            }
        }
    }
    foreach ($key in @('generator', 'binaryDir', 'toolset', 'architecture')) {
        if ($entry.$key) { $result[$key] = $entry.$key }
    }
    foreach ($key in @('environment', 'cacheVariables')) {
        if (-not $entry.$key) { continue }
        if (-not $result.ContainsKey($key)) { $result[$key] = @{} }
        foreach ($property in $entry.$key.PSObject.Properties) {
            $result[$key][$property.Name] = $property.Value
        }
    }
    return $result
}
$workflow = $null
$configureName = $Preset
if ($Action -eq 'Workflow') {
    $workflow = $presets.workflowPresets | Where-Object name -eq $Preset | Select-Object -First 1
    if ($workflow) { $configureName = ($workflow.steps | Where-Object type -eq configure | Select-Object -First 1).name }
} elseif ($Action -in @('Build', 'Test') -and -not $resolvedBinaryDir) {
    $entries = if ($Action -eq 'Build') { $presets.buildPresets } else { $presets.testPresets }
    $entry = $entries | Where-Object name -eq $Preset | Select-Object -First 1
    if ($entry) { $configureName = $entry.configurePreset }
}
$presetSettings = Get-HyleConfigurePreset $configureName
$observedBinary = $resolvedBinaryDir
if ($Action -eq 'Configure') {
    $argumentBinary = Get-HyleCmakeArgument -ShortOption '-B' -LongOption '--build-directory'
    if ($argumentBinary) { $observedBinary = [System.IO.Path]::GetFullPath($argumentBinary) }
}
if (-not $observedBinary -and $presetSettings.binaryDir) {
    $observedBinary = $presetSettings.binaryDir.Replace('${sourceDir}', $sourceDir).Replace('${presetName}', $configureName)
}
$cache = @{}
if ($observedBinary -and -not ($Action -in @('Configure', 'Workflow') -and '--fresh' -in $ExtraArgs)) {
    $cachePath = Join-Path $observedBinary 'CMakeCache.txt'
    if (Test-Path -LiteralPath $cachePath) {
        foreach ($line in Get-Content -LiteralPath $cachePath) {
            if ($line -match '^([^#/:][^:=]*):[^=]*=(.*)$') { $cache[$Matches[1]] = $Matches[2] }
        }
    }
}
$effectiveExtraArgs = @($ExtraArgs)
$generator = if ($cache.CMAKE_GENERATOR) { $cache.CMAKE_GENERATOR } elseif ($presetSettings.generator) { $presetSettings.generator } else { $env:CMAKE_GENERATOR }
$toolsetArgument = if ($cache.ContainsKey('CMAKE_GENERATOR_TOOLSET')) { $cache.CMAKE_GENERATOR_TOOLSET } else { $presetSettings.toolset }
if ($toolsetArgument -and $toolsetArgument -isnot [string]) { $toolsetArgument = $toolsetArgument.value }
$installation = $cache.CMAKE_GENERATOR_INSTANCE
$compiler = if ($cache.CMAKE_CXX_COMPILER) { $cache.CMAKE_CXX_COMPILER } else { $env:CXX }
$cCompiler = if ($cache.CMAKE_C_COMPILER) { $cache.CMAKE_C_COMPILER } else { $env:CC }
$compilerFromPreset = $false
$cCompilerFromPreset = $false
$toolchain = if ($cache.CMAKE_TOOLCHAIN_FILE) { $cache.CMAKE_TOOLCHAIN_FILE } else { $env:CMAKE_TOOLCHAIN_FILE }
$architecture = if ($cache.CMAKE_GENERATOR_PLATFORM) { $cache.CMAKE_GENERATOR_PLATFORM } else { $presetSettings.architecture }
if ($architecture -and $architecture -isnot [string]) { $architecture = $architecture.value }
if ($Action -in @('Configure', 'Workflow')) {
    # Preset cache definitions override an existing cache; environment compiler
    # defaults apply only before that cache has selected its compiler. Keep
    # their origin so a clang-cl preset still initializes the required VS SDK.
    foreach ($selection in @(
        @{ Cache='CMAKE_CXX_COMPILER'; Environment='CXX'; Variable='compiler'; FromPreset='compilerFromPreset' },
        @{ Cache='CMAKE_C_COMPILER'; Environment='CC'; Variable='cCompiler'; FromPreset='cCompilerFromPreset' })) {
        if ($presetSettings -and $presetSettings.cacheVariables.ContainsKey($selection.Cache)) {
            $presetCompiler = $presetSettings.cacheVariables[$selection.Cache]
            if ($presetCompiler -and $presetCompiler -isnot [string]) { $presetCompiler = $presetCompiler.value }
            if ($presetCompiler) {
                Set-Variable -Name $selection.Variable -Value $presetCompiler
                Set-Variable -Name $selection.FromPreset -Value $true
            }
        } elseif (-not $cache[$selection.Cache] -and $presetSettings -and $presetSettings.environment.ContainsKey($selection.Environment)) {
            Set-Variable -Name $selection.Variable -Value $presetSettings.environment[$selection.Environment]
            Set-Variable -Name $selection.FromPreset -Value $true
        }
    }
    $value = Get-HyleCmakeArgument -ShortOption '-G' -LongOption '--generator'
    if ($value) { $generator = $value }
    $value = Get-HyleCmakeArgument -ShortOption '-T' -LongOption '--toolset'
    if ($value) { $toolsetArgument = $value }
    $value = Get-HyleCmakeArgument -ShortOption '-A' -LongOption '--platform'
    if ($value) { $architecture = $value }
    $value = Get-HyleCmakeArgument -Definition CMAKE_GENERATOR_INSTANCE
    if ($value) { $installation = $value }
    $value = Get-HyleCmakeArgument -Definition CMAKE_CXX_COMPILER
    if ($value) { $compiler = $value; $compilerFromPreset = $false }
    $value = Get-HyleCmakeArgument -Definition CMAKE_C_COMPILER
    if ($value) { $cCompiler = $value; $cCompilerFromPreset = $false }
    $value = Get-HyleCmakeArgument -Definition CMAKE_TOOLCHAIN_FILE
    if ($value) { $toolchain = $value }
}
$generatorMajor = if ($generator -match '^Visual Studio (\d+) ') { [int]$Matches[1] } else { 0 }
$requiredToolset = $null
$targetArchitecture = 'x64'
if ($architecture) {
    $targetArchitecture = switch ($architecture.Split(',')[0]) { 'Win32' { 'x86' }; default { $architecture.Split(',')[0].ToLowerInvariant() } }
}
# MSVC compiler/linker paths pin both the original VS instance and SxS tools.
# A cached Ninja build therefore does not drift when another VS is installed.
$nativePath = @($compiler, $cCompiler, $cache.CMAKE_LINKER) | Where-Object { $_ -match '[/\\]VC[/\\]Tools[/\\]MSVC[/\\]' } | Select-Object -First 1
if ($nativePath -match '^(.*)[/\\]VC[/\\]Tools[/\\]MSVC[/\\]([^/\\]+)[/\\]bin[/\\]Host[^/\\]+[/\\]([^/\\]+)[/\\]') {
    if (-not $installation) { $installation = $Matches[1] }
    $requiredToolset = $Matches[2]
    if (-not $architecture) { $targetArchitecture = $Matches[3].ToLowerInvariant() }
}
if ($toolsetArgument -match '(?:^|,)version=([^,]+)') { $requiredToolset = $Matches[1] }
if (-not $requiredToolset -and $cache.CMAKE_VS_PLATFORM_TOOLSET_VERSION) { $requiredToolset = $cache.CMAKE_VS_PLATFORM_TOOLSET_VERSION }

# An external compiler/toolchain with no VS anchor owns its setup. Likewise,
# Build/Test with an absent cache should let CMake report that cache error.
$presetMsvcFrontend = ($compilerFromPreset -and [System.IO.Path]::GetFileName($compiler) -match '^(clang-cl|cl)(\.exe)?$') -or
    ($cCompilerFromPreset -and [System.IO.Path]::GetFileName($cCompiler) -match '^(clang-cl|cl)(\.exe)?$')
$externalToolchain = -not $generatorMajor -and -not $installation -and
    ($toolchain -or (($compiler -or $cCompiler) -and -not $presetMsvcFrontend))
$missingBuildCache = $Action -in @('Build', 'Test') -and $cache.Count -eq 0
$unknownPreset = -not $presetSettings -and -not $installation -and $cache.Count -eq 0
if (-not $externalToolchain -and -not $missingBuildCache -and -not $unknownPreset) {
    if (-not $installation -and -not $generatorMajor -and $env:VSINSTALLDIR -and
        (Test-Path -LiteralPath (Join-Path $env:VSINSTALLDIR 'Common7\Tools\VsDevCmd.bat'))) {
        $installation = $env:VSINSTALLDIR
        if (-not $requiredToolset) { $requiredToolset = $env:VCToolsVersion }
    }
    if (-not $installation) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
        if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Install Visual Studio/Build Tools with the C++ workload, or initialize the selected custom toolchain.' }
        $query = @('-latest', '-products', '*', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath')
        if ($generatorMajor) { $query += @('-version', "[$generatorMajor.0,$($generatorMajor + 1).0)") }
        $installation = & $vswhere @query
        if ($LASTEXITCODE -ne 0 -or -not $installation) { throw "No Visual Studio C++ installation matches '$generator'." }
    }
    # CMake instance syntax can append a version constraint after the path.
    $installation = [System.IO.Path]::GetFullPath(($installation -split ',version=', 2)[0]).TrimEnd('\')
    if ($requiredToolset) { $requiredToolset = $requiredToolset.TrimEnd('\') }
    $toolsetRoot = Join-Path $installation 'VC\Tools\MSVC'
    $toolsets = @(Get-ChildItem -LiteralPath $toolsetRoot -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { try { [pscustomobject]@{ Name=$_.Name; Version=[version]$_.Name } } catch {} } |
        Where-Object {
            if ($requiredToolset) { $_.Name -eq $requiredToolset -or $_.Name.StartsWith($requiredToolset.TrimEnd('\') + '.') }
            else {
                $v143 = $toolsetArgument -match '(^|,)v143(,|$)' -or ($generatorMajor -eq 17 -and -not $toolsetArgument)
                $v145 = $toolsetArgument -match '(^|,)v145(,|$)'
                $_.Version -ge [version]'14.43' -and (-not $v143 -or $_.Version -lt [version]'14.50') -and
                    (-not $v145 -or $_.Version -ge [version]'14.50')
            }
        } | Sort-Object Version -Descending)
    if (-not $toolsets.Count) { throw "Visual Studio at '$installation' has no matching MSVC toolset '$requiredToolset'." }
    $selectedToolset = $toolsets[0].Name
    # Only a fresh VS 2022 configure needs an automatic -T; never reset an
    # existing cache or replace its compiler/toolset choice implicitly.
    if ($generatorMajor -eq 17 -and $Action -in @('Configure', 'Workflow') -and
        $cache.Count -eq 0 -and -not $toolsetArgument) {
        $effectiveExtraArgs = @('-T', "version=$selectedToolset") + $effectiveExtraArgs
    }
    $vsDevCmd = Join-Path $installation 'Common7\Tools\VsDevCmd.bat'
    if (-not (Test-Path -LiteralPath $vsDevCmd)) { throw "Visual Studio environment script was not found: $vsDevCmd" }
    $activeInstallation = if ($env:VSINSTALLDIR) { [System.IO.Path]::GetFullPath($env:VSINSTALLDIR).TrimEnd('\') } else { '' }
    $hasSelectedEnvironment = $env:VSCMD_VER -and $env:VCINSTALLDIR -and $env:WindowsSdkDir -and
        $env:VSCMD_ARG_TGT_ARCH -eq $targetArchitecture -and $activeInstallation -eq $installation -and
        $env:VCToolsVersion -and $env:VCToolsVersion.TrimEnd('\') -eq $selectedToolset
    if (-not $hasSelectedEnvironment) {
        $environmentCommand = 'call "' + $vsDevCmd + '" -arch=' + $targetArchitecture + ' -host_arch=x64 -vcvars_ver=' + $selectedToolset + ' >nul && set'
        if ($env:VSCMD_VER -and $activeInstallation) {
            $activeDevCmd = Join-Path $activeInstallation 'Common7\Tools\VsDevCmd.bat'
            if (Test-Path -LiteralPath $activeDevCmd) {
                # The original VsDevCmd reverses its own setup before switching
                # instance/toolset. Otherwise VC may retain the previous VS root.
                $environmentCommand = 'call "' + $activeDevCmd + '" -clean_env >nul && ' + $environmentCommand
            }
        }
        $environmentLines = & $env:ComSpec /d /s /c $environmentCommand
        if ($LASTEXITCODE -ne 0) { throw "Failed to initialize Visual Studio at '$installation' with MSVC $selectedToolset." }
        foreach ($line in $environmentLines) {
            $separator = $line.IndexOf('=')
            if ($separator -le 0) { continue }
            $name = $line.Substring(0, $separator)
            $value = $line.Substring($separator + 1)
            if ($name -ine 'Path' -or $value.Replace('/', '\') -like "*$installation\VC\Tools\MSVC\*") {
                Set-Item -Path "Env:$name" -Value $value
            }
        }
    }
    Write-Host "HYLE Windows toolchain: $installation; MSVC $selectedToolset; $targetArchitecture"
}

$usesPresetToolchain = $Action -in @('Configure', 'Workflow')
$needsNinja = $Action -in @('ComponentContract', 'SelectedComponentSmoke', 'PackageSmoke', 'ProductShapeSmoke', 'SourcePackageSmoke') -or
    ($usesPresetToolchain -and $generator -match '^Ninja( Multi-Config)?$')
if ($needsNinja -and -not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    throw "The '$Preset' preset/action requires Ninja on PATH."
}
if ($usesPresetToolchain) {
    # Visual Studio selects its compiler through -T, rather than CC/CXX.
    $selectedCompilers = if ($generatorMajor) {
        if ($toolsetArgument -match '(^|,)ClangCL(,|$)') { @('clang-cl') } else { @() }
    } else { @($compiler, $cCompiler) }
    foreach ($selectedCompiler in $selectedCompilers | Select-Object -Unique) {
        if ($selectedCompiler -and [System.IO.Path]::GetFileName($selectedCompiler) -match '^clang-cl(\.exe)?$' -and
            -not (Get-Command $selectedCompiler -ErrorAction SilentlyContinue)) {
            throw "The selected compiler '$selectedCompiler' is unavailable. Install the Visual Studio LLVM component or initialize an environment that provides clang-cl."
        }
    }
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
        if ($workflow -and -not ($workflow.steps | Where-Object type -notin @('configure', 'build', 'test'))) {
            # Configure arguments (-T/-D/--fresh) belong to its configure step,
            # not to cmake --workflow. Keep build/test on their named presets.
            foreach ($step in $workflow.steps) {
                switch ($step.type) {
                    'configure' { & cmake --preset $step.name @effectiveExtraArgs }
                    'build' { & cmake --build --preset $step.name }
                    'test' { & ctest --preset $step.name }
                }
                if ($LASTEXITCODE -ne 0) { break }
            }
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
            -DHYLE_SMOKE_GENERATOR=Ninja @ExtraArgs -P $smokeScript
    }
    'SelectedComponentSmoke' {
        if (-not $SmokeRoot -or -not $Component) {
            throw 'Action SelectedComponentSmoke requires -SmokeRoot and -Component.'
        }
        $smokeScript = Join-Path $sourceDir 'cmake\HyleCiSelectedComponentSmoke.cmake'
        & cmake "-DHYLE_SOURCE_DIR=$sourceDir" "-DHYLE_SMOKE_ROOT=$SmokeRoot" `
            "-DHYLE_SMOKE_COMPONENT=$Component" -DHYLE_SMOKE_GENERATOR=Ninja `
            @ExtraArgs -P $smokeScript
    }
    'PackageSmoke' {
        if (-not $SmokeRoot) {
            throw 'Action PackageSmoke requires -SmokeRoot <scratch-directory>.'
        }
        $smokeScript = Join-Path $sourceDir 'cmake\HyleCiPackageSmoke.cmake'
        $smokeComponents = Get-HyleSmokeComponents
        & cmake "-DHYLE_SOURCE_DIR=$sourceDir" "-DHYLE_SMOKE_ROOT=$SmokeRoot" `
            "-DHYLE_SMOKE_COMPONENTS=$smokeComponents" `
            -DHYLE_SMOKE_GENERATOR=Ninja @ExtraArgs -P $smokeScript
    }
    'ProductShapeSmoke' {
        if (-not $SmokeRoot) {
            throw 'Action ProductShapeSmoke requires -SmokeRoot <scratch-directory>.'
        }
        $smokeScript = Join-Path $sourceDir 'cmake\HyleCiProductShapeSmoke.cmake'
        $smokeComponents = Get-HyleSmokeComponents
        & cmake "-DHYLE_SOURCE_DIR=$sourceDir" "-DHYLE_SMOKE_ROOT=$SmokeRoot" `
            "-DHYLE_SMOKE_COMPONENTS=$smokeComponents" `
            -DHYLE_SMOKE_GENERATOR=Ninja @ExtraArgs -P $smokeScript
    }
    'SourcePackageSmoke' {
        if (-not $SmokeRoot) {
            throw 'Action SourcePackageSmoke requires -SmokeRoot <scratch-directory>.'
        }
        $packageBuild = Join-Path $SmokeRoot 'package-driver'
        $smokeScript = Join-Path $sourceDir 'cmake\HyleCiSourcePackageSmoke.cmake'
        $smokeComponents = Get-HyleSmokeComponents
        & cmake -S $sourceDir -B $packageBuild -G Ninja `
            -DCMAKE_BUILD_TYPE=Release `
            "-DHYLE_COMPONENTS=$smokeComponents" `
            -DHYLE_BUILD_TESTS=OFF `
            -DHYLE_BUILD_BENCH=OFF `
            -DHYLE_IO_TLS=OFF `
            -DHYLE_IO_HTTP2=OFF
        if ($LASTEXITCODE -eq 0) {
            & cmake "-DHYLE_SOURCE_DIR=$sourceDir" `
                "-DHYLE_PACKAGE_BUILD=$packageBuild" `
                "-DHYLE_SMOKE_ROOT=$SmokeRoot" `
                "-DHYLE_SMOKE_COMPONENTS=$smokeComponents" `
                -DHYLE_SMOKE_GENERATOR=Ninja @ExtraArgs -P $smokeScript
        }
    }
}
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
