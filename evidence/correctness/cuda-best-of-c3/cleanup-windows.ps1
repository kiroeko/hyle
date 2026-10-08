$ErrorActionPreference = 'Stop'
$c3Workspace = 'C:\public\project-hyle\hyle'
$c3Targets = @(
    'C:\hv\cuda-c3-static-initial-unqualified',
    'C:\hv\cuda-c3-static-consumer',
    'C:\hv\cuda-c3-shared-consumer',
    'C:\hv\cuda-c3-core-checks'
)
$c3Running = @(Get-CimInstance Win32_Process | ForEach-Object { $_.ExecutablePath } | Where-Object { $_ })
$c3Rows = @()
foreach ($c3Named in $c3Targets) {
    if (-not (Test-Path -LiteralPath $c3Named)) { continue }
    $c3Resolved = (Resolve-Path -LiteralPath $c3Named).Path
    if ($c3Resolved -ne $c3Named -or -not $c3Resolved.StartsWith('C:\hv\cuda-c3-', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unexpected cleanup path: $c3Resolved"
    }
    if (@($c3Running | Where-Object { $_.StartsWith($c3Resolved + '\', [StringComparison]::OrdinalIgnoreCase) }).Count) {
        throw "Cleanup target has a live process: $c3Resolved"
    }
    $c3Bytes = (Get-ChildItem -LiteralPath $c3Resolved -File -Recurse | Measure-Object Length -Sum).Sum
    Remove-Item -LiteralPath $c3Resolved -Recurse -Force
    $c3Rows += [PSCustomObject]@{ path = $c3Resolved; bytes = $c3Bytes; removed = -not (Test-Path -LiteralPath $c3Resolved) }
}
$c3Rows | ConvertTo-Json | Set-Content -LiteralPath "$c3Workspace\evidence\correctness\cuda-best-of-c3\cleanup-windows.json" -Encoding utf8
$c3Rows | Format-Table
