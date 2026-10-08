$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath('C:\public\project-hyle\hyle\.scratch\validation\apple-parity')
$peerRoot = [IO.Path]::GetFullPath('C:\public\project-hyle\hyle\.scratch\perf-peers')
$removedBytes = 0L
foreach ($file in Get-ChildItem -LiteralPath $taskRoot -File -Filter '*.bundle') {
    $resolved = [IO.Path]::GetFullPath($file.FullName)
    if (-not $resolved.StartsWith($taskRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Transfer bundle escaped the owned task root: $resolved"
    }
    $removedBytes += $file.Length
    Remove-Item -LiteralPath $resolved -Force
}
$peer = [IO.Path]::GetFullPath((Join-Path $peerRoot 'mlx-1f8e74e3'))
if (-not $peer.StartsWith($peerRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Peer cleanup target escaped the named validation root'
}
if (Test-Path -LiteralPath $peer) {
    $item = Get-Item -LiteralPath $peer
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Peer root is a reparse point' }
    foreach ($file in Get-ChildItem -LiteralPath $peer -File -Recurse) { $removedBytes += $file.Length }
    Remove-Item -LiteralPath $peer -Recurse -Force
}
$bundle = Join-Path $peerRoot 'mlx-1f8e74e3.bundle'
if (Test-Path -LiteralPath $bundle) {
    $removedBytes += (Get-Item -LiteralPath $bundle).Length
    Remove-Item -LiteralPath $bundle -Force
}
$evidenceRoot = [IO.Path]::GetFullPath('C:\public\project-hyle\hyle\evidence\performance\apple-parity-overlap-p1')
if (-not (Test-Path -LiteralPath (Join-Path $evidenceRoot 'qualified\manifest.json')) -or
    -not (Test-Path -LiteralPath (Join-Path $evidenceRoot 'first-rejected\apple-overlap-aa.json'))) {
    throw 'Complete extracted performance evidence must exist before removing transfer packages'
}
foreach ($name in @('overlap-p1-format-7365f0fb9.tar.gz', 'overlap-first-326a40ee.tar.gz')) {
    $package = [IO.Path]::GetFullPath((Join-Path $evidenceRoot $name))
    if (-not $package.StartsWith($evidenceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Performance transfer package escaped its exact task root'
    }
    if (Test-Path -LiteralPath $package) {
        $removedBytes += (Get-Item -LiteralPath $package).Length
        Remove-Item -LiteralPath $package -Force
    }
}
[pscustomobject]@{scope='owned Apple transfer bundles, temporary Windows MLX reference, extracted performance transfer packages'; removed_bytes=$removedBytes} |
    ConvertTo-Json -Compress
