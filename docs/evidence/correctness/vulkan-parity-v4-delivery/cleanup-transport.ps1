[CmdletBinding()]
param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath('C:\hv\v4w').TrimEnd('\')
$taskNames = @('air-current.bundle','air-delta.bundle','air-w2.bundle','air-resume.bundle','air-w3.bundle','air-final-delta.bundle','air-last-fixture.bundle','wsl-source.tar')
$taskFiles = @()
foreach ($taskName in $taskNames) {
    $taskPath = [IO.Path]::GetFullPath((Join-Path $taskRoot $taskName))
    if ([IO.Path]::GetDirectoryName($taskPath) -ne $taskRoot) { throw 'Archive escaped the fixed task root' }
    if (-not (Test-Path -LiteralPath $taskPath -PathType Leaf)) { continue }
    $taskItem = Get-Item -LiteralPath $taskPath
    if ($taskItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refuse archive link' }
    if ($taskItem.Length -gt 64MB) { throw 'Unexpected archive size' }
    $taskFiles += [pscustomobject]@{ path=$taskPath; bytes=$taskItem.Length; sha256=(Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash }
}
$taskTotalBytes = ($taskFiles | Measure-Object bytes -Sum).Sum
if ($Execute) {
    foreach ($taskFile in $taskFiles) {
        if ((Get-FileHash -LiteralPath $taskFile.path -Algorithm SHA256).Hash -ne $taskFile.sha256) { throw 'Archive changed since inspection' }
        Remove-Item -LiteralPath $taskFile.path -Force
    }
}
$taskRemaining = @($taskFiles | Where-Object { Test-Path -LiteralPath $_.path })
[pscustomobject]@{ executed=[bool]$Execute; files=$taskFiles; bytes=$taskTotalBytes; remaining=$taskRemaining.Count } | ConvertTo-Json -Depth 5
