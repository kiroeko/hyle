$ErrorActionPreference = 'Stop'
$taskScratch = (Resolve-Path -LiteralPath 'C:\public\project-hyle\hyle\.scratch\validation\d5-cpu-apple-official-sources').ProviderPath
$qualifiedSdk = Join-Path $taskScratch 'cpu-sdk-full\libtorch\lib'
$obsoleteSdk = (Resolve-Path -LiteralPath (Join-Path $taskScratch 'cpu-sdk-subset')).ProviderPath
$probe = (Resolve-Path -LiteralPath (Join-Path $taskScratch 'cpu-carrier-probe')).ProviderPath
$obsoleteFiles = @((Get-ChildItem -LiteralPath $obsoleteSdk -File -Recurse)) + @((Get-ChildItem -LiteralPath $probe -File -Filter '*.dll'))
$checkedFiles = @()
foreach ($file in $obsoleteFiles) {
  $absolute = $file.FullName
  if (-not $absolute.StartsWith($taskScratch + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside task scratch: $absolute" }
  if ($file.Extension -ne '.dll' -and $file.Extension -ne '.lib') { continue }
  $replacement = Join-Path $qualifiedSdk $file.Name
  if (-not (Test-Path -LiteralPath $replacement -PathType Leaf)) { throw "Missing retained SDK file: $replacement" }
  $originalHash = (Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash
  $replacementHash = (Get-FileHash -LiteralPath $replacement -Algorithm SHA256).Hash
  if ($originalHash -ne $replacementHash) { throw "Different retained SDK file: $absolute" }
  $checkedFiles += [pscustomobject]@{ path=$absolute; bytes=$file.Length; sha256=$originalHash }
}
foreach ($file in $checkedFiles) { Remove-Item -LiteralPath $file.path -Force }
$receipt = [pscustomobject]@{ retained_sdk=$qualifiedSdk; files=$checkedFiles; released_bytes=($checkedFiles | Measure-Object -Property bytes -Sum).Sum; reason='Delete only byte-identical unused qualification DLL/lib copies. Keep directories, source inputs, original archive and product SDK.' }
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath 'C:\public\project-hyle\hyle\.scratch\validation\d5-cpu-apple-official-binding\cleanup-obsolete-sdk.json' -Encoding UTF8
$receipt.released_bytes
