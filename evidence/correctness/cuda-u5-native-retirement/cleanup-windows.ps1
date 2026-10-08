$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath 'C:\hv').Path
$names = @(
  'cuda-u4-static-relocated', 'cuda-u4-static-consumer',
  'cuda-u4-shared-relocated', 'cuda-u4-shared-consumer',
  'cuda-u4-provider-preflight', 'cuda-u4-provider-p1-attempt1',
  'cuda-u4-provider-p1', 'cuda-u4-peer-identity'
)
$results = @()
foreach ($name in $names) {
  $path = Join-Path $taskRoot $name
  if (-not (Test-Path -LiteralPath $path)) { continue }
  $resolved = (Resolve-Path -LiteralPath $path).Path
  if ((Split-Path -Parent $resolved) -ne $taskRoot -or
      (Split-Path -Leaf $resolved) -ne $name) {
    throw "Target outside declared owned validation root: $resolved"
  }
  $bytes = (Get-ChildItem -LiteralPath $resolved -File -Recurse -Force |
    Measure-Object -Property Length -Sum).Sum
  Remove-Item -LiteralPath $resolved -Recurse -Force
  $results += [pscustomobject]@{
    path = $resolved
    bytes = $bytes
    removed = (-not (Test-Path -LiteralPath $resolved))
  }
}
$results | ConvertTo-Json -Depth 4 |
  Set-Content -LiteralPath 'C:\public\project-hyle\hyle\.scratch\validation\cuda-u5-native\cleanup-windows.json' -Encoding utf8
$results | Select-Object path,@{N='GiB';E={[math]::Round($_.bytes/1GB,3)}},removed
