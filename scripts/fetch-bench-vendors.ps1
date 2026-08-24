[CmdletBinding()]
param(
  [string]$VendorDir = $env:HYLE_BENCH_VENDOR_DIR
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoDir = Split-Path -Parent $PSScriptRoot
$versionsFile = Join-Path $repoDir 'cmake/vendor-versions.cmake'
if ([string]::IsNullOrWhiteSpace($VendorDir)) {
  $VendorDir = 'C:/public/project-hyle'
}

function Invoke-Git {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
  & git @GitArgs
  if ($LASTEXITCODE -ne 0) {
    throw "git $($GitArgs -join ' ') failed with exit code $LASTEXITCODE"
  }
}

# Windows PowerShell 5.1 lacks New-TemporaryFile; this .NET API is available there too.
$manifest = [System.IO.Path]::GetTempFileName()
try {
  & cmake "-DHYLE_BENCH_VENDOR_OUTPUT_FILE=$manifest" -P $versionsFile
  if ($LASTEXITCODE -ne 0) {
    throw "cmake failed to read $versionsFile"
  }

  New-Item -ItemType Directory -Force -Path $VendorDir | Out-Null
  foreach ($entry in Get-Content $manifest) {
    if ([string]::IsNullOrWhiteSpace($entry)) { continue }
    $name, $repository, $revision = $entry -split '\|', 3
    $checkout = Join-Path $VendorDir $name
    $gitDir = Join-Path $checkout '.git'

    if (Test-Path $checkout) {
      if (-not (Test-Path $gitDir)) {
        throw "$checkout exists but is not a Git checkout; refusing to overwrite it."
      }
      $status = & git -C $checkout status --porcelain
      if ($LASTEXITCODE -ne 0) {
        throw "git status failed for $checkout"
      }
      if ($status) {
        throw "$checkout has local changes; refusing to overwrite it."
      }
    } else {
      Invoke-Git clone $repository $checkout
    }

    Invoke-Git -C $checkout fetch --depth=1 origin $revision
    Invoke-Git -C $checkout checkout --detach $revision
    $actual = (& git -C $checkout rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
      throw "git rev-parse failed for $checkout"
    }
    if ($actual -ne $revision) {
      throw "$name resolved to $actual, expected $revision."
    }
    & cmake "-DHYLE_VERIFY_BENCH_DEPENDENCY_NAME=$name" `
      "-DHYLE_VERIFY_BENCH_DEPENDENCY_DIR=$checkout" -P $versionsFile
    if ($LASTEXITCODE -ne 0) {
      throw "locked provenance verification failed for $checkout"
    }
    Write-Host "[ok] $name $actual -> $checkout"
  }
} finally {
  Remove-Item -Force $manifest -ErrorAction SilentlyContinue
}
