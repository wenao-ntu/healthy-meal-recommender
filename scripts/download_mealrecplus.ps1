param(
    [string]$Destination
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $Destination) {
    $Destination = Join-Path $ProjectRoot 'data\raw\mealrecplus-source'
}

$Repository = 'https://github.com/WUT-IDEA/MealRecPlus.git'
$PinnedCommit = '8bd7d7d7effb505be4a7d2074b28abad802fe0f7'
$ExpectedArtifacts = @{
    'MealRec+\MealRec+H\meta_data\course.zip' = @{
        Bytes = 195052658
        Sha256 = '00f10920166e21bbf2a581c835027cc413c7bc96c79053e4a8cfe0774dbff7ce'
    }
    'MealRec+\MealRec+H\meta_data\user_course.zip' = @{
        Bytes = 1880922
        Sha256 = '661892a6c73ad03dec66b74d8d92ddfb87cef3a6150ab0bfd178ae178c8e62e7'
    }
}

function Invoke-ProjectGit {
    param([string[]]$GitArguments)

    & git -c http.sslBackend=openssl @GitArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed: git $($GitArguments -join ' ')"
    }
}

function Test-ExpectedArtifacts {
    foreach ($RelativePath in $ExpectedArtifacts.Keys) {
        $Path = Join-Path $Destination $RelativePath
        if (-not (Test-Path -LiteralPath $Path)) {
            return $false
        }
        $Expected = $ExpectedArtifacts[$RelativePath]
        $Item = Get-Item -LiteralPath $Path
        if ($Item.Length -ne $Expected.Bytes) {
            return $false
        }
        $ActualHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualHash -ne $Expected.Sha256) {
            return $false
        }
    }
    return $true
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git is required but was not found.'
}
& git lfs version | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw 'Git LFS is required but was not found.'
}

if (-not (Test-Path -LiteralPath $Destination)) {
    $env:GIT_LFS_SKIP_SMUDGE = '1'
    Invoke-ProjectGit -GitArguments @('clone', '--depth', '1', $Repository, $Destination)
} elseif (-not (Test-Path -LiteralPath (Join-Path $Destination '.git'))) {
    throw "Destination exists but is not a Git repository: $Destination"
}

$CurrentCommit = (& git -C $Destination rev-parse HEAD).Trim()
if ($CurrentCommit -ne $PinnedCommit) {
    Invoke-ProjectGit -GitArguments @(
        '-C', $Destination, 'fetch', '--depth', '1', 'origin', $PinnedCommit
    )
    Invoke-ProjectGit -GitArguments @('-C', $Destination, 'checkout', '--detach', $PinnedCommit)
}

if (-not (Test-ExpectedArtifacts)) {
    & git -c http.sslBackend=openssl -C $Destination lfs pull `
        --include='MealRec+/MealRec+H/meta_data/*.zip' --exclude=''
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'Git LFS reported an error; artifact hashes will determine success.'
    }
}

foreach ($RelativePath in $ExpectedArtifacts.Keys) {
    $Path = Join-Path $Destination $RelativePath
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required artifact is missing: $RelativePath"
    }
    $Expected = $ExpectedArtifacts[$RelativePath]
    $Item = Get-Item -LiteralPath $Path
    $ActualHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Item.Length -ne $Expected.Bytes -or $ActualHash -ne $Expected.Sha256) {
        throw "Integrity check failed: $RelativePath"
    }
    Write-Output "VERIFIED $RelativePath $($Item.Length) bytes $ActualHash"
}

Write-Output "SOURCE_COMMIT $PinnedCommit"
Write-Warning 'The upstream repository requests citation but does not declare a LICENSE file.'
Write-Output 'MealRec+H is ready for the read-only schema audit.'
