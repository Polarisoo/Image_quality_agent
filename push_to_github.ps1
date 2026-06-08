param(
  [Parameter(Mandatory=$true)]
  [string]$RepoUrl,
  [string]$Branch = "main",
  [string]$CommitMessage = "Add task3 image-quality agent release"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  throw "git was not found. Please install Git for Windows first."
}

if (-not (Test-Path -LiteralPath ".git")) {
  git init
}

git checkout -B $Branch
git add .

$hasChanges = git status --porcelain
if ($hasChanges) {
  git commit -m $CommitMessage
} else {
  Write-Host "No file changes to commit."
}

$remoteNames = git remote
if ($remoteNames -contains "origin") {
  git remote set-url origin $RepoUrl
} else {
  git remote add origin $RepoUrl
}

git push -u origin $Branch
Write-Host "Pushed to $RepoUrl on branch $Branch"
