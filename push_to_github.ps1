# Push this folder to https://github.com/holleman-courses/hw1-Vaishnav-N
# Run in PowerShell from this directory (e.g. after cd into hw1-Vaishnav-N-main).
# Requires: Git installed and signed in (e.g. GitHub CLI or credential manager).

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git is not installed or not in PATH. Install from https://git-scm.com/download/win"
    exit 1
}

if (-not (Test-Path .git)) {
    Write-Host "Initializing git repo..."
    git init
    git remote add origin https://github.com/holleman-courses/hw1-Vaishnav-N.git
    git branch -M main
}

git remote -v
git add .github/ .gitignore README.md requirements.txt hw1_complete.py hw1_template.py hw1_test.py best_model.h5
git status
Write-Host "`nIf status looks good, uncomment and run the next two lines (commit + push):"
Write-Host "  git commit -m `"Complete HW1: models, training, best_model.h5`""
Write-Host "  git push -u origin main"
Write-Host "`nIf the repo already has commits (e.g. from Classroom), use: git pull origin main --rebase  then  git push -u origin main"
