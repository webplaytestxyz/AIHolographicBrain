<#
Create the 'aibrain' conda environment using the provided environment.yml.

Usage (PowerShell, Anaconda Prompt):
  # from repository root (D:\AIHolographicBrain)
  .\create_env.ps1

This script will:
  - create a conda env named 'aibrain' from environment.yml
  - attempt a pip install for CPU PyTorch wheels if needed
#>

$envFile = Join-Path $PSScriptRoot 'environment.yml'
if (-not (Test-Path $envFile)) {
    Write-Error "environment.yml not found in $PSScriptRoot"
    exit 1
}

Write-Host "Creating conda environment 'aibrain' from environment.yml..."

# Use conda if available
$conda = Get-Command conda -ErrorAction SilentlyContinue
if ($null -eq $conda) {
    Write-Warning "'conda' was not found on PATH. Open Anaconda Prompt or ensure conda is on PATH."
    exit 1
}

# Create environment
conda env create -f "$envFile" --force
if ($LASTEXITCODE -ne 0) {
    Write-Warning "conda env create failed. You can try creating the environment manually or inspect the environment.yml."
    exit $LASTEXITCODE
}

Write-Host "Environment created. Activate it with:`n  conda activate aibrain"
Write-Host "If PyTorch is not available or you need a GPU build, follow instructions at https://pytorch.org"
