param(
    [int]$Port = 8000
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = $null

function Test-PythonExecutable {
    param([string]$Candidate)
    if (-not $Candidate) {
        return $false
    }
    if (-not (Test-Path -LiteralPath $Candidate)) {
        return $false
    }
    & $Candidate -c "import sys; print(sys.version)" *> $null
    return $LASTEXITCODE -eq 0
}

if (Test-PythonExecutable $VenvPython) {
    $Python = $VenvPython
} else {
    $PathPython = Get-Command python -ErrorAction SilentlyContinue
    if ($PathPython) {
        & $PathPython.Source -c "import sys; print(sys.version)" *> $null
        if ($LASTEXITCODE -eq 0) {
            $Python = $PathPython.Source
        }
    }
}

if (-not $Python) {
    Write-Host "No working Python executable was found."
    Write-Host "Install Python, recreate .venv if needed, then run:"
    Write-Host "  .\.venv\Scripts\pip install -r requirements.txt"
    exit 1
}

Push-Location $Root
try {
    & $Python -m intellitraffic.app --port $Port
} finally {
    Pop-Location
}
