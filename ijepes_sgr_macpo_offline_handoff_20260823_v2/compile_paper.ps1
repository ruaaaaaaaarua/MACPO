$ErrorActionPreference = 'Stop'
$miktexBin = Join-Path $env:LOCALAPPDATA 'Programs/MiKTeX/miktex/bin/x64'
if (-not (Test-Path (Join-Path $miktexBin 'pdflatex.exe'))) {
    throw "MiKTeX was not found at $miktexBin"
}
$env:Path = "$miktexBin;$env:Path"
Push-Location $PSScriptRoot
try {
    pdflatex -interaction=nonstopmode -halt-on-error ijepes_sgr_macpo_revised_en.tex
    pdflatex -interaction=nonstopmode -halt-on-error ijepes_sgr_macpo_revised_en.tex
}
finally {
    Pop-Location
}
