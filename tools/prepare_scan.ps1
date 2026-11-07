# Make the two GLBs from a raw Scaniverse export (spec §6.2 step 4). Needs Node 22 (npx).
#
#   powershell -ExecutionPolicy Bypass -File tools\prepare_scan.ps1 maps\<name>
#
# reads   maps\<name>\raw\scan_raw.glb   (gitignored)
# writes  maps\<name>\scan.glb   web: meshopt + webp textures <= 2048 px   (D's R3F view; must be < 50 MB)
#         maps\<name>\sim.glb    analysis: simplified, UNcompressed (trimesh can't read meshopt)
param(
    [Parameter(Mandatory = $true)][string]$MapDir,
    [string]$Raw = "",
    [int]$Tex = 2048,
    [string]$Ratio = "0.25",
    [string]$Error = "0.001",
    [string]$GltfTransformVersion = "4"
)
$ErrorActionPreference = "Stop"
if (-not $Raw) { $Raw = Join-Path $MapDir "raw\scan_raw.glb" }
if (-not (Test-Path $Raw)) { throw "missing $Raw (export the Scaniverse GLB there)" }
$Scan = Join-Path $MapDir "scan.glb"
$Sim = Join-Path $MapDir "sim.glb"
$Pkg = "@gltf-transform/cli@$GltfTransformVersion"

function Invoke-GT([string[]]$GtArgs) {
    & npx --yes $Pkg @GtArgs
    if ($LASTEXITCODE -ne 0) { throw "gltf-transform $($GtArgs[0]) failed ($LASTEXITCODE)" }
}

# Web: an exact copy of the original when it fits the 50 MB limit (spec §1.2.6); else compress.
$rawSize = (Get-Item $Raw).Length
if ($rawSize -le 50MB) {
    Write-Host "== web: $Scan = exact copy of the original ($([math]::Round($rawSize / 1MB)) MB)"
    Copy-Item $Raw $Scan -Force
} else {
    Write-Host "== web: $Scan (compressed: original is $([math]::Round($rawSize / 1MB)) MB)"
    Invoke-GT @("optimize", $Raw, $Scan, "--compress", "meshopt", "--texture-compress", "webp", "--texture-size", "$Tex")
}

Write-Host "== sim: $Sim"
Invoke-GT @("simplify", $Raw, $Sim, "--ratio", $Ratio, "--error", $Error)
Invoke-GT @("resize", $Sim, $Sim, "--width", "$Tex", "--height", "$Tex")

Get-Item $Raw, $Scan, $Sim | Format-Table Name, @{n = "MB"; e = { "{0:N1}" -f ($_.Length / 1MB) } }
if ((Get-Item $Scan).Length -gt 50MB) {
    Write-Warning "scan.glb is > 50 MB. Retry with -Tex 1024, or -Ratio 0.5 + simplify before optimize (spec §6.4)."
}
Write-Host "next: python tools\scan_to_grid.py $Sim --entrance 0 0"
