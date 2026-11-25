# SPDX-License-Identifier: Apache-2.0
# Install the parts of NVIDIA's CUDA toolkit that a Windows build needs (compiler, runtime, CCCL and
# cuBLAS) from NVIDIA's redistributable archives, each checked against NVIDIA's manifest, into one
# directory. In GitHub Actions the directory becomes CUDA_PATH for the steps after this one.
param(
  [Parameter(Mandatory = $true)][string]$Version,
  [Parameter(Mandatory = $true)][string]$Root,
  [string]$ArchiveCache = "${Root}-archives"
)
$ErrorActionPreference = 'Stop'
$base = 'https://developer.download.nvidia.com/compute/cuda/redist/'
$manifest = Invoke-RestMethod "${base}redistrib_$Version.json"
$work = Join-Path ([IO.Path]::GetTempPath()) ('cuda-redist-' + [Guid]::NewGuid().ToString('N').Substring(0, 12))
$cache = Join-Path $ArchiveCache $Version
New-Item -ItemType Directory -Force $Root, $work, $cache | Out-Null
try {
  foreach ($name in 'cuda_nvcc', 'cuda_cudart', 'cuda_cccl', 'libcublas', 'cuda_nvtx', 'cuda_profiler_api') {
    $part = $manifest.$name.'windows-x86_64'
    $filename = Split-Path $part.relative_path -Leaf
    $zip = Join-Path $cache $filename
    if (!(Test-Path -LiteralPath $zip -PathType Leaf) -or (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -ne $part.sha256) {
      # Resume short responses privately. Only publish an archive after its digest matches.
      $download = Join-Path $work $filename
      for ($i = 0; $i -lt 20 -and -not ((Test-Path $download) -and (Get-Item $download).Length -ge [int64]$part.size); $i++) {
        curl.exe -fsSL -C - -o $download ($base + $part.relative_path)
      }
      if ((Get-FileHash -LiteralPath $download -Algorithm SHA256).Hash -ne $part.sha256) { throw "$name does not match the manifest" }
      Move-Item -LiteralPath $download -Destination $zip -Force
    }
    tar -xf $zip -C $work
    if ($LASTEXITCODE) { throw "$name does not extract" }
    $extracted = Join-Path $work ([IO.Path]::GetFileNameWithoutExtension($zip))
    Copy-Item -Recurse -Force (Join-Path $extracted '*') $Root
  }
} finally {
  Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}
if ($env:GITHUB_ENV) { "CUDA_PATH=$Root" >> $env:GITHUB_ENV }
