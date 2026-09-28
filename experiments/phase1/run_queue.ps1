# Phase-1 job queue for the shared workstation: runs this project's resumable jobs one after another, each only while
# machine-wide free commit memory >= $StartGB; if it falls below $StopGB, stops ONLY the tree it started (PIDs collected
# from its own launcher downwards), waits, and resumes (every job checkpoints per item). Never touches other processes.
# Usage: powershell -File experiments\phase1\run_queue.ps1 -Queue experiments\phase1\queue.json [-StartGB 2.5] [-StopGB 0.7]
# queue.json: [{"name": "...", "args": ["script.py", "--workers", "2", ...], "env": {"K": "V"}, "items": "glob", "expected": 160}]
param([string]$Queue, [double]$StartGB = 2.5, [double]$StopGB = 0.7, [int]$MaxCrashes = 3)
$root = "E:\Project-git\Joule\hotloop"
$py = "D:\_caches\envs\hotloop\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
foreach ($v in "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS") { Set-Item "env:$v" "1" }
function FreeGB { (Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB }
function Done($job) { @(Get-ChildItem (Join-Path $root $job.items) -ErrorAction SilentlyContinue).Count }
function Tree($id) {
  $tree = @($id); $lvl = @($id)
  for ($i = 0; $i -lt 4; $i++) {
    $lvl = @(Get-CimInstance Win32_Process | Where-Object { $lvl -contains $_.ParentProcessId } | ForEach-Object { $_.ProcessId })
    $tree += $lvl
  }
  $tree | Sort-Object -Descending -Unique
}
$jobs = Get-Content (Join-Path $root $Queue) -Raw | ConvertFrom-Json
foreach ($job in $jobs) {
  $crashes = 0
  if ($job.env) { foreach ($p in $job.env.PSObject.Properties) { Set-Item "env:$($p.Name)" $p.Value } }
  while ((Done $job) -lt $job.expected) {
    if ((FreeGB) -lt $StartGB) { Start-Sleep -Seconds 60; continue }
    $log = Join-Path $root "experiments\phase1\out\queue_$($job.name).log"
    $p = Start-Process -FilePath $py -ArgumentList $job.args -WorkingDirectory $root -PassThru -WindowStyle Hidden `
         -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    "$(Get-Date -Format 'MM-dd HH:mm:ss') [$($job.name)] started pid $($p.Id); done $(Done $job)/$($job.expected); free commit $([math]::Round((FreeGB),2)) GB"
    $stopped = $false
    while (-not $p.HasExited) {
      Start-Sleep -Seconds 15
      $free = FreeGB
      if ($free -lt $StopGB) {
        $t = Tree $p.Id
        foreach ($id in $t) { try { Stop-Process -Id $id -Force -ErrorAction Stop } catch {} }
        "$(Get-Date -Format 'MM-dd HH:mm:ss') [$($job.name)] free commit $([math]::Round($free,2)) GB < $StopGB GB: stopped own tree $($t -join ',')"
        $stopped = $true
        Start-Sleep -Seconds 120
        break
      }
    }
    if (-not $stopped -and (Done $job) -lt $job.expected) {
      $crashes++
      "$(Get-Date -Format 'MM-dd HH:mm:ss') [$($job.name)] exited with $(Done $job)/$($job.expected) done (crash $crashes/$MaxCrashes); see $log.err"
      if ($crashes -ge $MaxCrashes) { "[$($job.name)] giving up"; break }
      Start-Sleep -Seconds 60
    }
  }
  "$(Get-Date -Format 'MM-dd HH:mm:ss') [$($job.name)] finished: $(Done $job)/$($job.expected)"
  if ($job.env) { foreach ($p in $job.env.PSObject.Properties) { Remove-Item "env:$($p.Name)" -ErrorAction SilentlyContinue } }
}
"$(Get-Date -Format 'MM-dd HH:mm:ss') queue complete"
