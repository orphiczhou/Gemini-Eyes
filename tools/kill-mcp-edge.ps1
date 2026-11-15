<#
Kill msedge processes running on the dedicated MCP profile (edge-mcp-profile).

The dump flow (mcp-browser-profile.ps1 -Dump) starts Edge on that profile to
read cookies over CDP. Once the dump is done, a leftover browser window on
the profile is a liability: it keeps a second session alive that can sign
out or race the MCP's own cookie rotation. The health watchdog calls this
script right after a successful dump. Best effort by design: exit 0 even
when nothing matched, so recovery never trips on a harmless miss.

ASCII only on purpose (no-BOM .ps1 with non-ASCII text breaks under some
codepages).
#>
[CmdletBinding()]
param(
    [string]$Pattern = "edge-mcp-profile"
)

$ErrorActionPreference = "Continue"
$found = 0

try {
    $procs = Get-CimInstance Win32_Process -Filter "Name = 'msedge.exe'" -ErrorAction SilentlyContinue
} catch {
    $procs = @()
}

foreach ($p in $procs) {
    if ($p.CommandLine -and ($p.CommandLine -like "*$Pattern*")) {
        $found++
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Output ("killed msedge pid {0} (matched {1})" -f $p.ProcessId, $Pattern)
    }
}

if ($found -eq 0) {
    Write-Output ("no msedge processes matched {0}" -f $Pattern)
}
exit 0
