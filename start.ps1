Set-Location $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($env:LM_API_TOKEN)) {
    # A long-running PowerShell/Explorer process may not have inherited a
    # recently changed user variable. Read the saved user value explicitly.
    $userToken = [Environment]::GetEnvironmentVariable('LM_API_TOKEN', 'User')
    if (-not [string]::IsNullOrWhiteSpace($userToken)) {
        $env:LM_API_TOKEN = $userToken
        $userToken = $null
    }
}
if ([string]::IsNullOrWhiteSpace($env:LM_API_TOKEN)) {
    $tokenSecure = Read-Host 'LM Studio API token (input is hidden; leave blank if authentication is disabled)' -AsSecureString
    if ($tokenSecure.Length -gt 0) {
        $tokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($tokenSecure)
        try {
            $env:LM_API_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPointer)
        }
        finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPointer)
        }
    }
    $tokenSecure.Dispose()
}
& .\.venv\Scripts\python.exe -m uvicorn reviewer.app:app --host 127.0.0.1 --port 8765
