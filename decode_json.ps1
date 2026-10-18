$json = Get-Clipboard -Raw | ConvertFrom-Json

# Decode the nested model response
if ($json.choices[0].message.content) {
    $content = $json.choices[0].message.content

    try {
        $json.choices[0].message.content = $content |
            ConvertFrom-Json
    }
    catch {
        # Keep as decoded text if it isn't valid JSON
    }
}

# Format the complete response
$formatted = $json | ConvertTo-Json -Depth 100

# Copy back to clipboard
Set-Clipboard -Value $formatted

Write-Host "Formatted response copied to clipboard."