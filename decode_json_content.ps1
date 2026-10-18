$text = Get-Clipboard -Raw
$decoded = $text | ConvertFrom-Json

try {
    $formatted = $decoded | ConvertFrom-Json |
        ConvertTo-Json -Depth 100
} catch {
    $formatted = $decoded
}

Set-Clipboard $formatted