<?php
// SECURITY: optional simple token (recommended)
$token = $_GET['token'] ?? '';
if ($token !== 'LOCAL_STOCK_ANALYZER_RUN') {
    http_response_code(403);
    exit('Forbidden');
}

$logFile = __DIR__ . '/../logs/web_run.log';

if (!file_exists($logFile)) {
    echo "Log file not found.";
    exit;
}

// Read last N bytes (avoid huge logs)
$maxBytes = 50000; // ~50 KB
$size = filesize($logFile);

$fh = fopen($logFile, 'r');
if ($size > $maxBytes) {
    fseek($fh, -$maxBytes, SEEK_END);
}
$content = fread($fh, $maxBytes);
fclose($fh);

header("Content-Type: text/plain; charset=UTF-8");
echo $content;
