<?php
header('Content-Type: text/plain');

$logFile = realpath(__DIR__ . '/logs/market_correction_api.log');

if (!file_exists($logFile)) {
    echo "Log file not found";
    exit;
}

// Number of lines to show (last N lines)
$lines = 200;

// Efficient tail implementation
$file = new SplFileObject($logFile, 'r');
$file->seek(PHP_INT_MAX);
$lastLine = $file->key();

$start = max(0, $lastLine - $lines);

$output = [];
for ($i = $start; $i <= $lastLine; $i++) {
    $file->seek($i);
    $output[] = $file->current();
}

echo implode("", $output);
