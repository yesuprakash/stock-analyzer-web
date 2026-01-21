<?php
header('Content-Type: application/json');

$window   = $_GET['window']   ?? 30;
$minCorr  = $_GET['min_corr'] ?? -8;
$maxCorr  = $_GET['max_corr'] ?? -20;
$rsiMin   = $_GET['rsi_min']  ?? 40;
$maxAtr   = $_GET['max_atr']  ?? 5.5;
$minRR    = $_GET['min_rr']   ?? 1.8;
$skipFilters = (int)($_GET['skip_filters'] ?? 0);

$python = '"C:\\xampp\\htdocs\\Stock_Analyzer\\venv\\Scripts\\python.exe"';
$script = '"C:\\xampp\\htdocs\\Stock_Analyzer\\backend\\market_correction_api.py"';

$cmd = sprintf(
    '%s %s %s %s %s %s %s %s %s 2>&1',
    $python,
    $script,
    escapeshellarg($window),
    escapeshellarg($minCorr),
    escapeshellarg($maxCorr),
    escapeshellarg($rsiMin),
    escapeshellarg($maxAtr),
    escapeshellarg($minRR),
    escapeshellarg($skipFilters)
);

/* DEBUG LOG */
file_put_contents(
    'C:\\xampp\\htdocs\\Stock_Analyzer\\logs\\php_debug.log',
    date('Y-m-d H:i:s') . "\nCMD:\n$cmd\n",
    FILE_APPEND
);

$output = shell_exec($cmd);

file_put_contents(
    'C:\\xampp\\htdocs\\Stock_Analyzer\\logs\\php_debug.log',
    "OUTPUT:\n$output\n\n",
    FILE_APPEND
);

$output = trim($output);

/* Always return JSON */
if ($output === '' || $output === null) {
    echo json_encode([]);
} elseif ($output[0] !== '[' && $output[0] !== '{') {
    // Python crashed → return safe JSON error
    echo json_encode([
        "error" => "Python execution failed",
        "details" => $output
    ]);
} else {
    echo $output;
}
