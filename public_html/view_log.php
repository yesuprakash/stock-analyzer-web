<?php
declare(strict_types=1);

$scriptDirectory = str_replace('\\', '/', dirname($_SERVER['SCRIPT_NAME'] ?? '/'));
$sessionPath = rtrim($scriptDirectory, '/') . '/';
header('Content-Type: text/plain; charset=UTF-8');
header('Cache-Control: no-store');
session_set_cookie_params([
    'httponly' => true,
    'secure' => !empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off',
    'samesite' => 'Strict',
    'path' => $sessionPath,
]);
session_start();

if (($_SESSION['stock_alert_authenticated'] ?? false) !== true) {
    http_response_code(401);
    exit('Sign in through Stock List or Email Alerts first.');
}

require_once __DIR__ . DIRECTORY_SEPARATOR . 'stock_analyzer_bootstrap.php';

try {
    $projectRoot = stock_analyzer_project_root();
} catch (Throwable $error) {
    http_response_code(500);
    exit($error->getMessage());
}

$logFile = $projectRoot . DIRECTORY_SEPARATOR . 'logs' . DIRECTORY_SEPARATOR . 'web_run.log';
if (!is_file($logFile)) {
    echo 'Log file not found.';
    exit;
}

$maxBytes = 50000;
$size = filesize($logFile);
if ($size === false) {
    http_response_code(500);
    exit('Unable to inspect analyzer log.');
}

$handle = fopen($logFile, 'rb');
if ($handle === false) {
    http_response_code(500);
    exit('Unable to read analyzer log.');
}

if ($size > $maxBytes && fseek($handle, -$maxBytes, SEEK_END) !== 0) {
    fclose($handle);
    http_response_code(500);
    exit('Unable to read analyzer log.');
}
$content = fread($handle, $maxBytes);
fclose($handle);

if ($content === false) {
    http_response_code(500);
    exit('Unable to read analyzer log.');
}

echo $content;
