<?php
declare(strict_types=1);

$scriptDirectory = str_replace('\\', '/', dirname($_SERVER['SCRIPT_NAME'] ?? '/'));
$sessionPath = rtrim($scriptDirectory, '/') . '/';
header('Content-Type: application/json; charset=UTF-8');
header('Cache-Control: no-store');
session_set_cookie_params([
    'httponly' => true,
    'secure' => !empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off',
    'samesite' => 'Strict',
    'path' => $sessionPath,
]);
session_start();

require_once __DIR__ . DIRECTORY_SEPARATOR . 'stock_analyzer_bootstrap.php';

function stock_caller_respond(array $payload, int $status = 200): void
{
    http_response_code($status);
    echo json_encode($payload, JSON_INVALID_UTF8_SUBSTITUTE | JSON_PARTIAL_OUTPUT_ON_ERROR);
    exit;
}

$action = $_GET['action'] ?? '';
if ($action === 'state') {
    if (($_SESSION['stock_alert_authenticated'] ?? false) !== true) {
        stock_caller_respond(['authenticated' => false], 401);
    }
    if (!isset($_SESSION['stock_alert_csrf'])) {
        $_SESSION['stock_alert_csrf'] = bin2hex(random_bytes(32));
    }
    stock_caller_respond([
        'authenticated' => true,
        'csrf' => $_SESSION['stock_alert_csrf'],
    ]);
}

if ($action !== 'start' || $_SERVER['REQUEST_METHOD'] !== 'POST') {
    stock_caller_respond(['error' => 'Unsupported request.'], 405);
}
if (($_SESSION['stock_alert_authenticated'] ?? false) !== true) {
    stock_caller_respond(['error' => 'Sign in through Stock List or Email Alerts first.'], 401);
}

$body = json_decode((string)file_get_contents('php://input'), true);
$csrf = is_array($body) ? ($body['csrf'] ?? '') : '';
if (!is_string($csrf) || !isset($_SESSION['stock_alert_csrf'])
    || !hash_equals($_SESSION['stock_alert_csrf'], $csrf)) {
    stock_caller_respond(['error' => 'Your session expired. Reload the page and sign in again.'], 403);
}

try {
    $projectRoot = stock_analyzer_project_root();
    $script = $projectRoot . DIRECTORY_SEPARATOR . 'stock_analyzer.py';
    $pythonParts = DIRECTORY_SEPARATOR === '\\'
        ? ['venv', 'Scripts', 'python.exe']
        : ['venv', 'bin', 'python'];
    $python = $projectRoot . DIRECTORY_SEPARATOR . implode(DIRECTORY_SEPARATOR, $pythonParts);
    if (!is_file($python)) {
        throw new RuntimeException('Python virtual environment executable was not found.');
    }
    if (!function_exists('exec')) {
        throw new RuntimeException('PHP exec is disabled.');
    }

    $logDir = $projectRoot . DIRECTORY_SEPARATOR . 'logs';
    if (!is_dir($logDir) && !mkdir($logDir, 0700, true) && !is_dir($logDir)) {
        throw new RuntimeException('Unable to create the private analyzer log directory.');
    }

    $logFile = $logDir . DIRECTORY_SEPARATOR . 'web_run.log';
    if (file_put_contents($logFile, date('c') . " | Analyzer starting.\n") === false) {
        throw new RuntimeException('Unable to initialize the analyzer log.');
    }

    if (DIRECTORY_SEPARATOR === '\\') {
        $command = 'cd /d ' . escapeshellarg($projectRoot)
            . ' && ' . escapeshellarg($python) . ' ' . escapeshellarg($script)
            . ' >> ' . escapeshellarg($logFile) . ' 2>&1';
        $process = popen('start /B cmd /C "' . $command . '"', 'r');
        if ($process === false) {
            throw new RuntimeException('Unable to start the analyzer process.');
        }
        pclose($process);
        stock_caller_respond(['message' => 'Stock analyzer started.']);
    }

    $task = 'cd ' . escapeshellarg($projectRoot)
        . ' && ' . escapeshellarg($python) . ' ' . escapeshellarg($script)
        . ' >> ' . escapeshellarg($logFile) . ' 2>&1; exitCode=$?; printf "\\nANALYZER_EXIT_CODE=%s\\n" "$exitCode" >> '
        . escapeshellarg($logFile);
    $command = 'nohup sh -c ' . escapeshellarg($task) . ' >/dev/null 2>&1 </dev/null & echo $!';
    $output = [];
    $exitCode = 0;
    exec($command, $output, $exitCode);
    if ($exitCode !== 0 || !isset($output[0]) || !ctype_digit(trim($output[0]))) {
        throw new RuntimeException('Unable to start the analyzer process.');
    }
    stock_caller_respond([
        'message' => 'Stock analyzer started.',
        'process_id' => trim($output[0]),
    ]);
} catch (Throwable $error) {
    stock_caller_respond(['error' => $error->getMessage()], 500);
}
