<?php
declare(strict_types=1);

header('Content-Type: application/json; charset=UTF-8');

require_once __DIR__ . DIRECTORY_SEPARATOR . 'stock_analyzer_bootstrap.php';

try {
    $script = stock_analyzer_backend_file('market_correction_api.py');
    $projectRoot = dirname(dirname($script));
    $pythonParts = DIRECTORY_SEPARATOR === '\\'
        ? ['venv', 'Scripts', 'python.exe']
        : ['venv', 'bin', 'python'];
    $python = $projectRoot . DIRECTORY_SEPARATOR . implode(DIRECTORY_SEPARATOR, $pythonParts);

    if (!is_file($python)) {
        throw new RuntimeException('Python virtual environment executable was not found.');
    }
    if (!function_exists('shell_exec')) {
        throw new RuntimeException('PHP shell_exec is disabled.');
    }

    $arguments = [
        $python,
        $script,
        (string)($_GET['window'] ?? 30),
        (string)($_GET['min_corr'] ?? -8),
        (string)($_GET['max_corr'] ?? -20),
        (string)($_GET['rsi_min'] ?? 40),
        (string)($_GET['max_atr'] ?? 5.5),
        (string)($_GET['min_rr'] ?? 1.8),
        (string)(int)($_GET['skip_filters'] ?? 0),
    ];
    $command = implode(' ', array_map('escapeshellarg', $arguments)) . ' 2>&1';
    $output = shell_exec($command);

    if ($output === null || trim($output) === '') {
        throw new RuntimeException('Python returned no output.');
    }

    $output = trim($output);
    $decoded = json_decode($output);
    if (json_last_error() !== JSON_ERROR_NONE || !is_array($decoded)) {
        throw new RuntimeException('Python returned invalid JSON: ' . $output);
    }

    echo $output;
} catch (Throwable $error) {
    http_response_code(500);
    echo json_encode([
        'error' => 'Market Correction request failed',
        'details' => $error->getMessage(),
    ], JSON_INVALID_UTF8_SUBSTITUTE | JSON_PARTIAL_OUTPUT_ON_ERROR);
}
