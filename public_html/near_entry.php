<?php
declare(strict_types=1);

header('Content-Type: application/json; charset=UTF-8');
set_time_limit(0);

try {
    $siteParent = dirname(__DIR__);
    $projectRoots = [
        $siteParent,
        $siteParent . DIRECTORY_SEPARATOR . 'stock_analyzer',
    ];
    $projectRoot = null;
    $script = null;

    foreach ($projectRoots as $candidateRoot) {
        foreach ([
            $candidateRoot . DIRECTORY_SEPARATOR . 'near_entry_api.py',
            $candidateRoot . DIRECTORY_SEPARATOR . 'backend' . DIRECTORY_SEPARATOR . 'near_entry_api.py',
        ] as $candidateScript) {
            if (is_file($candidateScript)) {
                $projectRoot = $candidateRoot;
                $script = $candidateScript;
                break 2;
            }
        }
    }

    if ($projectRoot === null || $script === null) {
        throw new RuntimeException('Near Entry Python script was not found.');
    }

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

    $parameters = [
        'max_prediction_age_days' => $_GET['max_prediction_age_days'] ?? 3,
        'min_pct_tolerance' => $_GET['min_pct_tolerance'] ?? 2.0,
        'rsi_min' => $_GET['rsi_min'] ?? 45,
        'min_reward_risk' => $_GET['min_reward_risk'] ?? 1.5,
        'max_atr_pct' => $_GET['max_atr_pct'] ?? 5.0,
        'use_rsi' => $_GET['use_rsi'] ?? 1,
        'use_ma' => $_GET['use_ma'] ?? 1,
        'use_macd' => $_GET['use_macd'] ?? 1,
        'actionable_only' => $_GET['actionable_only'] ?? 0,
    ];

    $commandParts = [
        escapeshellarg($python),
        escapeshellarg($script),
    ];
    foreach ($parameters as $key => $value) {
        if (!is_scalar($value)) {
            throw new InvalidArgumentException('Invalid Near Entry parameter: ' . $key);
        }
        $commandParts[] = escapeshellarg($key . '=' . (string)$value);
    }

    $output = shell_exec(implode(' ', $commandParts) . ' 2>&1');
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
        'error' => 'Near Entry request failed',
        'details' => $error->getMessage(),
    ], JSON_INVALID_UTF8_SUBSTITUTE | JSON_PARTIAL_OUTPUT_ON_ERROR);
}
