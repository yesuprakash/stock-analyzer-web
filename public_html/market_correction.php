<?php
declare(strict_types=1);

header('Content-Type: application/json; charset=UTF-8');

require_once __DIR__ . DIRECTORY_SEPARATOR . 'stock_analyzer_bootstrap.php';

function market_correction_number(string $key, $default): float
{
    $value = $_GET[$key] ?? $default;
    if (!is_scalar($value) || !is_numeric((string)$value)) {
        throw new InvalidArgumentException('Invalid numeric filter: ' . $key);
    }

    $number = (float)$value;
    if (!is_finite($number)) {
        throw new InvalidArgumentException('Invalid numeric filter: ' . $key);
    }
    return $number;
}

try {
    $windowValue = market_correction_number('window', 30);
    $maxDataAgeValue = market_correction_number('max_data_age_days', 5);
    $window = (int)$windowValue;
    $maxDataAge = (int)$maxDataAgeValue;
    $minCorr = market_correction_number('min_corr', -20);
    $maxCorr = market_correction_number('max_corr', -8);
    $rsiMin = market_correction_number('rsi_min', 40);
    $maxAtr = market_correction_number('max_atr', 5.5);
    $minRr = market_correction_number('min_rr', 1.8);
    $skipFilters = $_GET['skip_filters'] ?? '0';

    if (floor($windowValue) !== $windowValue || floor($maxDataAgeValue) !== $maxDataAgeValue
        || !in_array($window, [5, 10, 15, 30, 45, 60], true)
        || $minCorr > $maxCorr
        || $rsiMin < 0 || $rsiMin > 100
        || $maxAtr < 0 || $minRr < 0
        || $maxDataAge < 0 || $maxDataAge > 30
        || !is_scalar($skipFilters)
        || !in_array((string)$skipFilters, ['0', '1'], true)) {
        throw new InvalidArgumentException('One or more Market Correction filters are out of range.');
    }

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
        (string)$window,
        (string)$minCorr,
        (string)$maxCorr,
        (string)$rsiMin,
        (string)$maxAtr,
        (string)$minRr,
        (string)(int)$skipFilters,
        (string)$maxDataAge,
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
} catch (InvalidArgumentException $error) {
    http_response_code(400);
    echo json_encode(['error' => $error->getMessage()], JSON_INVALID_UTF8_SUBSTITUTE);
} catch (Throwable $error) {
    http_response_code(500);
    echo json_encode([
        'error' => 'Market Correction request failed',
        'details' => $error->getMessage(),
    ], JSON_INVALID_UTF8_SUBSTITUTE | JSON_PARTIAL_OUTPUT_ON_ERROR);
}
