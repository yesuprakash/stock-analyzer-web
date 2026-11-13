<?php
declare(strict_types=1);

if (PHP_SAPI !== 'cli') {
    http_response_code(404);
    exit;
}

define('PRICE_PERFORMANCE_LIBRARY_ONLY', true);
require_once __DIR__ . DIRECTORY_SEPARATOR . 'price_performance.php';
require_once __DIR__ . DIRECTORY_SEPARATOR . 'stock_analyzer_bootstrap.php';
require_once stock_analyzer_backend_file('stock_alert_lib.php');
stock_alert_log_fatal_shutdown('Stock alert runner fatal PHP error');

$forceRefresh = in_array('--force', $argv, true);

try {
    $result = stock_alert_run($forceRefresh);
    echo json_encode($result, JSON_PRETTY_PRINT | JSON_INVALID_UTF8_SUBSTITUTE), PHP_EOL;
    exit($result['status'] === 'completed' ? 0 : 1);
} catch (Throwable $error) {
    stock_alert_log('Stock alert runner failed', $error);
    error_log('Stock alert runner failed: ' . $error->getMessage());
    fwrite(STDERR, 'Stock alert run failed. Check logs/stock_analyzer_php.log.' . PHP_EOL);
    exit(1);
}
