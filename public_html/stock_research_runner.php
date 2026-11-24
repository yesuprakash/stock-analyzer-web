<?php
declare(strict_types=1);

if (PHP_SAPI !== 'cli') {
    http_response_code(404);
    exit;
}

require_once __DIR__ . DIRECTORY_SEPARATOR . 'stock_analyzer_bootstrap.php';
require_once stock_analyzer_backend_file('stock_research_lib.php');
stock_alert_log_fatal_shutdown('Stock research runner fatal PHP error');

try {
    $result = stock_research_run();
    echo json_encode($result, JSON_PRETTY_PRINT | JSON_INVALID_UTF8_SUBSTITUTE), PHP_EOL;
    exit(0);
} catch (Throwable $error) {
    stock_alert_log('Stock research runner failed', $error);
    error_log('Stock research runner failed: ' . $error->getMessage());
    fwrite(STDERR, 'Stock research refresh failed. Check logs/stock_analyzer_php.log.' . PHP_EOL);
    exit(1);
}
