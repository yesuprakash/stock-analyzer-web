<?php
declare(strict_types=1);

header('Content-Type: application/json; charset=UTF-8');
header('Cache-Control: no-store');

require_once __DIR__ . DIRECTORY_SEPARATOR . 'stock_analyzer_bootstrap.php';
require_once stock_analyzer_backend_file('stock_research_lib.php');

function stock_research_respond(array $payload, int $status = 200): void
{
    http_response_code($status);
    echo json_encode($payload, JSON_INVALID_UTF8_SUBSTITUTE | JSON_PARTIAL_OUTPUT_ON_ERROR);
    exit;
}

stock_alert_log_fatal_shutdown('Stock research endpoint fatal PHP error');

try {
    if (($_GET['action'] ?? '') !== 'list') {
        stock_research_respond(['error' => 'Use action=list&symbol=... .'], 400);
    }

    $symbol = $_GET['symbol'] ?? '';
    if (!is_string($symbol) || !preg_match('/^[A-Z0-9][A-Z0-9.&-]{0,28}\.(NS|BO)$/', $symbol)) {
        stock_research_respond(['error' => 'A valid stock symbol is required.'], 400);
    }

    $connection = stock_alert_db();
    try {
        $activeSymbols = stock_master_symbols($connection);
        if (!in_array($symbol, $activeSymbols, true)) {
            stock_research_respond(['error' => 'Stock is not active in Stock List management.'], 404);
        }
        $recommendations = stock_research_list_for_symbol($connection, $symbol);
        $articles = stock_research_articles_for_symbol($connection, $symbol);
    } finally {
        pg_close($connection);
    }

    stock_research_respond([
        'symbol' => $symbol,
        'recommendations' => $recommendations,
        'count' => count($recommendations),
        'articles' => $articles,
        'article_count' => count($articles),
        'retention' => 'latest five trading dates with successful refreshes',
    ]);
} catch (Throwable $error) {
    stock_alert_log('Stock research request failed', $error, [
        'action' => (string)($_GET['action'] ?? ''),
        'symbol' => is_string($_GET['symbol'] ?? null) ? (string)$_GET['symbol'] : '',
    ]);
    error_log('Stock research request failed: ' . $error->getMessage());
    stock_research_respond(['error' => 'Research data could not be loaded. Check the Stock Analyzer PHP log.'], 500);
}
