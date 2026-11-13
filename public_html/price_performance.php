<?php
declare(strict_types=1);

header('Content-Type: application/json; charset=UTF-8');
header('Cache-Control: no-store');
set_time_limit(0);

const PERFORMANCE_PERIODS = [1, 2, 3, 4, 5, 10, 15, 30, 45, 60];
const BATCH_SIZE = 10;

require_once __DIR__ . DIRECTORY_SEPARATOR . 'stock_analyzer_bootstrap.php';
require_once stock_analyzer_backend_file('stock_master_lib.php');

function respond(array $payload, int $status = 200)
{
    http_response_code($status);
    echo json_encode($payload, JSON_INVALID_UTF8_SUBSTITUTE | JSON_PARTIAL_OUTPUT_ON_ERROR);
    exit;
}

function cachePaths(): array
{
    $documentRoot = $_SERVER['DOCUMENT_ROOT'] ?? dirname(__DIR__, 2);
    $privateRoot = dirname(rtrim($documentRoot, "\\/")) . DIRECTORY_SEPARATOR . 'Stock_Analyzer_cache';
    $cacheDir = $privateRoot . DIRECTORY_SEPARATOR . 'price_performance';

    if (!is_dir($cacheDir) && !mkdir($cacheDir, 0700, true) && !is_dir($cacheDir)) {
        throw new RuntimeException('Unable to create the private price cache directory.');
    }

    return [$cacheDir . DIRECTORY_SEPARATOR . 'daily_cache.json', $cacheDir . DIRECTORY_SEPARATOR . 'daily_cache.lock'];
}

function yahooUrl(string $symbol): string
{
    return 'https://query1.finance.yahoo.com/v8/finance/chart/' . rawurlencode($symbol)
        . '?range=6mo&interval=1d&events=div%2Csplits';
}

function parseYahooResponse(string $body, string $symbol): array
{
    $data = json_decode($body, true);
    $result = $data['chart']['result'][0] ?? null;
    if (!is_array($result) || !empty($data['chart']['error'])) {
        throw new RuntimeException('No daily price history returned.');
    }

    $timestamps = $result['timestamp'] ?? [];
    $quote = $result['indicators']['quote'][0] ?? [];
    $closes = $result['indicators']['adjclose'][0]['adjclose'] ?? $quote['close'] ?? [];
    if (!is_array($timestamps) || !is_array($closes) || count($timestamps) !== count($closes)) {
        throw new RuntimeException('Daily price history is incomplete.');
    }

    $timezoneName = $result['meta']['exchangeTimezoneName'] ?? 'Asia/Kolkata';
    try {
        $timezone = new DateTimeZone($timezoneName);
    } catch (Throwable $timezoneError) {
        $timezone = new DateTimeZone('Asia/Kolkata');
    }

    $history = [];
    foreach ($timestamps as $index => $timestamp) {
        $close = $closes[$index] ?? null;
        if (!is_numeric($timestamp) || !is_numeric($close) || !is_finite((float)$close) || (float)$close <= 0) {
            continue;
        }

        $history[] = [
            'date' => (new DateTimeImmutable('@' . (int)$timestamp))->setTimezone($timezone)->format('Y-m-d'),
            'close' => round((float)$close, 4),
        ];
    }

    if (count($history) < 2) {
        throw new RuntimeException('Not enough daily price history returned.');
    }

    return [
        'fetched_date' => (new DateTimeImmutable('now', new DateTimeZone('Asia/Kolkata')))->format('Y-m-d'),
        'fetched_at' => (new DateTimeImmutable('now', new DateTimeZone('Asia/Kolkata')))->format(DATE_ATOM),
        'history' => $history,
    ];
}

function fetchYahooBatch(array $symbols): array
{
    $multi = curl_multi_init();
    $handles = [];

    foreach ($symbols as $symbol) {
        $handle = curl_init(yahooUrl($symbol));
        curl_setopt_array($handle, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_FOLLOWLOCATION => true,
            CURLOPT_CONNECTTIMEOUT => 8,
            CURLOPT_TIMEOUT => 20,
            CURLOPT_SSL_VERIFYPEER => true,
            CURLOPT_SSL_VERIFYHOST => 2,
            CURLOPT_USERAGENT => 'Mozilla/5.0 StockPerformanceDashboard/1.0',
            CURLOPT_HTTPHEADER => ['Accept: application/json'],
        ]);
        curl_multi_add_handle($multi, $handle);
        $handles[$symbol] = $handle;
    }

    do {
        $status = curl_multi_exec($multi, $running);
        if ($running > 0) {
            $selected = curl_multi_select($multi, 1.0);
            if ($selected === -1) {
                usleep(100000);
            }
        }
    } while ($running > 0 && $status === CURLM_OK);

    $results = [];
    foreach ($handles as $symbol => $handle) {
        $body = curl_multi_getcontent($handle);
        $httpStatus = (int)curl_getinfo($handle, CURLINFO_HTTP_CODE);
        $curlError = curl_error($handle);

        try {
            if ($curlError !== '') {
                throw new RuntimeException($curlError);
            }
            if ($httpStatus !== 200 || !is_string($body)) {
                throw new RuntimeException('Price provider returned HTTP ' . $httpStatus . '.');
            }
            $results[$symbol] = ['data' => parseYahooResponse($body, $symbol)];
        } catch (Throwable $error) {
            $results[$symbol] = ['error' => $error->getMessage()];
        }

        curl_multi_remove_handle($multi, $handle);
        curl_close($handle);
    }

    curl_multi_close($multi);
    return $results;
}

function formatStockResult(string $symbol, array $cached, bool $stale): array
{
    $history = $cached['history'];
    $lastIndex = count($history) - 1;
    $latest = $history[$lastIndex];
    $previous = $history[$lastIndex - 1];
    $performance = [];

    foreach (PERFORMANCE_PERIODS as $period) {
        $baseIndex = $lastIndex - $period;
        $performance[(string)$period] = $baseIndex >= 0
            ? round((($latest['close'] / $history[$baseIndex]['close']) - 1) * 100, 2)
            : null;
    }

    return [
        'symbol' => $symbol,
        'latest_price' => $latest['close'],
        'latest_date' => $latest['date'],
        'daily_change_pct' => round((($latest['close'] / $previous['close']) - 1) * 100, 2),
        'performance' => $performance,
        'history' => array_slice($history, -61),
        'fetched_at' => $cached['fetched_at'] ?? null,
        'cache_date' => $cached['fetched_date'] ?? null,
        'stale' => $stale,
    ];
}

if (!defined('PRICE_PERFORMANCE_LIBRARY_ONLY')) {
stock_alert_log_fatal_shutdown('Price performance fatal PHP error');
try {
    $connection = stock_alert_db();
    stock_master_ensure_schema($connection);
    $performanceStocks = stock_master_symbols($connection);
    pg_close($connection);
    if ($performanceStocks === []) {
        respond(['error' => 'No active stocks are available. Enable stocks in Stock List management.'], 400);
    }

    $action = $_GET['action'] ?? '';
    if ($action === 'stocks') {
        respond([
            'stocks' => $performanceStocks,
            'count' => count($performanceStocks),
            'periods' => PERFORMANCE_PERIODS,
            'batch_size' => BATCH_SIZE,
            'cache_policy' => 'Each symbol is fetched at most once per calendar day; successful prices are cached privately.',
        ]);
    }

    if ($action !== 'analyze') {
        respond(['error' => 'Use action=stocks or action=analyze.'], 400);
    }

    $offset = filter_var($_GET['offset'] ?? null, FILTER_VALIDATE_INT);
    $limit = filter_var($_GET['limit'] ?? BATCH_SIZE, FILTER_VALIDATE_INT);
    if ($offset === false || $offset < 0 || $offset >= count($performanceStocks)
        || $limit === false || $limit < 1 || $limit > BATCH_SIZE) {
        respond(['error' => 'Invalid batch offset or limit.'], 400);
    }
    $forceRefresh = ($_GET['force'] ?? '0') === '1';

    $symbols = array_slice($performanceStocks, $offset, $limit);
    [$cacheFile, $lockFile] = cachePaths();
    $lock = fopen($lockFile, 'c');
    if ($lock === false || !flock($lock, LOCK_EX)) {
        throw new RuntimeException('Unable to lock the daily price cache.');
    }

    $cache = [];
    if (is_file($cacheFile)) {
        $saved = json_decode((string)file_get_contents($cacheFile), true);
        if (is_array($saved)) {
            $cache = $saved;
        }
    }

    $today = (new DateTimeImmutable('now', new DateTimeZone('Asia/Kolkata')))->format('Y-m-d');
    $missing = [];
    foreach ($symbols as $symbol) {
        $lastAttempt = $cache[$symbol]['attempted_date'] ?? $cache[$symbol]['fetched_date'] ?? null;
        if ($forceRefresh || $lastAttempt !== $today) {
            $missing[] = $symbol;
        }
    }

    $fetchResults = $missing === [] ? [] : fetchYahooBatch($missing);
    foreach ($fetchResults as $symbol => $result) {
        if (isset($result['data'])) {
            $cache[$symbol] = $result['data'] + ['attempted_date' => $today];
        } else {
            $cache[$symbol] ??= [];
            $cache[$symbol]['attempted_date'] = $today;
            $cache[$symbol]['last_error'] = $result['error'] ?? 'Price refresh failed.';
            stock_alert_log('Yahoo Finance price fetch failed', null, [
                'symbol' => $symbol,
                'error' => $cache[$symbol]['last_error'],
            ]);
        }
    }

    if ($fetchResults !== []) {
        $json = json_encode($cache, JSON_INVALID_UTF8_SUBSTITUTE | JSON_PARTIAL_OUTPUT_ON_ERROR);
        if ($json === false) {
            throw new RuntimeException('Unable to encode the daily price cache.');
        }
        $temporaryFile = $cacheFile . '.' . getmypid() . '.tmp';
        if (file_put_contents($temporaryFile, $json, LOCK_EX) === false
            || !rename($temporaryFile, $cacheFile)) {
            @unlink($temporaryFile);
            throw new RuntimeException('Unable to save the daily price cache.');
        }
    }

    $rows = [];
    foreach ($symbols as $symbol) {
        $entry = $cache[$symbol] ?? null;
        if (is_array($entry) && isset($entry['history'], $entry['fetched_date'])) {
            $row = formatStockResult($symbol, $entry, $entry['fetched_date'] !== $today);
            if (isset($entry['last_error'])) {
                $row['provider_error'] = $entry['last_error'];
            }
            $rows[] = $row;
        } else {
            $rows[] = [
                'symbol' => $symbol,
                'error' => $entry['last_error'] ?? $fetchResults[$symbol]['error'] ?? 'No cached price history is available.',
            ];
        }
    }

    flock($lock, LOCK_UN);
    fclose($lock);

    respond([
        'offset' => $offset,
        'count' => count($rows),
        'total' => count($performanceStocks),
        'force_refresh' => $forceRefresh,
        'rows' => $rows,
    ]);
} catch (Throwable $error) {
    stock_alert_log('Price performance request failed', $error, [
        'action' => (string)($_GET['action'] ?? ''),
    ]);
    error_log('Price performance request failed: ' . $error->getMessage());
    respond(['error' => 'Price analysis failed. Check the Stock Analyzer PHP log for details.'], 500);
}
}
