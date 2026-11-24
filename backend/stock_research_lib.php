<?php
declare(strict_types=1);

require_once __DIR__ . DIRECTORY_SEPARATOR . 'stock_alert_lib.php';
require_once __DIR__ . DIRECTORY_SEPARATOR . 'stock_master_lib.php';

const STOCK_RESEARCH_RATINGS = [
    'STRONG_BUY',
    'BUY',
    'ACCUMULATE',
    'OUTPERFORM',
    'HOLD',
    'NEUTRAL',
    'UNDERPERFORM',
    'REDUCE',
    'SELL',
    'STRONG_SELL',
];

function stock_research_parse_feed(string $body, array $allowedSymbols): array
{
    if (strlen($body) > 5000000) {
        throw new RuntimeException('Research feed response exceeds the 5 MB limit.');
    }

    $payload = json_decode($body, true, 512, JSON_THROW_ON_ERROR);
    if (!is_array($payload) || !isset($payload['recommendations']) || !is_array($payload['recommendations'])) {
        throw new RuntimeException('Research feed must contain a recommendations array.');
    }

    $allowed = array_fill_keys($allowedSymbols, true);
    $recommendations = [];
    foreach ($payload['recommendations'] as $index => $item) {
        if (!is_array($item)) {
            throw new RuntimeException('Research feed item ' . $index . ' is not an object.');
        }

        $symbol = $item['symbol'] ?? null;
        $source = $item['source'] ?? null;
        $rating = $item['recommendation'] ?? null;
        if (!is_string($symbol) || !isset($allowed[$symbol])) {
            throw new RuntimeException('Research feed item ' . $index . ' has an unknown stock symbol.');
        }
        if (!is_string($source) || trim($source) === '' || strlen($source) > 120) {
            throw new RuntimeException('Research feed item ' . $index . ' has an invalid source name.');
        }
        if (!is_string($rating)) {
            throw new RuntimeException('Research feed item ' . $index . ' has no recommendation.');
        }
        $rating = strtoupper(str_replace([' ', '-'], '_', trim($rating)));
        if (!in_array($rating, STOCK_RESEARCH_RATINGS, true)) {
            throw new RuntimeException('Research feed item ' . $index . ' has an unsupported recommendation.');
        }

        $analyst = $item['analyst'] ?? '';
        $horizon = $item['horizon'] ?? '';
        if (!is_string($analyst) || strlen($analyst) > 120 || !is_string($horizon) || strlen($horizon) > 120) {
            throw new RuntimeException('Research feed item ' . $index . ' has invalid analyst or horizon text.');
        }

        $targetPrice = $item['target_price'] ?? null;
        if ($targetPrice !== null
            && (!is_numeric($targetPrice) || !is_finite((float)$targetPrice) || (float)$targetPrice <= 0)) {
            throw new RuntimeException('Research feed item ' . $index . ' has an invalid target price.');
        }

        $currency = strtoupper((string)($item['currency'] ?? 'INR'));
        if (!preg_match('/^[A-Z]{3}$/', $currency)) {
            throw new RuntimeException('Research feed item ' . $index . ' has an invalid currency.');
        }

        $publishedAt = $item['published_at'] ?? null;
        if ($publishedAt !== null) {
            if (!is_string($publishedAt) || trim($publishedAt) === '') {
                throw new RuntimeException('Research feed item ' . $index . ' has an invalid publication date.');
            }
            try {
                $publishedAt = (new DateTimeImmutable($publishedAt))->format(DATE_ATOM);
            } catch (Throwable $error) {
                throw new RuntimeException('Research feed item ' . $index . ' has an invalid publication date.', 0, $error);
            }
        }

        $sourceUrl = $item['source_url'] ?? null;
        if ($sourceUrl !== null && $sourceUrl !== '') {
            if (!is_string($sourceUrl)) {
                throw new RuntimeException('Research feed item ' . $index . ' has an invalid source URL.');
            }
            $parts = parse_url($sourceUrl);
            if (!is_array($parts) || strtolower((string)($parts['scheme'] ?? '')) !== 'https'
                || empty($parts['host']) || strlen($sourceUrl) > 2048) {
                throw new RuntimeException('Research feed item ' . $index . ' must use a valid HTTPS source URL.');
            }
        } else {
            $sourceUrl = null;
        }

        $recommendations[] = [
            'symbol' => $symbol,
            'source' => trim($source),
            'analyst' => trim($analyst),
            'recommendation' => $rating,
            'target_price' => $targetPrice === null ? null : (string)$targetPrice,
            'currency' => $currency,
            'horizon' => trim($horizon),
            'published_at' => $publishedAt,
            'source_url' => $sourceUrl,
        ];
    }

    return $recommendations;
}

function stock_research_fetch_feed(array $symbols): array
{
    if (!function_exists('curl_init')) {
        throw new RuntimeException('PHP cURL extension is required for the research feed.');
    }

    $feedUrl = stock_alert_env('STOCK_RESEARCH_FEED_URL');
    if ($feedUrl === null || $feedUrl === '') {
        throw new RuntimeException('STOCK_RESEARCH_FEED_URL is not configured in the project .env file.');
    }
    $parts = parse_url($feedUrl);
    if (!is_array($parts) || strtolower((string)($parts['scheme'] ?? '')) !== 'https' || empty($parts['host'])) {
        throw new RuntimeException('STOCK_RESEARCH_FEED_URL must be an HTTPS URL.');
    }

    $separator = strpos($feedUrl, '?') === false ? '?' : '&';
    $url = $feedUrl . $separator . http_build_query(['symbols' => implode(',', $symbols)], '', '&', PHP_QUERY_RFC3986);
    $headers = ['Accept: application/json'];
    $token = stock_alert_env('STOCK_RESEARCH_FEED_TOKEN');
    if ($token !== null && $token !== '') {
        $headers[] = 'Authorization: Bearer ' . $token;
    }

    $curl = curl_init($url);
    curl_setopt_array($curl, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_FOLLOWLOCATION => true,
        CURLOPT_PROTOCOLS => CURLPROTO_HTTPS,
        CURLOPT_REDIR_PROTOCOLS => CURLPROTO_HTTPS,
        CURLOPT_CONNECTTIMEOUT => 10,
        CURLOPT_TIMEOUT => 60,
        CURLOPT_SSL_VERIFYPEER => true,
        CURLOPT_SSL_VERIFYHOST => 2,
        CURLOPT_USERAGENT => 'StockAnalyzerResearch/1.0',
        CURLOPT_HTTPHEADER => $headers,
    ]);
    $body = curl_exec($curl);
    $curlError = curl_error($curl);
    $status = (int)curl_getinfo($curl, CURLINFO_HTTP_CODE);
    curl_close($curl);

    if ($body === false) {
        throw new RuntimeException('Research feed request failed' . ($curlError !== '' ? ': ' . $curlError : '.'));
    }
    if ($status < 200 || $status >= 300) {
        throw new RuntimeException('Research feed returned HTTP ' . $status . '.');
    }

    return stock_research_parse_feed($body, $symbols);
}

function stock_research_news_terms(array $stock): array
{
    $terms = [];
    $companyName = trim((string)($stock['company_name'] ?? ''));
    if ($companyName !== '') {
        $terms[] = $companyName;
    }

    $symbol = (string)($stock['symbol'] ?? '');
    $ticker = preg_replace('/\.(NS|BO)$/', '', $symbol);
    if (is_string($ticker) && strlen($ticker) >= 3) {
        $terms[] = $ticker;
    }

    $unique = [];
    foreach ($terms as $term) {
        $term = trim(preg_replace('/[\r\n\t]+/', ' ', $term));
        if ($term !== '') {
            $unique[$term] = $term;
        }
    }

    return array_values($unique);
}

function stock_research_article_matches_term(string $text, string $term): bool
{
    $pattern = '/(?<![\pL\pN])' . preg_quote($term, '/') . '(?![\pL\pN])/iu';
    return preg_match($pattern, $text) === 1;
}

function stock_research_map_news_articles(array $articles, array $stocks): array
{
    $mapped = [];
    foreach ($articles as $article) {
        if (!is_array($article)) {
            continue;
        }
        $title = $article['title'] ?? null;
        $url = $article['url'] ?? null;
        if (!is_string($title) || trim($title) === '' || !is_string($url) || strlen($url) > 2048) {
            continue;
        }
        $urlParts = parse_url($url);
        if (!is_array($urlParts) || strtolower((string)($urlParts['scheme'] ?? '')) !== 'https'
            || empty($urlParts['host'])) {
            continue;
        }

        $description = $article['description'] ?? '';
        $description = is_string($description) ? $description : '';
        $searchText = $title . ' ' . $description;
        foreach ($stocks as $stock) {
            $symbol = (string)($stock['symbol'] ?? '');
            foreach (stock_research_news_terms($stock) as $term) {
                if (!stock_research_article_matches_term($searchText, $term)) {
                    continue;
                }

                $sourceData = $article['source'] ?? [];
                $source = is_array($sourceData) ? ($sourceData['name'] ?? 'NewsAPI') : 'NewsAPI';
                $author = $article['author'] ?? '';
                $publishedAt = $article['publishedAt'] ?? null;
                if (is_string($publishedAt)) {
                    try {
                        $publishedAt = (new DateTimeImmutable($publishedAt))->format(DATE_ATOM);
                    } catch (Throwable $error) {
                        $publishedAt = null;
                    }
                } else {
                    $publishedAt = null;
                }

                $key = $symbol . "\0" . $url;
                $mapped[$key] = [
                    'symbol' => $symbol,
                    'source' => is_string($source) && trim($source) !== '' ? trim($source) : 'NewsAPI',
                    'author' => is_string($author) ? substr(trim($author), 0, 250) : '',
                    'title' => substr(trim($title), 0, 1000),
                    'description' => substr(trim($description), 0, 4000),
                    'source_url' => $url,
                    'published_at' => $publishedAt,
                ];
                break;
            }
        }
    }

    return array_values($mapped);
}

function stock_research_fetch_news(array $stocks, string $apiKey): array
{
    if (!function_exists('curl_init')) {
        throw new RuntimeException('PHP cURL extension is required for NewsAPI.');
    }

    $batches = [];
    $current = [];
    $currentTerms = [];
    foreach ($stocks as $stock) {
        $terms = stock_research_news_terms($stock);
        if ($terms === []) {
            continue;
        }
        $candidateTerms = array_values(array_unique(array_merge($currentTerms, $terms)));
        $candidateQuery = '(' . implode(' OR ', array_map(
            static fn(string $term): string => '"' . str_replace('"', '\\"', $term) . '"',
            $candidateTerms
        )) . ')';
        if ($current !== [] && (count($current) >= 10 || strlen($candidateQuery) > 450)) {
            $batches[] = ['stocks' => $current, 'terms' => $currentTerms];
            $current = [];
            $currentTerms = [];
        }
        $current[] = $stock;
        $currentTerms = array_values(array_unique(array_merge($currentTerms, $terms)));
    }
    if ($current !== []) {
        $batches[] = ['stocks' => $current, 'terms' => $currentTerms];
    }

    $allArticles = [];
    $errors = [];
    $successfulBatches = 0;
    $fromDate = (new DateTimeImmutable('now', new DateTimeZone('Asia/Kolkata')))
        ->modify('-14 days')
        ->format('Y-m-d');

    foreach ($batches as $batchIndex => $batch) {
        $query = '(' . implode(' OR ', array_map(
            static fn(string $term): string => '"' . str_replace('"', '\\"', $term) . '"',
            $batch['terms']
        )) . ')';
        $url = 'https://newsapi.org/v2/everything?' . http_build_query([
            'q' => $query,
            'from' => $fromDate,
            'sortBy' => 'publishedAt',
            'language' => 'en',
            'pageSize' => 100,
        ], '', '&', PHP_QUERY_RFC3986);
        $curl = curl_init($url);
        curl_setopt_array($curl, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_FOLLOWLOCATION => true,
            CURLOPT_PROTOCOLS => CURLPROTO_HTTPS,
            CURLOPT_REDIR_PROTOCOLS => CURLPROTO_HTTPS,
            CURLOPT_CONNECTTIMEOUT => 10,
            CURLOPT_TIMEOUT => 30,
            CURLOPT_SSL_VERIFYPEER => true,
            CURLOPT_SSL_VERIFYHOST => 2,
            CURLOPT_USERAGENT => 'StockAnalyzerResearch/1.0',
            CURLOPT_HTTPHEADER => ['Accept: application/json', 'X-Api-Key: ' . $apiKey],
        ]);
        $body = curl_exec($curl);
        $curlError = curl_error($curl);
        $status = (int)curl_getinfo($curl, CURLINFO_HTTP_CODE);
        curl_close($curl);

        if ($body === false || $status < 200 || $status >= 300) {
            $errors[] = 'NewsAPI batch ' . ($batchIndex + 1) . ' failed'
                . ($status > 0 ? ' with HTTP ' . $status : '')
                . ($curlError !== '' ? ': ' . $curlError : '.');
            continue;
        }
        if (strlen($body) > 5000000) {
            $errors[] = 'NewsAPI batch ' . ($batchIndex + 1) . ' exceeded the 5 MB response limit.';
            continue;
        }

        try {
            $payload = json_decode($body, true, 512, JSON_THROW_ON_ERROR);
        } catch (Throwable $error) {
            $errors[] = 'NewsAPI batch ' . ($batchIndex + 1) . ' returned invalid JSON.';
            continue;
        }
        if (!is_array($payload) || ($payload['status'] ?? '') !== 'ok' || !is_array($payload['articles'] ?? null)) {
            $errors[] = 'NewsAPI batch ' . ($batchIndex + 1) . ' returned an invalid response.';
            continue;
        }

        $successfulBatches++;
        $allArticles = array_merge(
            $allArticles,
            stock_research_map_news_articles($payload['articles'], $batch['stocks'])
        );
    }

    $deduplicated = [];
    foreach ($allArticles as $article) {
        $deduplicated[$article['symbol'] . "\0" . $article['source_url']] = $article;
    }

    return [
        'articles' => array_values($deduplicated),
        'batch_count' => count($batches),
        'successful_batches' => $successfulBatches,
        'errors' => $errors,
    ];
}

function stock_research_run(): array
{
    $connection = stock_alert_db();
    $runId = null;
    $lockAcquired = false;

    try {
        stock_master_ensure_schema($connection);
        $stockResult = stock_alert_query(
            $connection,
            'SELECT symbol, company_name FROM stock_master WHERE enabled = TRUE ORDER BY id'
        );
        $stocks = pg_fetch_all($stockResult) ?: [];
        if ($stocks === []) {
            throw new RuntimeException('No active stocks are available in stock_master.');
        }
        $symbols = array_map(static fn(array $stock): string => $stock['symbol'], $stocks);
        if (!pg_query_params($connection, 'SELECT pg_advisory_lock($1)', ['771903141'])) {
            throw new RuntimeException('Could not acquire the research-run lock.');
        }
        $lockAcquired = true;

        $tradingDate = (new DateTimeImmutable('now', new DateTimeZone('Asia/Kolkata')))->format('Y-m-d');
        $runResult = stock_alert_query(
            $connection,
            "INSERT INTO stock_research_runs (trading_date, status)
             VALUES ($1::date, 'running') RETURNING id",
            [$tradingDate]
        );
        $run = pg_fetch_assoc($runResult);
        if ($run === false) {
            throw new RuntimeException('Could not create a research refresh record.');
        }
        $runId = (string)$run['id'];

        $recommendations = [];
        $articles = [];
        $sourceStatus = ['ratings' => 'not_configured', 'news' => 'not_configured'];
        $sourceErrors = [];

        $feedUrl = stock_alert_env('STOCK_RESEARCH_FEED_URL');
        if ($feedUrl !== null && $feedUrl !== '') {
            try {
                $recommendations = stock_research_fetch_feed($symbols);
                $sourceStatus['ratings'] = 'completed';
            } catch (Throwable $error) {
                $sourceStatus['ratings'] = 'failed';
                $sourceErrors[] = 'Ratings feed: ' . $error->getMessage();
            }
        }

        $newsApiKey = stock_alert_env('NEWSAPI_KEY');
        if ($newsApiKey !== null && $newsApiKey !== '') {
            try {
                $newsResult = stock_research_fetch_news($stocks, $newsApiKey);
                $articles = $newsResult['articles'];
                if ($newsResult['successful_batches'] > 0) {
                    $sourceStatus['news'] = $newsResult['errors'] === [] ? 'completed' : 'partial';
                } else {
                    $sourceStatus['news'] = 'failed';
                }
                $sourceErrors = array_merge($sourceErrors, $newsResult['errors']);
            } catch (Throwable $error) {
                $sourceStatus['news'] = 'failed';
                $sourceErrors[] = 'NewsAPI: ' . $error->getMessage();
            }
        }

        if (!in_array($sourceStatus['ratings'], ['completed'], true)
            && !in_array($sourceStatus['news'], ['completed', 'partial'], true)) {
            throw new RuntimeException($sourceErrors !== []
                ? implode(' ', $sourceErrors)
                : 'No research source is configured. Set NEWSAPI_KEY or STOCK_RESEARCH_FEED_URL.');
        }

        stock_alert_query($connection, 'BEGIN');
        foreach ($recommendations as $recommendation) {
            stock_alert_query(
                $connection,
                'INSERT INTO stock_research_recommendations
                    (run_id, symbol, source_name, analyst_name, recommendation, target_price,
                     currency, horizon, published_at, source_url)
                 VALUES ($1::bigint, $2, $3, $4, $5, $6, $7, $8, $9::timestamptz, $10)',
                [
                    $runId,
                    $recommendation['symbol'],
                    $recommendation['source'],
                    $recommendation['analyst'],
                    $recommendation['recommendation'],
                    $recommendation['target_price'],
                    $recommendation['currency'],
                    $recommendation['horizon'],
                    $recommendation['published_at'],
                    $recommendation['source_url'],
                ]
            );
        }

        foreach ($articles as $article) {
            stock_alert_query(
                $connection,
                'INSERT INTO stock_research_articles
                    (run_id, symbol, source_name, author, title, description, source_url, published_at)
                 VALUES ($1::bigint, $2, $3, $4, $5, $6, $7, $8::timestamptz)
                 ON CONFLICT (run_id, symbol, source_url) DO NOTHING',
                [
                    $runId,
                    $article['symbol'],
                    $article['source'],
                    $article['author'],
                    $article['title'],
                    $article['description'],
                    $article['source_url'],
                    $article['published_at'],
                ]
            );
        }

        stock_alert_query(
            $connection,
            "UPDATE stock_research_runs
             SET status = 'completed', completed_at = NOW(), error = $2
             WHERE id = $1::bigint",
            [$runId, $sourceErrors === [] ? null : substr(implode(' ', $sourceErrors), 0, 1000)]
        );
        stock_alert_query(
            $connection,
            "DELETE FROM stock_research_runs
             WHERE trading_date NOT IN (
                 SELECT trading_date
                 FROM (
                     SELECT DISTINCT trading_date
                     FROM stock_research_runs
                     WHERE status = 'completed'
                     ORDER BY trading_date DESC
                     LIMIT 5
                 ) AS retained_dates
             )"
        );
        stock_alert_query($connection, 'COMMIT');
        if ($sourceErrors !== []) {
            stock_alert_log('Stock research refresh completed with source errors', null, [
                'run_id' => $runId,
                'source_status' => $sourceStatus,
                'errors' => $sourceErrors,
            ]);
        }

        return [
            'status' => 'completed',
            'run_id' => $runId,
            'trading_date' => $tradingDate,
            'recommendation_count' => count($recommendations),
            'article_count' => count($articles),
            'stock_count' => count($symbols),
            'source_status' => $sourceStatus,
            'source_errors' => $sourceErrors,
        ];
    } catch (Throwable $error) {
        @pg_query($connection, 'ROLLBACK');
        if ($runId !== null) {
            stock_alert_query(
                $connection,
                "UPDATE stock_research_runs
                 SET status = 'failed', completed_at = NOW(), error = $2
                 WHERE id = $1::bigint",
                [$runId, substr($error->getMessage(), 0, 1000)]
            );
        }
        throw $error;
    } finally {
        if ($lockAcquired) {
            @pg_query_params($connection, 'SELECT pg_advisory_unlock($1)', ['771903141']);
        }
        pg_close($connection);
    }
}

function stock_research_list_for_symbol($connection, string $symbol): array
{
    $result = stock_alert_query(
        $connection,
        "SELECT DISTINCT ON (
                recommendation.source_name,
                recommendation.analyst_name,
                COALESCE(recommendation.source_url, '')
            )
            recommendation.source_name,
            recommendation.analyst_name,
            recommendation.recommendation,
            recommendation.target_price::text AS target_price,
            recommendation.currency,
            recommendation.horizon,
            recommendation.published_at,
            recommendation.source_url,
            run.trading_date,
            run.started_at AS fetched_at
         FROM stock_research_recommendations AS recommendation
         JOIN stock_research_runs AS run ON run.id = recommendation.run_id
         WHERE recommendation.symbol = $1
           AND run.status = 'completed'
           AND run.trading_date IN (
               SELECT trading_date
               FROM (
                   SELECT DISTINCT trading_date
                   FROM stock_research_runs
                   WHERE status = 'completed'
                   ORDER BY trading_date DESC
                   LIMIT 5
               ) AS retained_dates
           )
         ORDER BY
            recommendation.source_name,
            recommendation.analyst_name,
            COALESCE(recommendation.source_url, ''),
            run.started_at DESC",
        [$symbol]
    );

    $rows = pg_fetch_all($result) ?: [];
    usort($rows, static function (array $left, array $right): int {
        $leftDate = $left['published_at'] ?? $left['fetched_at'];
        $rightDate = $right['published_at'] ?? $right['fetched_at'];
        return strcmp((string)$rightDate, (string)$leftDate);
    });

    return $rows;
}

function stock_research_articles_for_symbol($connection, string $symbol): array
{
    $result = stock_alert_query(
        $connection,
        "SELECT DISTINCT ON (article.source_url)
            article.source_name,
            article.author,
            article.title,
            article.description,
            article.source_url,
            article.published_at,
            run.trading_date,
            run.started_at AS fetched_at
         FROM stock_research_articles AS article
         JOIN stock_research_runs AS run ON run.id = article.run_id
         WHERE article.symbol = $1
           AND run.status = 'completed'
           AND run.trading_date IN (
               SELECT trading_date
               FROM (
                   SELECT DISTINCT trading_date
                   FROM stock_research_runs
                   WHERE status = 'completed'
                   ORDER BY trading_date DESC
                   LIMIT 5
               ) AS retained_dates
           )
         ORDER BY article.source_url, run.started_at DESC",
        [$symbol]
    );

    $rows = pg_fetch_all($result) ?: [];
    usort($rows, static function (array $left, array $right): int {
        $leftDate = $left['published_at'] ?? $left['fetched_at'];
        $rightDate = $right['published_at'] ?? $right['fetched_at'];
        return strcmp((string)$rightDate, (string)$leftDate);
    });

    return $rows;
}
