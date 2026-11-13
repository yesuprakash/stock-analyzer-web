<?php
declare(strict_types=1);

const STOCK_ALERT_PERIODS = [1, 3, 5];

function stock_alert_log(string $event, ?Throwable $error = null, array $context = []): void
{
    $logDirectory = dirname(__DIR__) . DIRECTORY_SEPARATOR . 'logs';
    if (!is_dir($logDirectory) && !@mkdir($logDirectory, 0700, true) && !is_dir($logDirectory)) {
        error_log('Unable to create the Stock Analyzer log directory.');
        return;
    }

    $entry = [
        'timestamp' => gmdate(DATE_ATOM),
        'level' => 'error',
        'event' => $event,
        'context' => $context,
    ];
    if ($error !== null) {
        $entry['exception'] = [
            'class' => get_class($error),
            'message' => $error->getMessage(),
            'file' => $error->getFile(),
            'line' => $error->getLine(),
            'trace' => $error->getTraceAsString(),
        ];
    }

    $line = json_encode($entry, JSON_INVALID_UTF8_SUBSTITUTE | JSON_UNESCAPED_SLASHES);
    $logFile = $logDirectory . DIRECTORY_SEPARATOR . 'stock_analyzer_php.log';
    if ($line === false || @file_put_contents($logFile, $line . PHP_EOL, FILE_APPEND | LOCK_EX) === false) {
        error_log('Unable to write the Stock Analyzer custom log: ' . $event);
        return;
    }
    @chmod($logFile, 0600);
}

function stock_alert_log_fatal_shutdown(string $event): void
{
    register_shutdown_function(static function () use ($event): void {
        $error = error_get_last();
        if ($error !== null && in_array($error['type'], [E_ERROR, E_PARSE, E_CORE_ERROR, E_COMPILE_ERROR], true)) {
            stock_alert_log($event, null, [
                'php_error_type' => $error['type'],
                'message' => $error['message'],
                'file' => $error['file'],
                'line' => $error['line'],
            ]);
        }
    });
}

function stock_alert_env(string $key, ?string $default = null): ?string
{
    $value = getenv($key);
    if ($value !== false) {
        return $value;
    }

    static $fileValues = null;
    if ($fileValues === null) {
        $envFile = dirname(__DIR__) . DIRECTORY_SEPARATOR . '.env';
        $fileValues = is_file($envFile) ? parse_ini_file($envFile, false, INI_SCANNER_RAW) : [];
        if ($fileValues === false) {
            throw new RuntimeException('Unable to parse the project .env file.');
        }
    }

    return isset($fileValues[$key]) ? (string)$fileValues[$key] : $default;
}

function stock_alert_default_thresholds(): array
{
    return [
        '1' => ['fall' => 5.0, 'rise' => 5.0],
        '3' => ['fall' => 8.0, 'rise' => 8.0],
        '5' => ['fall' => 12.0, 'rise' => 12.0],
    ];
}

function stock_alert_db()
{
    if (!function_exists('pg_connect')) {
        throw new RuntimeException('PHP PostgreSQL extension (pgsql) is required.');
    }

    $required = ['DB_NAME', 'DB_USER', 'DB_PASSWORD'];
    $values = [];
    foreach ($required as $key) {
        $values[$key] = stock_alert_env($key);
        if ($values[$key] === null || $values[$key] === '') {
            throw new RuntimeException("Database configuration $key is missing.");
        }
    }
    $values['DB_HOST'] = stock_alert_env('DB_HOST', 'localhost') ?? 'localhost';
    $values['DB_PORT'] = stock_alert_env('DB_PORT', '5432') ?? '5432';

    $connectionInfo = [];
    foreach ([
        'host' => $values['DB_HOST'],
        'port' => $values['DB_PORT'],
        'dbname' => $values['DB_NAME'],
        'user' => $values['DB_USER'],
        'password' => $values['DB_PASSWORD'],
    ] as $key => $value) {
        $escaped = str_replace(['\\', "'"], ['\\\\', "\\'"], (string)$value);
        $connectionInfo[] = $key . "='" . $escaped . "'";
    }

    $connection = @pg_connect(implode(' ', $connectionInfo), PGSQL_CONNECT_FORCE_NEW);
    if ($connection === false) {
        throw new RuntimeException('Could not connect to the configured PostgreSQL database.');
    }
    if (pg_set_client_encoding($connection, 'UTF8') !== 0) {
        pg_close($connection);
        throw new RuntimeException('Could not set PostgreSQL client encoding.');
    }

    return $connection;
}

function stock_alert_query($connection, string $sql, array $params = [])
{
    $result = pg_query_params($connection, $sql, $params);
    if ($result === false) {
        throw new RuntimeException('Alert database operation failed: ' . pg_last_error($connection));
    }
    return $result;
}

function stock_alert_ensure_schema($connection): void
{
    $queries = [
        "CREATE TABLE IF NOT EXISTS stock_alert_recipients (
            id BIGSERIAL PRIMARY KEY,
            email TEXT NOT NULL,
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            alert_rises BOOLEAN NOT NULL DEFAULT TRUE,
            alert_falls BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )",
        "CREATE UNIQUE INDEX IF NOT EXISTS stock_alert_recipients_email_lower_uq
            ON stock_alert_recipients (LOWER(email))",
        "CREATE TABLE IF NOT EXISTS stock_alert_settings (
            id SMALLINT PRIMARY KEY CHECK (id = 1),
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            thresholds JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )",
        "CREATE TABLE IF NOT EXISTS stock_alert_events (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL CHECK (direction IN ('rise', 'fall')),
            period_sessions SMALLINT NOT NULL CHECK (period_sessions IN (1, 3, 5)),
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            change_pct NUMERIC(10, 4) NOT NULL,
            start_close NUMERIC(18, 4) NOT NULL,
            end_close NUMERIC(18, 4) NOT NULL,
            detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (symbol, direction, period_sessions, end_date)
        )",
        "CREATE TABLE IF NOT EXISTS stock_alert_deliveries (
            id BIGSERIAL PRIMARY KEY,
            event_id BIGINT NOT NULL REFERENCES stock_alert_events(id) ON DELETE CASCADE,
            recipient_id BIGINT NOT NULL REFERENCES stock_alert_recipients(id) ON DELETE CASCADE,
            status TEXT NOT NULL CHECK (status IN ('sent', 'failed')),
            error TEXT,
            sent_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (event_id, recipient_id)
        )",
        "CREATE TABLE IF NOT EXISTS stock_alert_runs (
            run_date DATE PRIMARY KEY,
            status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
            event_count INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMPTZ
        )",
    ];

    foreach ($queries as $query) {
        stock_alert_query($connection, $query);
    }

    $defaults = json_encode(stock_alert_default_thresholds(), JSON_THROW_ON_ERROR);
    stock_alert_query(
        $connection,
        "INSERT INTO stock_alert_settings (id, enabled, thresholds)
         VALUES (1, TRUE, $1::jsonb) ON CONFLICT (id) DO NOTHING",
        [$defaults]
    );
}

function stock_alert_load_settings($connection): array
{
    $result = stock_alert_query(
        $connection,
        'SELECT enabled, thresholds FROM stock_alert_settings WHERE id = 1'
    );
    $row = pg_fetch_assoc($result);
    $thresholds = json_decode($row['thresholds'] ?? '{}', true);
    if (!is_array($thresholds)) {
        $thresholds = stock_alert_default_thresholds();
    }

    return [
        'enabled' => ($row['enabled'] ?? 'f') === 't',
        'thresholds' => $thresholds,
    ];
}

function stock_alert_cache_path(): string
{
    if (!empty($_SERVER['DOCUMENT_ROOT'])) {
        $documentRoot = $_SERVER['DOCUMENT_ROOT'];
    } else {
        $projectRoot = dirname(__DIR__);
        $serverDocumentRoot = dirname($projectRoot) . DIRECTORY_SEPARATOR . 'public_html';
        $documentRoot = is_dir($serverDocumentRoot) ? $serverDocumentRoot : dirname($projectRoot);
    }
    $privateRoot = dirname(rtrim($documentRoot, "\\/")) . DIRECTORY_SEPARATOR . 'Stock_Analyzer_cache';
    return $privateRoot . DIRECTORY_SEPARATOR . 'price_performance' . DIRECTORY_SEPARATOR . 'daily_cache.json';
}

function stock_alert_refresh_cache(bool $forceRefresh, array $symbols): array
{
    $cacheFile = stock_alert_cache_path();
    $cacheDir = dirname($cacheFile);
    if (!is_dir($cacheDir) && !mkdir($cacheDir, 0700, true) && !is_dir($cacheDir)) {
        throw new RuntimeException('Unable to create the private price cache directory.');
    }
    $lock = fopen($cacheFile . '.lock', 'c');
    if ($lock === false || !flock($lock, LOCK_EX)) {
        throw new RuntimeException('Unable to lock the daily price cache.');
    }

    try {
        $saved = is_file($cacheFile) ? json_decode((string)file_get_contents($cacheFile), true) : [];
        $cache = is_array($saved) ? $saved : [];
        $today = (new DateTimeImmutable('now', new DateTimeZone('Asia/Kolkata')))->format('Y-m-d');
        $requested = [];
        foreach ($symbols as $symbol) {
            $lastAttempt = $cache[$symbol]['attempted_date'] ?? $cache[$symbol]['fetched_date'] ?? null;
            if ($forceRefresh || $lastAttempt !== $today) {
                $requested[] = $symbol;
            }
        }

        $failures = [];
        foreach (array_chunk($requested, BATCH_SIZE) as $symbols) {
            $results = fetchYahooBatch($symbols);
            foreach ($results as $symbol => $result) {
                if (isset($result['data'])) {
                    $cache[$symbol] = $result['data'] + ['attempted_date' => $today];
                } else {
                    $cache[$symbol] ??= [];
                    $cache[$symbol]['attempted_date'] = $today;
                    $cache[$symbol]['last_error'] = $result['error'] ?? 'Price refresh failed.';
                    $failures[$symbol] = $cache[$symbol]['last_error'];
                }
            }
        }

        if ($requested !== []) {
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

        return ['cache' => $cache, 'today' => $today, 'requested' => count($requested), 'failures' => $failures];
    } finally {
        flock($lock, LOCK_UN);
        fclose($lock);
    }
}

function stock_alert_smtp_configured(): bool
{
    return stock_alert_env('STOCK_ALERT_SMTP_HOST') !== null
        && stock_alert_env('STOCK_ALERT_SMTP_USER') !== null
        && stock_alert_env('STOCK_ALERT_SMTP_PASSWORD') !== null
        && stock_alert_env('STOCK_ALERT_FROM_EMAIL') !== null
        && stock_alert_env('STOCK_ALERT_SMTP_HOST') !== ''
        && stock_alert_env('STOCK_ALERT_SMTP_USER') !== ''
        && stock_alert_env('STOCK_ALERT_SMTP_PASSWORD') !== ''
        && filter_var(stock_alert_env('STOCK_ALERT_FROM_EMAIL'), FILTER_VALIDATE_EMAIL) !== false;
}

function stock_alert_send_email(string $recipient, string $subject, string $body): void
{
    if (!stock_alert_smtp_configured()) {
        throw new RuntimeException('SMTP is not configured in the project .env file.');
    }

    $host = stock_alert_env('STOCK_ALERT_SMTP_HOST');
    $port = (int)(stock_alert_env('STOCK_ALERT_SMTP_PORT', '587') ?? '587');
    $encryption = strtolower(stock_alert_env('STOCK_ALERT_SMTP_ENCRYPTION', 'tls') ?? 'tls');
    if (!in_array($encryption, ['tls', 'ssl'], true) || $port < 1 || $port > 65535) {
        throw new RuntimeException('SMTP encryption or port configuration is invalid.');
    }

    $fromEmail = stock_alert_env('STOCK_ALERT_FROM_EMAIL');
    $fromName = stock_alert_env('STOCK_ALERT_FROM_NAME', 'Stock Performance Alerts') ?? 'Stock Performance Alerts';
    $fromName = preg_replace('/[\r\n]+/', ' ', $fromName);
    $subject = preg_replace('/[\r\n]+/', ' ', $subject);
    $message = implode("\r\n", [
        'Date: ' . date(DATE_RFC2822),
        'To: ' . $recipient,
        'From: ' . mb_encode_mimeheader($fromName, 'UTF-8') . ' <' . $fromEmail . '>',
        'Subject: ' . mb_encode_mimeheader($subject, 'UTF-8'),
        'MIME-Version: 1.0',
        'Content-Type: text/plain; charset=UTF-8',
        'Content-Transfer-Encoding: 8bit',
        '',
        $body,
        '',
    ]);

    $url = ($encryption === 'ssl' ? 'smtps://' : 'smtp://') . $host . ':' . $port;
    $curl = curl_init($url);
    curl_setopt_array($curl, [
        CURLOPT_UPLOAD => true,
        CURLOPT_PROTOCOLS => CURLPROTO_SMTP | CURLPROTO_SMTPS,
        CURLOPT_USERNAME => stock_alert_env('STOCK_ALERT_SMTP_USER'),
        CURLOPT_PASSWORD => stock_alert_env('STOCK_ALERT_SMTP_PASSWORD'),
        CURLOPT_MAIL_FROM => '<' . $fromEmail . '>',
        CURLOPT_MAIL_RCPT => ['<' . $recipient . '>'],
        CURLOPT_USE_SSL => $encryption === 'tls' ? CURLUSESSL_ALL : CURLUSESSL_NONE,
        CURLOPT_SSL_VERIFYPEER => true,
        CURLOPT_SSL_VERIFYHOST => 2,
        CURLOPT_CONNECTTIMEOUT => 10,
        CURLOPT_TIMEOUT => 30,
        CURLOPT_READFUNCTION => static function ($handle, $stream, int $length) use (&$message): string {
            $chunk = substr($message, 0, $length);
            $message = substr($message, strlen($chunk));
            return $chunk;
        },
    ]);

    $sent = curl_exec($curl);
    $error = curl_error($curl);
    $status = (int)curl_getinfo($curl, CURLINFO_RESPONSE_CODE);
    curl_close($curl);
    if ($sent === false || $status < 200 || $status >= 400) {
        throw new RuntimeException('SMTP delivery failed' . ($error !== '' ? ': ' . $error : " (SMTP status $status)."));
    }
}

function stock_alert_run(bool $forceRefresh = false, bool $forceDelivery = false): array
{
    $connection = stock_alert_db();
    stock_alert_ensure_schema($connection);
    stock_master_ensure_schema($connection);
    $symbols = stock_master_symbols($connection);
    if ($symbols === []) {
        pg_close($connection);
        throw new RuntimeException('No active stocks are available in stock_master.');
    }
    if (!pg_query($connection, 'SELECT pg_advisory_lock(771903140)')) {
        pg_close($connection);
        throw new RuntimeException('Could not acquire the alert-run lock.');
    }

    try {
        $today = (new DateTimeImmutable('now', new DateTimeZone('Asia/Kolkata')))->format('Y-m-d');
        $previousRun = stock_alert_query(
            $connection,
            "SELECT status FROM stock_alert_runs WHERE run_date = $1::date",
            [$today]
        );
        $runRow = pg_fetch_assoc($previousRun);
        $mustRefresh = $forceRefresh || $runRow === false;

        stock_alert_query(
            $connection,
            "INSERT INTO stock_alert_runs (run_date, status, started_at, completed_at, error)
             VALUES ($1::date, 'running', NOW(), NULL, NULL)
             ON CONFLICT (run_date) DO UPDATE SET
               status = 'running', started_at = NOW(), completed_at = NULL, error = NULL",
            [$today]
        );

        $refresh = stock_alert_refresh_cache($mustRefresh, $symbols);
        $settings = stock_alert_load_settings($connection);
        $recipientsResult = stock_alert_query(
            $connection,
            'SELECT id, email, alert_rises, alert_falls
             FROM stock_alert_recipients WHERE enabled = TRUE ORDER BY id'
        );
        $recipients = pg_fetch_all($recipientsResult) ?: [];
        $eventIds = [];
        $newEvents = 0;

        if ($settings['enabled']) {
            foreach ($symbols as $symbol) {
                $entry = $refresh['cache'][$symbol] ?? null;
                if (!is_array($entry) || ($entry['fetched_date'] ?? null) !== $today) {
                    continue;
                }
                $history = $entry['history'] ?? [];
                $lastIndex = count($history) - 1;
                if ($lastIndex < 1 || ($history[$lastIndex]['date'] ?? null) !== $today) {
                    continue;
                }

                foreach (STOCK_ALERT_PERIODS as $period) {
                    $startIndex = $lastIndex - $period;
                    if ($startIndex < 0) {
                        continue;
                    }
                    $start = $history[$startIndex];
                    $end = $history[$lastIndex];
                    if (($start['close'] ?? 0) <= 0 || ($end['close'] ?? 0) <= 0) {
                        continue;
                    }
                    $change = ((float)$end['close'] / (float)$start['close'] - 1) * 100;
                    $threshold = $settings['thresholds'][(string)$period] ?? null;
                    if (!is_array($threshold)) {
                        continue;
                    }

                    $direction = null;
                    if ($change <= -(float)($threshold['fall'] ?? INF)) {
                        $direction = 'fall';
                    } elseif ($change >= (float)($threshold['rise'] ?? INF)) {
                        $direction = 'rise';
                    }
                    if ($direction === null) {
                        continue;
                    }

                    $inserted = stock_alert_query(
                        $connection,
                        "INSERT INTO stock_alert_events
                           (symbol, direction, period_sessions, start_date, end_date, change_pct, start_close, end_close)
                         VALUES ($1, $2, $3, $4::date, $5::date, $6, $7, $8)
                         ON CONFLICT (symbol, direction, period_sessions, end_date) DO NOTHING
                         RETURNING id",
                        [
                            $symbol, $direction, (string)$period, $start['date'], $end['date'],
                            (string)$change, (string)$start['close'], (string)$end['close'],
                        ]
                    );
                    $eventId = pg_fetch_result($inserted, 0, 'id');
                    if ($eventId !== false) {
                        $newEvents++;
                    }
                    if ($eventId === false) {
                        $existing = stock_alert_query(
                            $connection,
                            "SELECT id FROM stock_alert_events
                             WHERE symbol = $1 AND direction = $2 AND period_sessions = $3
                               AND end_date = $4::date",
                            [$symbol, $direction, (string)$period, $end['date']]
                        );
                        $eventId = pg_fetch_result($existing, 0, 'id');
                    }
                    if ($eventId !== false) {
                        $eventIds[(string)$eventId] = true;
                    }
                }
            }
        }

        $deliveryErrors = [];
        $emailsSent = 0;
        foreach ($recipients as $recipient) {
            $eligible = [];
            foreach (array_keys($eventIds) as $eventId) {
                $events = stock_alert_query(
                    $connection,
                    "SELECT e.* FROM stock_alert_events e
                     WHERE e.id = $1
                       AND (($2::boolean AND e.direction = 'rise') OR ($3::boolean AND e.direction = 'fall'))
                       AND ($5::boolean OR NOT EXISTS (
                             SELECT 1 FROM stock_alert_deliveries d
                             WHERE d.event_id = e.id AND d.recipient_id = $4 AND d.status = 'sent'
                           ))
                     ORDER BY e.end_date DESC, e.symbol, e.period_sessions",
                    [
                        $eventId,
                        $recipient['alert_rises'],
                        $recipient['alert_falls'],
                        $recipient['id'],
                        $forceDelivery ? 'true' : 'false',
                    ]
                );
                $eligible = array_merge($eligible, pg_fetch_all($events) ?: []);
            }
            if ($eligible === []) {
                continue;
            }

            $lines = ["Stock movement alerts for $today:", ''];
            foreach ($eligible as $event) {
                $period = (int)$event['period_sessions'];
                $lines[] = sprintf(
                    "%s: %s %s %.2f%% over %d trading session(s) (%s to %s); adjusted close %.2f to %.2f.",
                    $event['symbol'],
                    strtoupper($event['direction']),
                    $event['change_pct'] > 0 ? '+' : '',
                    (float)$event['change_pct'],
                    $period,
                    $event['start_date'],
                    $event['end_date'],
                    (float)$event['start_close'],
                    (float)$event['end_close']
                );
            }

            $error = null;
            try {
                stock_alert_send_email($recipient['email'], "Stock movement alerts — $today", implode("\n", $lines));
                $emailsSent++;
            } catch (Throwable $sendError) {
                stock_alert_log('Stock alert email delivery failed', $sendError);
                $error = $sendError->getMessage();
                $deliveryErrors[] = $recipient['email'] . ': ' . $error;
            }

            foreach ($eligible as $event) {
                stock_alert_query(
                    $connection,
                    "INSERT INTO stock_alert_deliveries (event_id, recipient_id, status, error, sent_at, updated_at)
                     VALUES ($1, $2, $3, $4, CASE WHEN $3 = 'sent' THEN NOW() ELSE NULL END, NOW())
                     ON CONFLICT (event_id, recipient_id) DO UPDATE SET
                       status = EXCLUDED.status, error = EXCLUDED.error,
                       sent_at = EXCLUDED.sent_at, updated_at = NOW()
                     WHERE stock_alert_deliveries.status <> 'sent' OR EXCLUDED.status = 'sent'",
                    [$event['id'], $recipient['id'], $error === null ? 'sent' : 'failed', $error]
                );
            }
        }

        $failureSummary = $deliveryErrors === [] ? null : implode("\n", $deliveryErrors);
        $status = $failureSummary === null ? 'completed' : 'failed';
        stock_alert_query(
            $connection,
            "UPDATE stock_alert_runs
             SET status = $2, event_count = $3, error = $4, completed_at = NOW()
             WHERE run_date = $1::date",
            [$today, $status, (string)$newEvents, $failureSummary]
        );

        return [
            'date' => $today,
            'status' => $status,
            'new_events' => $newEvents,
            'emails_sent' => $emailsSent,
            'price_requests' => $refresh['requested'],
            'price_errors' => count($refresh['failures']),
            'delivery_errors' => $deliveryErrors,
            'refreshed_prices' => $mustRefresh,
        ];
    } catch (Throwable $error) {
        try {
            stock_alert_query(
                $connection,
                "UPDATE stock_alert_runs SET status = 'failed', error = $2, completed_at = NOW()
                 WHERE run_date = $1::date",
                [
                    (new DateTimeImmutable('now', new DateTimeZone('Asia/Kolkata')))->format('Y-m-d'),
                    $error->getMessage(),
                ]
            );
        } catch (Throwable $recordError) {
            stock_alert_log('Unable to record stock alert run failure', $recordError);
            error_log('Unable to record stock alert run failure.');
        }
        throw $error;
    } finally {
        @pg_query($connection, 'SELECT pg_advisory_unlock(771903140)');
        pg_close($connection);
    }
}

function stock_alert_event_history($connection, int $limit = 100): array
{
    $limit = max(1, min(200, $limit));
    $result = stock_alert_query(
        $connection,
        "SELECT e.symbol, e.direction, e.period_sessions, e.start_date, e.end_date,
                e.change_pct, e.start_close, e.end_close, e.detected_at,
                COALESCE(string_agg(r.email || ': ' || d.status, ', ' ORDER BY r.email), 'Not sent') AS deliveries
         FROM stock_alert_events e
         LEFT JOIN stock_alert_deliveries d ON d.event_id = e.id
         LEFT JOIN stock_alert_recipients r ON r.id = d.recipient_id
         GROUP BY e.id
         ORDER BY e.detected_at DESC, e.id DESC
         LIMIT $1",
        [(string)$limit]
    );
    return pg_fetch_all($result) ?: [];
}
