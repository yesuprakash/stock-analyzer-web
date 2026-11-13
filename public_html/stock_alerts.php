<?php
declare(strict_types=1);

$scriptDirectory = str_replace('\\', '/', dirname($_SERVER['SCRIPT_NAME'] ?? '/'));
$sessionPath = rtrim($scriptDirectory, '/') . '/';
header('Content-Type: application/json; charset=UTF-8');
header('Cache-Control: no-store');
set_time_limit(0);

session_set_cookie_params([
    'httponly' => true,
    'secure' => !empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off',
    'samesite' => 'Strict',
    'path' => $sessionPath,
]);
session_start();

define('PRICE_PERFORMANCE_LIBRARY_ONLY', true);
require_once __DIR__ . DIRECTORY_SEPARATOR . 'price_performance.php';
require_once stock_analyzer_backend_file('stock_alert_lib.php');
stock_alert_log_fatal_shutdown('Stock alerts fatal PHP error');

function stock_alert_api_respond(array $payload, int $status = 200): void
{
    http_response_code($status);
    echo json_encode($payload, JSON_INVALID_UTF8_SUBSTITUTE | JSON_PARTIAL_OUTPUT_ON_ERROR);
    exit;
}

function stock_alert_api_body(): array
{
    $body = json_decode((string)file_get_contents('php://input'), true);
    return is_array($body) ? $body : [];
}

function stock_alert_api_csrf(array $body): void
{
    $token = $body['csrf'] ?? '';
    if (!is_string($token) || !isset($_SESSION['stock_alert_csrf'])
        || !hash_equals($_SESSION['stock_alert_csrf'], $token)) {
        stock_alert_api_respond(['error' => 'Your session expired. Reload the page and sign in again.'], 403);
    }
}

function stock_alert_api_bool(array $body, string $key): bool
{
    if (!array_key_exists($key, $body)) {
        throw new InvalidArgumentException("Missing setting: $key.");
    }
    $value = filter_var($body[$key], FILTER_VALIDATE_BOOLEAN, FILTER_NULL_ON_FAILURE);
    if ($value === null) {
        throw new InvalidArgumentException("Invalid setting: $key.");
    }
    return $value;
}

try {
    $action = $_GET['action'] ?? '';
    if ($action === 'login') {
        $body = stock_alert_api_body();
        $password = stock_alert_env('STOCK_ALERT_ADMIN_PASSWORD');
        if ($password === null || $password === '') {
            stock_alert_api_respond([
                'error' => 'Set STOCK_ALERT_ADMIN_PASSWORD in the project .env before signing in.',
                'admin_configured' => false,
            ], 503);
        }

        $now = time();
        if (($_SESSION['stock_alert_lock_until'] ?? 0) > $now) {
            stock_alert_api_respond(['error' => 'Too many sign-in attempts. Wait a few minutes and try again.'], 429);
        }
        $provided = is_string($body['password'] ?? null) ? $body['password'] : '';
        if (!hash_equals($password, $provided)) {
            $_SESSION['stock_alert_login_attempts'] = ($_SESSION['stock_alert_login_attempts'] ?? 0) + 1;
            if ($_SESSION['stock_alert_login_attempts'] >= 5) {
                $_SESSION['stock_alert_lock_until'] = $now + 300;
                $_SESSION['stock_alert_login_attempts'] = 0;
            }
            stock_alert_api_respond(['error' => 'Incorrect management password.'], 401);
        }

        session_regenerate_id(true);
        $_SESSION['stock_alert_authenticated'] = true;
        $_SESSION['stock_alert_login_attempts'] = 0;
        $_SESSION['stock_alert_csrf'] = bin2hex(random_bytes(32));
        stock_alert_api_respond(['authenticated' => true, 'csrf' => $_SESSION['stock_alert_csrf']]);
    }

    if ($action === 'logout') {
        $body = stock_alert_api_body();
        stock_alert_api_csrf($body);
        $_SESSION = [];
        session_destroy();
        stock_alert_api_respond(['authenticated' => false]);
    }

    if (($_SESSION['stock_alert_authenticated'] ?? false) !== true) {
        stock_alert_api_respond(['authenticated' => false], 401);
    }

    if (!isset($_SESSION['stock_alert_csrf'])) {
        $_SESSION['stock_alert_csrf'] = bin2hex(random_bytes(32));
    }

    if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
        stock_alert_api_csrf(stock_alert_api_body());
    }

    $connection = stock_alert_db();
    stock_alert_ensure_schema($connection);

    if ($action === 'state') {
        $recipientsResult = stock_alert_query(
            $connection,
            'SELECT id, email, enabled, alert_rises, alert_falls, created_at
             FROM stock_alert_recipients ORDER BY id'
        );
        $recipientRows = pg_fetch_all($recipientsResult) ?: [];
        foreach ($recipientRows as &$recipient) {
            foreach (['enabled', 'alert_rises', 'alert_falls'] as $field) {
                $recipient[$field] = $recipient[$field] === 't';
            }
        }
        unset($recipient);

        stock_alert_api_respond([
            'authenticated' => true,
            'csrf' => $_SESSION['stock_alert_csrf'],
            'settings' => stock_alert_load_settings($connection),
            'recipients' => $recipientRows,
            'history' => stock_alert_event_history($connection),
            'smtp_configured' => stock_alert_smtp_configured(),
        ]);
    }

    if ($action === 'save_settings') {
        $body = stock_alert_api_body();
        $thresholds = [];
        foreach (STOCK_ALERT_PERIODS as $period) {
            $values = $body['thresholds'][(string)$period] ?? null;
            if (!is_array($values)) {
                throw new InvalidArgumentException("Missing thresholds for $period sessions.");
            }
            foreach (['fall', 'rise'] as $direction) {
                $value = filter_var($values[$direction] ?? null, FILTER_VALIDATE_FLOAT);
                if ($value === false || $value < 0.1 || $value > 100) {
                    throw new InvalidArgumentException('Thresholds must be between 0.1% and 100%.');
                }
                $thresholds[(string)$period][$direction] = round((float)$value, 2);
            }
        }
        $encoded = json_encode($thresholds, JSON_THROW_ON_ERROR);
        stock_alert_query(
            $connection,
            'UPDATE stock_alert_settings SET enabled = $1, thresholds = $2::jsonb, updated_at = NOW() WHERE id = 1',
            [stock_alert_api_bool($body, 'enabled') ? 'true' : 'false', $encoded]
        );
        stock_alert_api_respond(['saved' => true]);
    }

    if ($action === 'add_recipient') {
        $body = stock_alert_api_body();
        $email = trim((string)($body['email'] ?? ''));
        if (filter_var($email, FILTER_VALIDATE_EMAIL) === false) {
            throw new InvalidArgumentException('Enter a valid email address.');
        }
        $inserted = stock_alert_query(
            $connection,
            "INSERT INTO stock_alert_recipients (email)
             VALUES ($1) ON CONFLICT DO NOTHING RETURNING id",
            [$email]
        );
        $id = pg_fetch_result($inserted, 0, 'id');
        if ($id === false) {
            stock_alert_api_respond(['error' => 'That email address is already on the recipient list.'], 409);
        }
        stock_alert_api_respond(['added' => true]);
    }

    if ($action === 'update_recipient') {
        $body = stock_alert_api_body();
        $id = filter_var($body['id'] ?? null, FILTER_VALIDATE_INT);
        if ($id === false || $id < 1) {
            throw new InvalidArgumentException('Invalid recipient.');
        }
        stock_alert_query(
            $connection,
            'UPDATE stock_alert_recipients
             SET enabled = $2, alert_rises = $3, alert_falls = $4, updated_at = NOW()
             WHERE id = $1',
            [
                (string)$id,
                stock_alert_api_bool($body, 'enabled') ? 'true' : 'false',
                stock_alert_api_bool($body, 'alert_rises') ? 'true' : 'false',
                stock_alert_api_bool($body, 'alert_falls') ? 'true' : 'false',
            ]
        );
        stock_alert_api_respond(['updated' => true]);
    }

    if ($action === 'delete_recipient') {
        $body = stock_alert_api_body();
        $id = filter_var($body['id'] ?? null, FILTER_VALIDATE_INT);
        if ($id === false || $id < 1) {
            throw new InvalidArgumentException('Invalid recipient.');
        }
        stock_alert_query($connection, 'DELETE FROM stock_alert_recipients WHERE id = $1', [(string)$id]);
        stock_alert_api_respond(['deleted' => true]);
    }

    if ($action === 'test_email') {
        $body = stock_alert_api_body();
        $email = trim((string)($body['email'] ?? ''));
        if (filter_var($email, FILTER_VALIDATE_EMAIL) === false) {
            throw new InvalidArgumentException('Enter a valid destination email address.');
        }
        stock_alert_send_email(
            $email,
            'Stock alerts — test message',
            'This is a test email from the Stock Performance alert manager.'
        );
        stock_alert_api_respond(['sent' => true]);
    }

    if ($action === 'run') {
        $body = stock_alert_api_body();
        if (stock_alert_api_bool($body, 'force_refresh')) {
            stock_alert_api_respond(['error' => 'Use the daily alert check; manual cache bypass is not allowed here.'], 400);
        }
        pg_close($connection);
        $connection = null;
        stock_alert_api_respond(['result' => stock_alert_run()]);
    }

    if ($action === 'force_send') {
        pg_close($connection);
        $connection = null;
        stock_alert_api_respond(['result' => stock_alert_run(true, true)]);
    }

    stock_alert_api_respond(['error' => 'Unknown action.'], 404);
} catch (InvalidArgumentException $error) {
    stock_alert_api_respond(['error' => $error->getMessage()], 400);
} catch (Throwable $error) {
    stock_alert_log('Stock alert manager request failed', $error, [
        'action' => (string)($_GET['action'] ?? ''),
    ]);
    error_log('Stock alert manager error: ' . $error->getMessage());
    stock_alert_api_respond(['error' => $error->getMessage()], 500);
} finally {
    if (isset($connection) && $connection !== null && function_exists('pg_close')) {
        @pg_close($connection);
    }
}
