<?php
declare(strict_types=1);

$scriptDirectory = str_replace('\\', '/', dirname($_SERVER['SCRIPT_NAME'] ?? '/'));
$sessionPath = rtrim($scriptDirectory, '/') . '/';
header('Content-Type: application/json; charset=UTF-8');
header('Cache-Control: no-store');
session_set_cookie_params([
    'httponly' => true,
    'secure' => !empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off',
    'samesite' => 'Strict',
    'path' => $sessionPath,
]);
session_start();

require_once __DIR__ . DIRECTORY_SEPARATOR . 'stock_analyzer_bootstrap.php';
require_once stock_analyzer_backend_file('stock_master_lib.php');
stock_alert_log_fatal_shutdown('Stock master fatal PHP error');

function stock_master_respond(array $payload, int $status = 200): void
{
    http_response_code($status);
    echo json_encode($payload, JSON_INVALID_UTF8_SUBSTITUTE | JSON_PARTIAL_OUTPUT_ON_ERROR);
    exit;
}

function stock_master_body(): array
{
    $body = json_decode((string)file_get_contents('php://input'), true);
    return is_array($body) ? $body : [];
}

function stock_master_check_csrf(array $body): void
{
    $token = $body['csrf'] ?? '';
    if (!is_string($token) || !isset($_SESSION['stock_alert_csrf'])
        || !hash_equals($_SESSION['stock_alert_csrf'], $token)) {
        stock_master_respond(['error' => 'Your session expired. Reload the page and sign in again.'], 403);
    }
}

function stock_master_bool(array $body, string $key): bool
{
    if (!array_key_exists($key, $body)) {
        throw new InvalidArgumentException("Missing field: $key.");
    }
    $value = filter_var($body[$key], FILTER_VALIDATE_BOOLEAN, FILTER_NULL_ON_FAILURE);
    if ($value === null) {
        throw new InvalidArgumentException("Invalid field: $key.");
    }
    return $value;
}

$connection = null;
try {
    $action = $_GET['action'] ?? '';
    if ($action === 'login') {
        $body = stock_master_body();
        $password = stock_alert_env('STOCK_ALERT_ADMIN_PASSWORD');
        if ($password === null || $password === '') {
            stock_master_respond(['error' => 'Set STOCK_ALERT_ADMIN_PASSWORD in the project .env first.'], 503);
        }

        $now = time();
        if (($_SESSION['stock_alert_lock_until'] ?? 0) > $now) {
            stock_master_respond(['error' => 'Too many sign-in attempts. Wait a few minutes and try again.'], 429);
        }
        $provided = is_string($body['password'] ?? null) ? $body['password'] : '';
        if (!hash_equals($password, $provided)) {
            $_SESSION['stock_alert_login_attempts'] = ($_SESSION['stock_alert_login_attempts'] ?? 0) + 1;
            if ($_SESSION['stock_alert_login_attempts'] >= 5) {
                $_SESSION['stock_alert_lock_until'] = $now + 300;
                $_SESSION['stock_alert_login_attempts'] = 0;
            }
            stock_master_respond(['error' => 'Incorrect management password.'], 401);
        }

        session_regenerate_id(true);
        $_SESSION['stock_alert_authenticated'] = true;
        $_SESSION['stock_alert_login_attempts'] = 0;
        $_SESSION['stock_alert_csrf'] = bin2hex(random_bytes(32));
        stock_master_respond(['authenticated' => true, 'csrf' => $_SESSION['stock_alert_csrf']]);
    }

    if ($action === 'logout') {
        stock_master_check_csrf(stock_master_body());
        $_SESSION = [];
        session_destroy();
        stock_master_respond(['authenticated' => false]);
    }

    if (($_SESSION['stock_alert_authenticated'] ?? false) !== true) {
        stock_master_respond(['authenticated' => false], 401);
    }
    if (!isset($_SESSION['stock_alert_csrf'])) {
        $_SESSION['stock_alert_csrf'] = bin2hex(random_bytes(32));
    }

    $body = $_SERVER['REQUEST_METHOD'] === 'GET' ? [] : stock_master_body();
    if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
        stock_master_check_csrf($body);
    }

    $connection = stock_alert_db();
    stock_master_ensure_schema($connection);

    if ($action === 'state') {
        $result = stock_alert_query(
            $connection,
            'SELECT id, symbol, company_name, enabled, created_at
             FROM stock_master ORDER BY id'
        );
        $stocks = pg_fetch_all($result) ?: [];
        foreach ($stocks as &$stock) {
            $stock['enabled'] = $stock['enabled'] === 't';
        }
        unset($stock);
        stock_master_respond([
            'csrf' => $_SESSION['stock_alert_csrf'],
            'stocks' => $stocks,
            'active_count' => count(array_filter($stocks, static fn(array $stock): bool => $stock['enabled'])),
        ]);
    }

    if ($action === 'add') {
        $symbol = strtoupper(trim((string)($body['symbol'] ?? '')));
        $name = trim((string)($body['company_name'] ?? ''));
        if (!preg_match('/^[A-Z0-9][A-Z0-9.&-]{0,28}\.(NS|BO)$/', $symbol)) {
            throw new InvalidArgumentException('Use a Yahoo Finance ticker such as RELIANCE.NS or a ticker ending in .BO.');
        }
        if (mb_strlen($name) > 120) {
            throw new InvalidArgumentException('Company name must be 120 characters or fewer.');
        }
        $result = stock_alert_query(
            $connection,
            'INSERT INTO stock_master (symbol, company_name)
             VALUES ($1, $2) ON CONFLICT (symbol) DO NOTHING RETURNING id',
            [$symbol, $name]
        );
        if (pg_fetch_result($result, 0, 'id') === false) {
            stock_master_respond(['error' => 'That ticker already exists. Enable it from the list if it is inactive.'], 409);
        }
        stock_master_respond(['added' => true]);
    }

    if ($action === 'update') {
        $id = filter_var($body['id'] ?? null, FILTER_VALIDATE_INT);
        $name = trim((string)($body['company_name'] ?? ''));
        if ($id === false || $id < 1) {
            throw new InvalidArgumentException('Invalid stock record.');
        }
        if (mb_strlen($name) > 120) {
            throw new InvalidArgumentException('Company name must be 120 characters or fewer.');
        }
        stock_alert_query(
            $connection,
            'UPDATE stock_master SET company_name = $2, enabled = $3, updated_at = NOW() WHERE id = $1',
            [(string)$id, $name, stock_master_bool($body, 'enabled') ? 'true' : 'false']
        );
        stock_master_respond(['updated' => true]);
    }

    stock_master_respond(['error' => 'Unknown action.'], 404);
} catch (InvalidArgumentException $error) {
    stock_master_respond(['error' => $error->getMessage()], 400);
} catch (Throwable $error) {
    stock_alert_log('Stock master manager request failed', $error, [
        'action' => (string)($_GET['action'] ?? ''),
    ]);
    error_log('Stock master manager error: ' . $error->getMessage());
    stock_master_respond(['error' => 'Stock master operation failed. Check the Stock Analyzer PHP log.'], 500);
} finally {
    if ($connection !== null) {
        pg_close($connection);
    }
}
