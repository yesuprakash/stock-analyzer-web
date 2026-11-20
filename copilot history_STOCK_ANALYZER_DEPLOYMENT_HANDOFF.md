# Stock Analyzer Deployment Handoff

Updated: 2026-11-20

## Project and server locations

- Local project: `C:\xampp\htdocs\Stock_Analyzer`
- Collected server files: `D:\stock project\server files`
- Server Python project: `/home/bhavat5/stock_analyzer`
- Server website document root: `/home/bhavat5/public_html`
- Domain: `bhavatusabamangalam.com`
- Website PHP: 7.4 (`ea-php74`)
- Server CLI PHP: 8.3.36
- Server PostgreSQL: 9.6.23
- Server Python virtual environment: `/home/bhavat5/stock_analyzer/venv`
- The server venv has the project's dependencies, including pandas, numpy, yfinance, psycopg2-binary, python-dotenv, requests, and ta. No `requirements.txt` was present.

The site and backend are separate directories, not a single project tree. Deployment paths must preserve this layout.

## Work implemented locally

- Stock Performance page with cached Yahoo Finance daily history, session performance and graph views.
- Email Alerts manager, PostgreSQL event/settings/recipient/delivery/run tables, and a CLI daily runner.
- Stock List manager backed by a PostgreSQL `stock_master` table.
- Active stocks drive the Stock Performance screen, email-alert runner, stock analyzer, Near Entry results, and Market Correction results.
- Existing Near Entry and Market Correction prediction calculations continue to use the `predictions` table and filter their symbols using active `stock_master` records.
- Navigation links to Performance, Email Alerts, and Stock List have been added to the three existing screens.

## Database deployment

Database script:

`D:\stock project\server files\stock_analyzer_new_tables.sql`

The script creates `stock_master` and five `stock_alert_*` tables, inserts default alert thresholds, and seeds the original 130 tickers. It is idempotent and does not alter prediction tables.

The user ran this script on the server and confirmed the expected row counts: 130 stock-master rows, one settings row, and zero initial rows in recipient/event/delivery/run tables. The server app database user has `CREATE` permission on schema `public`.

The server's existing `predictions` and `prediction_summary` table schemas were inspected. No existing stock-master table was present before this deployment.

## Backups

The user confirmed creating:

- `/home/bhavat5/stock-analyzer-backup.tar.gz` — analyzer project including `.env`, excluding `venv`.
- `/home/bhavat5/stock-web-backup/` — copies of `stockcaller.html`, `near_entry.html`, `market_correction.html`, and `market_correction.php`.

The website backup was checked and all four listed files were present.

## Deployment package and current state

Deployment staging tree:

`D:\stock project\server files\deployment`

Latest package:

`D:\stock project\server files\stock_analyzer_server_deploy_php74.zip`

The first package was extracted on the server; afterward, `stock_master.html` produced a browser error: `Failed to execute 'json' on 'Response': Unexpected end of JSON input`.

Investigation found PHP 8-only `catch (Throwable)` syntax in shared files. It was changed to PHP 7-compatible syntax and the corrected archive was rebuilt and confirmed deployed by the user. On 2026-11-11, custom private PHP logging was added, the deployment archive was rebuilt, and the user provided a log entry from the new logger, confirming the logging-enabled code is deployed.

The archive contains 19 files under:

- `public_html/` — new management/performance pages and endpoints, a backend path bootstrap, and updated copies of the three existing HTML screens.
- `stock_analyzer/` — updated analyzer and backend modules.

The archive intentionally does not contain `.env`, `venv`, or cache files. The staged Stock Caller HTML was copied from the server-provided version to preserve its existing launcher behavior.
PHP errors are also recorded as JSON lines in the private project log `stock_analyzer/logs/stock_analyzer_php.log`, outside `public_html`. The PHP web user must be able to write to the project `logs` directory; the logger creates it if needed.

## 2026-11-11 performance endpoint database permissions

The user reported that both Performance buttons failed with `Could not load stock list (HTTP 500)`. The custom log captured:

- Event: `Price performance request failed`, action `stocks`.
- Exception: `Alert database operation failed: ERROR: permission denied for relation stock_master`.
- Trace: `stock_master_ensure_schema()` attempted its idempotent seed `INSERT` into `stock_master`; at that time the application DB role lacked required table privileges.

The custom log is at `/home/bhavat5/stock_analyzer/logs/stock_analyzer_php.log`. Apache log entries for missing `wp-login.php` and `cgi-bin` were unrelated bot requests, not this application error.

The app-role permissions were granted on 2026-11-12 using `grant_stock_analyzer_app_permissions.sql`. The script must target the exact server `DB_USER` from `.env`; the database SQL-editor login can be a different role. The user confirmed the Performance stock-list issue was fixed after applying the grants.

The first GRANT attempt through a wrapped SQL editor did not execute. A later raw SQL execution returned `BEGIN`, three `GRANT` results, and `COMMIT`. Do not share the database username or password in chat.

## 2026-11-12 Email Alerts deployment and status

The user configured Gmail SMTP values in the existing server `.env`; use a Google App Password, not the normal Gmail password. The actual SMTP credentials were not shared. The user reports that the Email Alerts screen now works and that their email address has been added as a recipient. Successful test-email delivery has not yet been confirmed.

The first Email Alerts page load failed with `must be owner of relation stock_alert_recipients`. The trace identified `CREATE UNIQUE INDEX IF NOT EXISTS` in `stock_alert_ensure_schema()` as request-time DDL; the app database role had DML grants but was not the table owner. The attempted ownership-change script failed because the SQL-editor role was not a member of the target app role. That ownership change is unnecessary with the corrected deployment.

The runtime index-creation statement was removed from `deployment/stock_analyzer/backend/stock_alert_lib.php`; the index remains created by `stock_analyzer_new_tables.sql`. The PHP file passed `php -l`, and the 19-file deployment archive was rebuilt:

`D:\stock project\server files\stock_analyzer_server_deploy_php74.zip`

The user deployed that archive at the server home directory, preserving the sibling `public_html/` and `stock_analyzer/` layout and the existing `.env`. The user confirmed the Email Alerts screen works and added their email. Do not run `fix_stock_alert_recipients_owner.sql`; the runtime DDL fix avoids the ownership requirement.

## Immediate next steps

1. On `email_alerts.html`, use the separate **Send test email** form and confirm receipt (check spam too). Adding a recipient does not itself send a test message.
2. After SMTP test delivery succeeds, use **Run daily alert check now** to verify the runner. Its first run each trading day refreshes prices for all 130 active stocks and can send alert digests.
3. Configure a daily cron job for `stock_alert_runner.php` after market close only after the test email and manual runner both succeed. No server schedule has been confirmed.
4. Keep the existing server `.env`; do not overwrite it. Verify a unique, strong `STOCK_ALERT_ADMIN_PASSWORD` is set before exposing management screens. Do not use the simple local development password.
5. Check required extensions for the actual website PHP 7.4 runtime, especially `pgsql` and `curl`. The server CLI showed PHP modules, but the cPanel website runtime still needs confirmation. CLI and web PHP configurations can differ.
6. Confirm the endpoints/pages as needed:
   - `https://www.bhavatusabamangalam.com/stock_master.php?action=state` (unauthenticated should return JSON with HTTP 401).
   - `https://www.bhavatusabamangalam.com/stock_master.html`
   - `https://www.bhavatusabamangalam.com/price_performance.php?action=stocks`
   - `https://www.bhavatusabamangalam.com/email_alerts.html`
   - `https://www.bhavatusabamangalam.com/stock_alerts.php?action=state` (unauthenticated should return JSON with HTTP 401).
7. If an endpoint returns an empty/non-JSON response or an error, check the custom PHP log and share only the HTTP status, response body, and sanitized log entry. Never share cookies, passwords, API keys, App Passwords, or tokens.
8. Replace/rotate the existing Stock Caller launcher token. The server-provided page has a hard-coded token shared in prior conversation; do not repeat it or include it in documentation or chat.

## Important compatibility and deployment notes

- PHP web version is 7.4; preserve PHP 7.4 syntax in all deployed PHP files.
- The new API session cookie path is derived from the request's script directory, so it works from the server root as well as local XAMPP.
- PHP endpoints use `stock_analyzer_bootstrap.php` to locate the backend in either the local project layout or the server's sibling-folder layout.
- Alert cache is intended to be private, outside `public_html`; verify the server user can create/write the cache directory on the first runner execution.
- Yahoo Finance chart data is unofficial and subject to availability/rate limits.
- Gmail SMTP settings were reportedly added to the server `.env`; test-email receipt remains unconfirmed. No automatic cron schedule has been installed.
- Do not upload `.env` from the local machine or the local `venv`.

## 2026-11-13 email schedule and resend controls

The local deployment staging tree now includes:

- An admin-only **Resend today's alerts now** action that refreshes prices and resends today's matching movement digest even when those events were already delivered. Failed forced retries do not erase a prior successful delivery record.

The user clarified that execution time should be configured in cPanel cron, not in the application. For noon Monday–Friday, use cron expression `0 12 * * 1-5` with `php /home/bhavat5/public_html/stock_alert_runner.php --force` (use the server's PHP CLI path). Cron uses the server/cPanel timezone, so verify it is set to Asia/Kolkata if noon IST is intended. `--force` refreshes prices; it does not bypass prior delivery checks. The manager's resend button is the separate action that bypasses those checks.

The deployment archive should contain the resend-button changes only; no database migration is needed for scheduling. Server deployment and cron configuration are not yet confirmed. The scheduled job sends email only when enabled rules detect qualifying movements; it does not send an empty daily email.

## 2026-11-18 analyst research and Performance page changes

The user requested recent brokerage/platform research for each stock, shown in a per-stock popup, alongside NewsAPI headlines. The implementation was added locally as:

- `backend/stock_research_lib.php` — stores and queries structured ratings and related NewsAPI articles, retaining the latest five trading dates.
- `public_html/stock_research.php` — popup data endpoint.
- `public_html/stock_research_runner.php` — CLI refresh runner.
- `stock_research_tables.sql` — research tables.

The runner reads the existing `NEWSAPI_KEY` from the local/server `.env` for news articles. NewsAPI does **not** provide structured Buy/Sell/Hold analyst ratings; headlines are displayed separately and are not converted into ratings. Structured ratings require an authorized provider configured with `STOCK_RESEARCH_FEED_URL` and, if required, `STOCK_RESEARCH_FEED_TOKEN`. No ratings provider is currently configured, so an empty ratings list is expected. The user reported that the popup later loaded NewsAPI articles; structured ratings remained empty.

The user ran the research schema and initially encountered a PostgreSQL permission error on `stock_research_recommendations`. Grants for the research tables and sequences were provided; the user subsequently reported that the popup loaded. A successful read with zero rows does not necessarily produce a PHP error-log entry. To populate news rows, run the CLI runner from cPanel Terminal/SSH using the server PHP CLI:

`<server-php-cli> /home/bhavat5/public_html/stock_research_runner.php`

The local research files are untracked and are not included in the current Git-only deployment ZIP. Do not assume the deployed server has this research feature until a package containing those files is intentionally prepared and uploaded. Never copy `.env` values into a package or chat.

The user also requested a stock-name filter, sortable performance-session columns, and a session selector inside the expanded graph popup. There was confusion when a deployment-only copy was replaced by the smaller Git version. The old staged HTML was overwritten; a prior size difference (27,178 bytes versus the smaller Git page) does not prove which specific controls that file contained. No independent backup containing the old stock filter was found. The requested controls were then added to the Git page:

- Live stock filter searches the symbol and available company-name fields.
- Clicking a session column sorts ascending/descending; the active column is highlighted and shows a direction arrow. The existing **Rank by** selector remains.
- The expanded graph popup has a **Trend sessions** selector synchronized with the page-level trend selector.

These changes were committed and pushed on 2026-11-18:

`9a5efc0 Restore performance table controls`

The current deployment ZIP, `D:\stock project\server files\stock_analyzer_server_deploy_php74.zip`, was rebuilt from the Git deployment set plus the updated Performance page. It contains 31 files and excludes `.env`, virtualenvs, logs, caches, and debug/test files. Its top-level layout preserves sibling `public_html/` and `stock_analyzer/` directories. The user has not yet confirmed uploading this latest ZIP.

Important current-state distinctions:

- `origin/main` is at `9a5efc0`; the filter, sortable columns, and expanded-graph session selector are committed.
- The Git-only ZIP does not contain the untracked analyst-research files. The prior research UI/backend work was intentionally not included when the user selected a Git-only deployment.
- The hover tooltip previously added to a non-Git page copy was dropped when the page was restored from Git and has not been re-added to the current page.
- `public_html/stockcaller.html` has an unrelated local modification, and the four research files listed above remain untracked. They were not included in commit `9a5efc0`.
- The existing `D:\stock project\server files\deployment` staging tree is not authoritative for the latest ZIP; use the ZIP itself for deployment.

## 2026-11-18 NSE index symbol comparison

Added local script `C:\xampp\htdocs\Stock_Analyzer\compare_nse_index_symbols.py`. It reads the 16 supplied NSE live-equity index pages, then calls the page's public `getIndicesData` API for the constituent symbols. NSE symbols are normalized to the application's `SYMBOL.NS` format and deduplicated across indices.

The script reads all rows from `stock_master`, including disabled entries, and reports per-index matches, NSE symbols missing from the database, and database symbols outside the supplied indices. It is report-only and does not insert, update, or delete database records. An optional `--csv <path>` writes a union report with membership and enabled status.

Verified against the live NSE responses and the database configured by the local `.env` on 2026-11-18: 16 indices, 463 total index memberships, 287 unique NSE symbols, 130 database symbols, 175 NSE symbols missing from the database, and 18 database symbols outside these indices. These are a point-in-time comparison; constituents and database rows can change.

Run locally from the project root with:

`.\venv\Scripts\python.exe .\compare_nse_index_symbols.py --csv "D:\stock project\server files\nse_stock_master_comparison.csv"`

The script is currently untracked, not committed, and not included in the 31-file deployment ZIP. It does not modify the database. To compare against the server database, place the script in the server project root and run it with the server Python environment and server `.env`; do not copy the local `.env`.

At the user's request, a raw PostgreSQL snapshot script was also generated at `D:\stock project\server files\nse_stock_master_compare.sql`. It includes 287 unique `.NS` symbols from the 16 supplied indices, selects matching `stock_master` rows, reports the missing count, and lists missing symbols. It uses a temporary table inside a transaction and does not change permanent database tables. The user intends to run this SQL manually.

The user next requested copying a stock's complete Performance row with each session's percent return and price. A per-stock **Copy** button was added next to **Expand**. It copies a tab-separated header and data row for Excel, including the latest adjusted close and, for each 1/2/3/4/5/10/15/30/45/60-session return, the return plus its reference adjusted close and date (N sessions before the latest close). Clipboard API and a browser fallback are used; copy failures are shown in the page. JavaScript syntax and output-format checks passed.

The deployment ZIP was refreshed on 2026-11-18 at `D:\stock project\server files\stock_analyzer_server_deploy_php74.zip` (31 files; no `.env`, virtualenv, logs, caches, or debug/test files). It contains the current Performance page with the copy button, filter, sortable columns, and expanded-graph session selector. The copy-button change remains uncommitted; the comparison Python utility and raw SQL file are not included in this deployment ZIP. Upload of this latest ZIP has not been confirmed.

## 2026-11-20 additional Performance sessions

The Stock Performance page now supports these additional trading-session periods:

- 75 sessions
- 90 sessions
- 105 sessions
- 120 sessions
- 150 sessions
- 180 sessions

The periods are included in the PHP performance calculations, Rank by and Trend sessions selectors, performance table, sortable columns, expanded graph selector, sparkline/chart display, and Copy row output. Yahoo Finance cache history for each stock retains 181 daily records so the 180-session comparison has one prior reference point for the latest adjusted close.

No database or SQL migration is required. The new periods are calculated in PHP from the existing private daily Yahoo Finance cache and are not stored in PostgreSQL.

The PHP endpoint passed `php -l`. The deployment ZIP was rebuilt on 2026-11-20 at:

`D:\stock project\server files\stock_analyzer_server_deploy_php74.zip`

The rebuilt ZIP contains 31 files and excludes `.env`, virtualenvs, logs, caches, debug/test files, and analyst-research work. It includes the updated `public_html/price_performance.html` and `public_html/price_performance.php`.
