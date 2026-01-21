<?php
// SECURITY: simple token
$token = $_GET['token'] ?? '';
if ($token !== '9f7c1d2a_StockAnalyzer_Run_2026!') {
    http_response_code(403);
    exit('Forbidden');
}


$cmd = 'cd /d C:\xampp\htdocs\Stock_Analyzer && '
     . 'venv\Scripts\activate && '
     . 'python stock_analyzer.py > logs\web_run.log 2>&1';

pclose(popen("start /B cmd /C \"$cmd\"", "r"));

echo "Stock analyzer started in background";

// Command to run analyzer in background
// $cmd = 'cd /home/bhavat5/stock_analyzer && '
//      . 'source venv/bin/activate && '
//      . 'nohup python stock_analyzer.py > logs/web_run.log 2>&1 &';

//exec($cmd);

//echo "Stock analyzer started in background";
