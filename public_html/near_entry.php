<?php
header('Content-Type: application/json');
set_time_limit(0);

$python = "C:\\xampp\\htdocs\\Stock_Analyzer\\venv\\Scripts\\python.exe";
$script = "C:\\xampp\\htdocs\\Stock_Analyzer\\near_entry_api.py";

$params = [];

foreach ($_GET as $key => $value) {
    $params[] = $key . "=" . escapeshellarg($value);
}

$cmd = "\"$python\" \"$script\" " . implode(" ", $params);

$output = shell_exec($cmd);

echo $output;
