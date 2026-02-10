<?php
header('Content-Type: application/json');
set_time_limit(0);

// Python executable
$python = '"C:\\xampp\\htdocs\\Stock_Analyzer\\venv\\Scripts\\python.exe"';
// Python script
$script = '"C:\\xampp\\htdocs\\Stock_Analyzer\\backend\\near_entry_api.py"';

// (no params yet – we’ll add later)
$cmd = "$python $script 2>&1";

// Run Python
$output = shell_exec($cmd);
$output = trim($output);

// Always return valid JSON
if ($output === '' || $output === null) {
    echo json_encode([]);
    exit;
}

// If Python accidentally printed text, fail safely
if ($output[0] !== '[' && $output[0] !== '{') {
    echo json_encode([
        "error" => "Invalid Python output",
        "raw" => $output
    ]);
    exit;
}

// Normal case
echo $output;
exit;
