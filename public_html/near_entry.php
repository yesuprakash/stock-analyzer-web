<?php
header("Content-Type: application/json");

// 🔥 FULL PATH to python.exe (very important)
$python = "C:\\xampp\\htdocs\\Stock_Analyzer\\venv\\Scripts\\python.exe";

// 🔥 FULL PATH to api file
$script = "C:\\xampp\\htdocs\\Stock_Analyzer\\near_entry_api.py";

// Build command
$cmd = "\"$python\" \"$script\"";

// Execute
$output = shell_exec($cmd);

// If execution failed
if ($output === null) {
    echo json_encode(["error" => "Python execution failed"]);
    exit;
}

// Print JSON from Python
echo $output;
