<?php
declare(strict_types=1);

function stock_analyzer_project_root(): string
{
    $siteParent = dirname(__DIR__);
    foreach ([
        $siteParent,
        $siteParent . DIRECTORY_SEPARATOR . 'stock_analyzer',
    ] as $candidate) {
        if (is_file($candidate . DIRECTORY_SEPARATOR . 'stock_analyzer.py')) {
            return $candidate;
        }
    }

    throw new RuntimeException('Stock Analyzer project directory not found.');
}

function stock_analyzer_backend_file(string $file): string
{
    if ($file === '' || basename($file) !== $file) {
        throw new InvalidArgumentException('Invalid backend file name.');
    }

    $baseDirectories = [
        dirname(__DIR__) . DIRECTORY_SEPARATOR . 'backend',
        dirname(__DIR__) . DIRECTORY_SEPARATOR . 'stock_analyzer' . DIRECTORY_SEPARATOR . 'backend',
    ];
    foreach ($baseDirectories as $baseDirectory) {
        $path = $baseDirectory . DIRECTORY_SEPARATOR . $file;
        if (is_file($path)) {
            return $path;
        }
    }

    throw new RuntimeException('Stock Analyzer backend file not found: ' . $file);
}
