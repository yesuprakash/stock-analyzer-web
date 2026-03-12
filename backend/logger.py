import os
import logging
from datetime import datetime, timedelta

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_FILE = os.path.join(BASE_DIR, "near_entry.log")

def clean_old_log_entries(log_file, days=2):
    if not os.path.exists(log_file):
        return

    cutoff = datetime.now() - timedelta(days=days)
    new_lines = []

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                # timestamp format used in formatter
                timestamp_str = line.split(" | ")[0]
                log_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

                if log_time >= cutoff:
                    new_lines.append(line)

            except Exception:
                # keep line if format unexpected
                new_lines.append(line)

    with open(log_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


# 🔹 Clean logs older than 2 days before logger starts
clean_old_log_entries(LOG_FILE, 2)

# Create logger
logger = logging.getLogger("near_entry_logger")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if reloaded
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)