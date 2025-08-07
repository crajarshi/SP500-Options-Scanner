#!/usr/bin/env python
"""Test options with a small watchlist"""
import subprocess
import sys

# Run the scanner with options on a small watchlist
cmd = [
    sys.executable,
    "sp500_options_scanner.py",
    "--watchlist", "tech.txt",
    "--options",
    "--top", "3",
    "--no-regime",  # Skip market check for speed
    "--mode", "bullish"  # Force bullish mode to get some BUY signals
]

print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=False, text=True)
sys.exit(result.returncode)