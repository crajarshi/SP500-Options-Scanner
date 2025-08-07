#!/usr/bin/env python
"""Test options display with demo data that has BUY signals"""
import subprocess
import sys

# Run the scanner with demo data that should have some BUY signals
cmd = [
    sys.executable,
    "sp500_options_scanner.py",
    "--demo",
    "--options",
    "--top", "3",
    "--filter", "BUY,STRONG_BUY"  # Only show BUY signals
]

print("Running:", " ".join(cmd))
print("This should show options contracts for stocks with BUY signals...")
print("-" * 60)
result = subprocess.run(cmd, capture_output=False, text=True)
sys.exit(result.returncode)