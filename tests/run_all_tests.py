"""
LabInsight AI - Master Test Suite Runner
Discovers and executes all unit and integration tests across the project.
"""

import sys
import unittest
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


def run_master_test_suite():
    print("=" * 70)
    print("[*] LABINSIGHT AI - AUTOMATED TEST SUITE")
    print("=" * 70)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python:    {sys.version.split()[0]}")
    print(f"Directory: {BASE_DIR}")
    print("-" * 70)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(BASE_DIR / "tests"), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_time

    print("-" * 70)
    print(f"Tests Run:      {result.testsRun}")
    print(f"Errors:         {len(result.errors)}")
    print(f"Failures:       {len(result.failures)}")
    print(f"Skipped:        {len(result.skipped)}")
    print(f"Execution Time: {elapsed:.2f} seconds")

    if result.wasSuccessful():
        print("RESULT:         [PASS] ALL TESTS PASSED SUCCESSFULLY")
        print("=" * 70)
        return 0
    else:
        print("RESULT:         [FAIL] TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = run_master_test_suite()
    sys.exit(exit_code)
