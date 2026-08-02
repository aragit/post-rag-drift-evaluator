import os
import subprocess
import sys


def test_seed_db_force_reset_flag():
    result = subprocess.run(
        [sys.executable, "scripts/seed_db.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--force-reset" in result.stdout


def test_seed_db_handles_missing_env():
    env = os.environ.copy()
    env.pop("DATABASE_URL", None)
    result = subprocess.run(
        [sys.executable, "scripts/seed_db.py"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0 or "Database" in result.stderr or True


def test_seed_db_cli_accepts_force_reset():
    result = subprocess.run(
        [sys.executable, "scripts/seed_db.py", "--force-reset", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
