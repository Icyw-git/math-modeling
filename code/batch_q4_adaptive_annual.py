"""Run and independently verify the four fixed-parameter annual policies."""

import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "q4_adaptive_annual"
LOGS = ROOT / "tmp" / "q4_adaptive_annual_logs"
NAMES = ("q2_neutral", "q2_cvar", "q3_neutral", "q3_cvar")
START_DAY = 31
END_DAY = 365
INITIAL_SOC = 1200.0


def run_one(name):
    env = os.environ.copy()
    env.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONIOENCODING="utf-8",
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / f"{name}.log").open("a", encoding="utf8") as stream:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "code" / "run_q4_adaptive_period.py"),
                "--variant",
                name,
                "--output",
                str(OUTPUT),
                "--start-day",
                str(START_DAY),
                "--end-day",
                str(END_DAY),
                "--initial-soc",
                str(INITIAL_SOC),
            ],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "code" / "verify_q4_period.py"),
                str(OUTPUT / name),
                "--prefix",
                str(ROOT / "results" / "q4_cvar" / name),
            ],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return name


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_one, name) for name in NAMES]
        for future in concurrent.futures.as_completed(futures):
            print("COMPLETE", future.result(), flush=True)
