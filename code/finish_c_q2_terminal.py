"""Run annual experiment, independent verification, and data report in order."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    for command in [
        ['run_c_q2_terminal.py','--output','results/q2_terminal_value'],
        ['verify_c_q2_terminal.py','results/q2_terminal_value'],
        ['report_c_q2_terminal.py'],
    ]:
        subprocess.run([sys.executable,str(ROOT/'code'/command[0]),*command[1:]],cwd=ROOT,check=True)
