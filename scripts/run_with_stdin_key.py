"""One-shot launcher: keep credentials out of argv and disk."""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['T2Q_API_KEY']=sys.stdin.readline().strip()
from text2query.cli import main
main()
