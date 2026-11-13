"""Prepare evaluation; execute its frozen bundle using the generated sbatch."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent/'evaluation'))
from evaluation import main
if __name__ == '__main__':
    raise SystemExit(main())
