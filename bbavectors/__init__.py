import os

__version__ = "0.0.1"

ROOT = os.path.abspath(os.path.join(__file__, os.pardir))
WORK_DIR = os.path.join(ROOT, 'work_dir')
TEMP_DIR = os.path.join(WORK_DIR, 'temp')
