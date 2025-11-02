"""OpenVLA evaluation wrapper for RoboVerse tasks"""

import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from vla_eval import main

if __name__ == "__main__":
    main()
