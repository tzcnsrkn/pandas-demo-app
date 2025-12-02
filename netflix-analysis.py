from os.path import expanduser

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path

script_dir = Path(__file__).parent
input_path = script_dir / 'datasets' / 'netflix_titles.csv'
df = pd.read_csv(input_path)

