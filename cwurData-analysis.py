import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
from pathlib import Path

script_dir = Path(__file__).parent
input_path = script_dir / 'datasets' / 'dataset-university-rankings' / 'cwurData.csv'

# onlydirs = [entry.name for entry in input_path.iterdir() if entry.is_dir()]
# allContent = [entry.name for entry in input_path.iterdir()]
# print(allContent)

df = pd.read_csv(input_path)
# print(df.head())
g = (df[(df['year'] == 2015) & (df['world_rank'] <= 100)]
     .groupby(['country'])
     .size()
     .sort_values(ascending=False)
     .to_frame(name='count'))
print(g)

pass