import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path

script_dir = Path(__file__).parent
input_path = script_dir / 'datasets' / 'netflix_titles.csv'
df = pd.read_csv(input_path)

# Make sure the date_added is datetime, not to cause unexpected parsing errors.
# Best practice: trimming the empty characters while parsing.
df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
# Store converted date_added as year and then use it as attribute.
df['year_added'] = df['date_added'].dt.year
# We need to be explicit about type of year_added, otherwise mapped to float by default due to empty values.
df.dropna(subset=['year_added'], inplace=True)
df['year_added'] = df['year_added'].astype(int)

g = (df
     .groupby(['year_added', 'type'])
     .size())
print(g)


pass