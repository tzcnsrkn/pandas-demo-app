import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path

script_dir = Path(__file__).parent
input_path = script_dir / 'datasets' / 'netflix_titles.csv'
df = pd.read_csv(input_path)

# %% How many movies vs. TV shows were added to Netflix each year?
# Make sure the date_added is datetime, not to cause unexpected parsing errors.
# Best practice: trimming the empty characters while parsing.
df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')

# Store converted date_added as year and then use it as attribute.
df['year_added'] = df['date_added'].dt.year
df.dropna(subset=['year_added'], inplace=True)

# We need to be explicit about type of year_added, otherwise mapped to float by default due to empty values.
df['year_added'] = df['year_added'].astype(int)
g = (df
     .groupby(['year_added', 'type'])
     .size())
print(g)

# %% What is the most common content rating on Netflix?
g = (df.groupby(['rating'])
        .size())
# print(g)

# There are some dirty ratings such as 'x min' as well as blanks.
# Better to filter known strings only.
valid_ratings = ['G', 'PG', 'PG-13', 'R', 'NC-17', 'NR', 'UR',
                 'TV-Y', 'TV-Y7', 'TV-Y7-FV', 'TV-G', 'TV-PG',
                 'TV-14', 'TV-MA']
g = (df[df['rating'].isin(valid_ratings)]
     .groupby(['rating'])
     .size()
     .sort_values(ascending=False))
print(g)

# If we only need the most common:
vc = df['rating'].value_counts()
max_count = vc.iloc[0]
print(f'Most common rating: {vc.index[0]} with the count: {max_count}')
pass