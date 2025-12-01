import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path

script_dir = Path(__file__).parent
input_path = script_dir / 'datasets' / 'netflix_titles.csv'
df = pd.read_csv(input_path)

#region How many movies vs. TV shows were added to Netflix each year?
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
#endregion

#region What is the most common content rating on Netflix?
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
#endregion

#region What percentage of the dataset is Movies vs. TV Shows?
type_counts = (df['type']).value_counts()
movies_count = type_counts['Movie']
tvshows_count = type_counts['TV Show']
total = type_counts.sum()

# print(f'Movies count: {movies_count}')
# print(f'TV Show count: {tvshows_count}')
print(f'Movies are {movies_count*100 / total:.2f} of total dataset while TV shows are {tvshows_count*100 / total:.2f}')
#endregion

#region How many movies starring 'Tom Cruise' are in the dataset?
tc_starred_count = df['cast'].str.contains('tom cruise', case=False, na=False).sum()
print(f'Tom Cruise starred in {tc_starred_count} of the dataset records.')
#endregion

#region Which directors have the most content? (Top 10)
# By default, .value_counts() excludes NaN values from the result.
# If blanks are needed, provide dropna=False.
vc_director = df['director'].value_counts()
print(vc_director.head(10)) # or vc_director.iloc[:10])
#endregion

#region Clean the duration column: Separate it into two columns, duration_value (int) and unit (string).
df['unit'] = df['duration'].str.extract(r'(min|Seasons?)$').fillna('-')
df['duration_value'] = df['duration'].str.extract(r'^(\d+)').fillna('-')
print(df[['duration_value', 'unit']])
#endregion

#region Find all content where the description contains the words 'kill' or 'violence'.
df['plus13_content'] = df['description'].str.contains('\bkill|\bviolence', case=False)
print(df[['show_id', 'plus13_content']].to_string())

# Alternative if memory is a concern - as above creates an additional column.
mask = df['description'].str.contains('\bkill|\bviolence', case=False)
print(df.loc[mask, 'show_id'].to_string())
#endregion

#region "Which country has the highest production output of Movies?"
mask = df['type'] == 'Movie'
mov_out = df.loc[mask, 'country'].value_counts(dropna=True)
print(mov_out)
#endregion

#region "What is the average number of seasons for TV Shows?"
mask_avg_seasons = df['type'] == 'TV Show'
avg_seasons = (df.loc[mask_avg_seasons, 'duration'].str.extract(r'^(\d+)', expand=False)
               .astype(float)
               .mean())
print(f'Average number of seasons for TV Shows: {avg_seasons:.2f}')
#endregion
