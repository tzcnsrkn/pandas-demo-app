from os.path import expanduser

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

#region Which country has the highest production output of Movies?
mask = df['type'] == 'Movie'
mov_out = df.loc[mask, 'country'].value_counts(dropna=True)
print(mov_out)
#endregion

#region What is the average number of seasons for TV Shows?
mask_avg_seasons = df['type'] == 'TV Show'
avg_seasons = (df.loc[mask_avg_seasons, 'duration'].str.extract(r'^(\d+)', expand=False)
               .astype(float)
               .mean())
print(f'Average number of seasons for TV Shows: {avg_seasons:.2f}')
#endregion

#region Which year had the highest number of content releases?
print(f'Most content was released in year {df['release_year'].mode()[0]}.')
#endregion

#region Compare the number of Movies vs. TV Shows added to Netflix per year.
df['years_released'] = df['date_added'].str.extract(r'(\d{4})$', expand=False)
yearly_count_by_type = (df
                        .groupby(['years_released','type'])
                        .size()
                        .unstack(fill_value=0)
                        .sort_index())

print(yearly_count_by_type)
#endregion

#region In which month are the most Movies added to Netflix?
monthly_vc = df['date_added'].str.extract(r'^([a-zA-Z]+)', expand=False).value_counts()
print(monthly_vc.index[0])
#endregion

#region What is the average delay between the release_year and the date_added to Netflix?
# 1. Dirty practice
df['added_year'] = df['date_added'].str.extract(r'(\d{4})$', expand=False).fillna(-1)
cols_to_convert = ['added_year', 'release_year']
df[cols_to_convert] = df[cols_to_convert].astype(int)

df['delay_duration'] = df['added_year'] - df['release_year']
is_valid_year = df['delay_duration'] >= 0
avg_delay_in_years = df.loc[is_valid_year, 'delay_duration']
print(avg_delay_in_years.mean())

# 2. Best practice
df['date_added_dt'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce') # This is NaN safe, NaN mapped to NaT
df['avg_delay_release'] = (df['date_added_dt'] - pd.to_datetime(df['release_year'])).dt.days
print(f'Average delay in days: {df['avg_delay_release'].mean()}')
#endregion

#region Find all movies that are 'Documentaries' AND were produced in the 'United Kingdom'.
docu_uk_mask = ((df['type'] == 'Movie') &
                (df['listed_in'].str.contains('Documentaries', na=False)) &
                (df['country'].str.contains('United Kingdom', na=False)))
print(f'Movies that are \'Documentaries\' AND were produced in the UK: '
      f'{df.loc[docu_uk_mask, ['show_id', 'title']]}')
#endregion

#region Identify the longest Movie and the TV Show with the most Seasons.
# 1. Movie
mask_mov = df['type'] == 'Movie'
movies_dur = (df.loc[mask_mov, 'duration'].str.extract(r'^(\d+)', expand=False)
                    .dropna()
                    .astype(int)
                    .sort_values(ascending=False))
print(f'The longest movie: {df.loc[movies_dur.index[0]]}')

# 2. TV Show
mask_tv_series = df['type'] == 'TV Show'
tv_series_dur = (df.loc[mask_tv_series, 'duration'].str.extract(r'^(\d+)', expand=False)
                    .dropna()
                    .astype(int)
                    .sort_values(ascending=False))
print(f'The longest running tv series: {df.loc[tv_series_dur.index[0]]}')
#endregion

# ===========================================
#               VISUALIZATION
# ===========================================
import matplotlib.pyplot as plt
import seaborn as sns

#region Plot the growth of Netflix's content library over time (by date_added)
# 1. via Matplotlib
df['year_month_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
(df.groupby(pd.Grouper(key='year_month_added', freq='ME'))
 .size()
 .plot())

plt.title('Average content count')
plt.xlabel = 'Date added'
plt.ylabel = 'Count'
plt.show()

# 2. via Seaborn
df['year_month_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
sns_data = df.groupby(pd.Grouper(key='year_month_added', freq='ME')).size().reset_index(name='count')
ax = sns.barplot(data=sns_data, x='year_month_added', y='count')
# Nice-to-have: Fixing the baseline
# Show every 10th label
for i, label in enumerate(ax.get_xticklabels()):
    if i % 10 != 0:
        label.set_visible(False)
plt.xticks(rotation=45, ha='right')
plt.show()
#endregion

#region Show the distribution of Movie durations (in minutes)
mov_mask = df['type'] == 'Movie'
movies_dur = df.loc[mov_mask, 'duration'].str.extract(r'^(\d+)', expand=False).fillna(0).astype(int)
movies_dur.hist(bins=30, legend=True)

plt.xlabel('Duration (minutes)')
plt.ylabel('Count')
plt.show()
#endregion

#region Create a Heatmap showing the density of content releases by Month and Country (for the top 10 countries)
df['country_split'] = df['country'].str.split(',') # country: 'United States, India, France' -> country_split: ['United States','India','France']
df_exploded = df.explode('country_split') # creating a separate row for each item in the 'country'
df_exploded['country_split'] = df_exploded['country_split'].str.strip()

top10_countries = df_exploded['country_split'].value_counts().head(10).keys()
top10countries_mask = df_exploded['country_split'].isin(top10_countries)
df_exploded['month_added'] = df_exploded['date_added'].str.extract(r'^([a-zA-Z]+)', expand=False).fillna('N/A')

country_month = (df_exploded.loc[top10countries_mask]
                 .groupby(['country_split', 'month_added'])
                 .size()
                 .unstack(fill_value=0))
# print(country_month.head(10))

# Beautify the output months - chronologically sorted
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
country_month = country_month.reindex(columns=month_order, fill_value=0)

plt.figure(figsize=(16, 12))
sns.heatmap(country_month.astype(int), fmt='g', annot=True, cmap='YlGnBu')
plt.show()
#endregion