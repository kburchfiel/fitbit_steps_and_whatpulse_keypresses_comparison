# %% [markdown]
# # WhatPulse Keypress Stats Analyzer
# 
# By Kenneth Burchfiel
# 
# Released under the MIT license
# 
# *(I am not affiliated with WhatPulse (https://whatpulse.org) but highly recommend checking out the program, which I've used since September 2008. You can find my online WhatPulse page here: https://whatpulse.org/KBurchfiel)*
# 
# This script allows you to perform various analyses of your WhatPulse typing stats. It does so by accessing the local WhatPulse database on your computer; reading this data into Pandas DataFrames, and then summarizing and graphing that data. The output of the script includes:
# 
# 1. Rolling average calculations at the 7-, 28-, and 365-day level
# 2. Percentile and rank calculations (so that you can see how today's keypresses compare to past days)
# 3. Weekly and hourly keypress stats
# 4. Static (.png) and interactive (.html) keypress charts
# 
# By converting this notebook into a Python file and then instructing your computer to run it on an hourly basis, you can also keep track of how much you're typing during your day. Which is sort of nerdy, to be honest. But so is this whole program ;) 
# 
# [More documentation will be provided in the future]

# %% [markdown]
# I'll first import a number of packages that the program will use:

# %%
import time
start_time = time.time() # Allows the program's runtime to be measured
import pandas as pd
import sqlalchemy
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import statsmodels.api as sm
from scipy.stats import percentileofscore
import plotly.express as px
import kaleido
from IPython.display import Image

# %% [markdown]
# ## Importing Whatpulse data
# 
# In order to analyze my WhatPulse data, I'll first need to import it from my local Whatpulse SQLite database. I'll also import a copy of the Whatpulse SQLite database stored on my old laptop (so that my analysis doesn't have to be limited to my current computer's keypress data.)
# 
# You'll of course need to update the following cell with the path(s) to your own WhatPulse database(s).
# 
# Note: if you want to run this program on your own, but don't have access to a WhatPulse database, you can still run this program (assuming you've downloaded or cloned it from GitHub). Skip ahead to the line where I read in 

# %%
database_paths_list = [r'C:\Users\kburc\AppData\Local\whatpulse\whatpulse.db', r'C:\Users\kburc\D1V1\Documents\whatpulse_database_backups\a13r2_whatpulse.db'] 
# Note that the first path is to my computer's active database, and that the second path is to a copy of the database stored on my old computer. 

# %% [markdown]
# The following function analyzes each database's keypresses table at either the daily or hourly level.

# %%
def generate_keypress_totals(database_path, level = 'daily'):
    '''The level argument can be 'daily', in which case the DataFrame
    returned by the function will show daily keypress totals, or 'hourly',
    which will preserve the hourly keypress totals in the original database.'''
    file_name = database_path.split('\\')[-1]
    sqlalchemy_sqlite_engine = sqlalchemy.create_engine(
        'sqlite:///'+database_path) # Based on https://docs.sqlalchemy.org/en/13/dialects/sqlite.html#connect-strings
    sqlalchemy_connection = sqlalchemy_sqlite_engine.connect()
    df_keypresses = pd.read_sql("select * from keypresses", 
    con = sqlalchemy_sqlite_engine) # Simply reads all of the data from this 
    # table into a Pandas DataFrame
    df_keypresses = df_keypresses.query("day != '0000-00-00'").copy() # Removes
    # this blank date value from the database if it happens to be there
    if level == 'daily':
        df_keypresses = df_keypresses.pivot_table(
            index = 'day', values = 'count', aggfunc = 'sum')
        df_keypresses.sort_values('day', inplace = True)
    elif level == 'hourly': # The original data is already displayed 
        # at the hourly level, so there's no need for a pivot_table() call.
        df_keypresses.sort_values(['day', 'hour'], inplace = True)
    else:
        raise ValueError("Unrecognized level argument passed to function.")
    df_keypresses.rename(columns={'count':'keypresses'},inplace=True)  
    # The above line groups the hourly rows in the DataFrame into daily rows.
    return df_keypresses

# %%
keypress_databases_list = []

for path in database_paths_list:
    keypress_databases_list.append(generate_keypress_totals(path, level = 'daily'))

df_combined_daily_keypresses = pd.concat([keypress_databases_list[i] for i in range(len(keypress_databases_list))])
df_combined_daily_keypresses.sort_index(inplace=True)

# At this point, my copy of df_combined_daily_keypresses has multiple
# entries for a few days in which I logged keys on both computers.
# Therefore, the following line groups these entries into a single row
# for each date.
df_combined_daily_keypresses = df_combined_daily_keypresses.reset_index().pivot_table(index = 'day', values = 'keypresses', aggfunc = 'sum')
df_combined_daily_keypresses.index = pd.to_datetime(df_combined_daily_keypresses.index)

df_combined_daily_keypresses.to_csv('data/df_combined_daily_keypresses.csv')
df_combined_daily_keypresses

# %% [markdown]
# The following line updates df_combined_daily_keypresses with the copy of this DataFrame that just got exported to a .csv. The purpose of this cell is to allow you to run this script even if you don't have your own WhatPulse database.

# %%
df_combined_daily_keypresses = pd.read_csv('data/df_combined_daily_keypresses.csv', index_col='day')
# The following line allows the DataFrame's index to be compatible with certain
# date operations that the following code block will perform.
df_combined_daily_keypresses.index = pd.to_datetime(df_combined_daily_keypresses.index)
df_combined_daily_keypresses

# %% [markdown]
# The following code block fills in the DataFrame with missing dates (e.g. dates in which I did not have any keypresses). I want to add in those missing dates so that I can calculate more accurate rolling averages.

# %%
first_date = df_combined_daily_keypresses.index[0]
last_date = df_combined_daily_keypresses.index[-1]
full_date_range = pd.date_range(start=first_date, end = last_date) # https://pandas.pydata.org/docs/reference/api/pandas.date_range.html
df_combined_daily_keypresses = df_combined_daily_keypresses.reindex(full_date_range, fill_value=0) # See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reindex.html
df_combined_daily_keypresses.index.name = 'Date'
df_combined_daily_keypresses.reset_index(inplace=True) 

# %% [markdown]
# Now that I have a more complete daily keypress history, I can begin performing analyses on this data. First, I will use the rolling() function within Pandas to calculate 7-, 28-, and 365-day moving averages.

# %%
df_combined_daily_keypresses['7_day_ma'] = df_combined_daily_keypresses['keypresses'].rolling(7).mean()
df_combined_daily_keypresses['28_day_ma'] = df_combined_daily_keypresses['keypresses'].rolling(28).mean() 
# I switched from a 30-day MA to a 28-day MA because my average keypresses vary significantly by weekday, meaning
# that a 30-day average would be skewed by the number of Saturdays and Sundays present in the data.
df_combined_daily_keypresses['365_day_ma'] = df_combined_daily_keypresses['keypresses'].rolling(365).mean()
df_combined_daily_keypresses['percentile'] = 100*df_combined_daily_keypresses['keypresses'].rank(pct=True)
df_combined_daily_keypresses['rank'] = df_combined_daily_keypresses['keypresses'].rank(ascending = False)
df_combined_daily_keypresses

# %% [markdown]
# <h2 style="color:cyan"> Here with editing</h2>

# %% [markdown]
# Adding in weekdays:

# %%
df_weekday_mapping = pd.DataFrame({"Number":[0, 1, 2, 3, 4, 5, 6], "Weekday":["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]})
# Weekday numbers in Python begin with 0 for Monday and end with 6 for Sunday. See https://docs.python.org/3/library/datetime.html#datetime.date.weekday
df_weekday_mapping

# %%
df_combined_daily_keypresses['weekday_num'] = df_combined_daily_keypresses['Date'].dt.weekday.copy()
df_combined_daily_keypresses

# %%
df_combined_daily_keypresses = df_combined_daily_keypresses.merge(df_weekday_mapping, left_on = 'weekday_num', right_on = 'Number', how = 'left')
# If the 'how' field is left blank, the resulting DataFrame will be sorted by weekday instead of by date.
# Sorting by the Date column is necessary here since the merge operation sorted the data by 
df_combined_daily_keypresses.drop(['weekday_num', 'Number'], axis = 1, inplace = True)
df_combined_daily_keypresses

# %% [markdown]
# Summary daily keypress data statistics:

# %%
df_combined_daily_keypresses['keypresses'].describe()

# %%
df_combined_daily_keypresses.tail(10) # Last 10 days of data

# %% [markdown]
# ## My top 50 keypress totals:

# %%
df_max_keypresses = df_combined_daily_keypresses.sort_values('keypresses', ascending = False).reset_index(drop=True)
df_max_keypresses.insert(0, 'Rank', df_max_keypresses.index+1)
keypress_difference_list = [df_max_keypresses.iloc[i]['keypresses'] - df_max_keypresses.iloc[i+1]['keypresses'] for i in range(len(df_max_keypresses) -1 )]
# This list comprehension calculates the difference between each row and the row below it. This isn't possible for the final row,
# so we'll instead append a np.Nan to it.)
keypress_difference_list.append(np.NaN)
df_max_keypresses['difference_from_lower_rank'] = keypress_difference_list
df_max_keypresses.head(50)

# %%
plt.bar(x = df_max_keypresses.head(50)['Rank'], height= df_max_keypresses.head(50)['keypresses'])

# %%
df_combined_daily_keypresses.describe()

# %% [markdown]
# ## Keypress percentiles:

# %%
df_combined_daily_keypresses['keypresses'].describe(percentiles=np.linspace(0.9,1,11))

# %% [markdown]
# Finding the percentile corresponding to a day with only a single keypress:

# %%
percentileofscore(df_combined_daily_keypresses['keypresses'], 1)

# %% [markdown]
# ## Full percentile list (at 5% increments):

# %%
df_percentiles = df_combined_daily_keypresses['keypresses'].describe(
percentiles=np.linspace(0,1,21))[4:-1].reset_index().sort_values(
    'keypresses', ascending = False).reset_index(drop=True).rename(columns={'index':'percentile'})
# The first 3 rows and final row provide descriptive statistics that aren't 
# necessary to include within this DataFrame, so we can get rid of them
# by adding [4:-1] to the end of this line.
# Calling reset_index() twice appears inefficient, but it makes it easier
# to sort by a particular value (in this case, keypresses).
keypress_difference_list = [df_percentiles.iloc[i, 1] - df_percentiles.iloc[i+1, 1] for i in range(len(df_percentiles) -1 )]
# This list comprehension calculates the difference between each row and the row below it. This isn't possible for the final row,
# so we'll instead append a np.Nan to it.)
keypress_difference_list.append(np.NaN)
df_percentiles['difference_from_lower_percentile'] = keypress_difference_list
df_percentiles

# %%
plt.bar(df_percentiles['percentile'], df_percentiles['difference_from_lower_percentile'])

# %%


# %%
plt.hist(df_combined_daily_keypresses['keypresses'], bins = 40)
plt.xlabel('Keypresses')
plt.ylabel('Number of days')
plt.title('Daily keypress histogram')
plt.savefig('graphs\\keypress_histogram.png')

# %%


# %%
df_weekday_pivot = df_combined_daily_keypresses.pivot_table(index = 'Weekday', values = 'keypresses', aggfunc = 'mean').sort_values('keypresses', ascending = False).reset_index()
df_weekday_pivot

# %%
# The following line makes the Weekday column categorical so that it can be 
# sorted in a custom order (which I specify in the 'categories' parameter). 
# See https://pandas.pydata.org/docs/user_guide/categorical.html
df_weekday_pivot['Weekday'] = pd.Categorical(df_weekday_pivot['Weekday'], 
categories = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 
'Thursday', 'Friday', 'Saturday'], ordered = True)

# %%
df_weekday_pivot.sort_values('Weekday', inplace = True)
df_weekday_pivot

# %% [markdown]
# Graphing my average keypresses per weekday:

# %%
df_weekday_pivot

# %% [markdown]
# The following cell creates a series of colors that will be used as the bar labels for each weekday. Weekdays with more keypresses will be bluer, and weekdays with fewer keypresses will be grayer.

# %%
max_weekday_keypresses = max(df_weekday_pivot['keypresses'])
min_weekday_keypresses = min(df_weekday_pivot['keypresses'])
# The following line designates the colors as (r, g, b) tuples. r and g are always set to 0.5. b will be 0.5 on the day with the fewest keypresses and 1 on the day with the most keypresses.
bar_color_list = [(0.5, 0.5, 0.5 + 0.5*(keypress_value - min_weekday_keypresses)/(max_weekday_keypresses - min_weekday_keypresses)) for keypress_value in df_weekday_pivot['keypresses']]
bar_color_list

# %% [markdown]
# ## Creating a chart of weekly average keypresses:

# %%
fig, ax = plt.subplots(figsize = [10, 5])
fig.set_facecolor('white')
bc = ax.bar(x = df_weekday_pivot['Weekday'], height = df_weekday_pivot['keypresses'], color = bar_color_list) # bc stands for 'bar container'
ax.bar_label(bc, label_type = 'center', color = 'white') 
plt.title("Average Keypresses by Weekday")
plt.ylabel("Keypresses")
# See https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_label_demo.html
plt.savefig('graphs\\average_keypresses_by_weekday.png', dpi=400)

# %%
print("Total keypresess since", str(df_combined_daily_keypresses.iloc[0]['Date'])+":",'{:,}'.format(sum(df_combined_daily_keypresses['keypresses'])))

# %%
df_combined_daily_keypresses.tail(50) # Last 50 days

# %% [markdown]
# # (See above for recent keypress stats)

# %%
days_with_data = len(df_combined_daily_keypresses)
#  The following column cell shows the ranks immediately above the ranks for the most recent day.
keypresses_today = df_combined_daily_keypresses.iloc[-1]['keypresses']
percentile_today = df_combined_daily_keypresses.iloc[-1]['percentile']
rank_today = df_combined_daily_keypresses.iloc[-1]['rank']
print("Ranks are out of", days_with_data, "days.")
print(f"Today's keypresses: {keypresses_today}")
print(f"Your keypress totals 7, 28, and 365 days ago were \
{df_combined_daily_keypresses.iloc[-8]['keypresses']}, \
{df_combined_daily_keypresses.iloc[-29]['keypresses']}, \
and {df_combined_daily_keypresses.iloc[-366]['keypresses']}, respectively.")
# If your keypresses today are higher than these values, the moving averages
# associated with those values will increase.
print(f"Today's percentile: {percentile_today}")
print(f"Today's rank: {rank_today} (in front of {days_with_data - rank_today} days)")

# %% [markdown]
# Days ranked just ahead of today (along with today's rank):

# %%
df_days_with_higher_keypresses = df_combined_daily_keypresses.sort_values('rank').query("rank <= @rank_today").tail(11)
keypress_difference_list = [df_days_with_higher_keypresses.iloc[i]['keypresses'] - df_days_with_higher_keypresses.iloc[i+1]['keypresses'] for i in range(len(df_days_with_higher_keypresses) -1 )]
keypress_difference_list.append(np.NaN)
df_days_with_higher_keypresses['diff_from_following_day'] = keypress_difference_list
df_days_with_higher_keypresses['diff_from_current_day'] = df_days_with_higher_keypresses['keypresses'] - df_days_with_higher_keypresses.iloc[-1]['keypresses']
df_days_with_higher_keypresses

# %% [markdown]
# Looking for days with identical keypress totals:

# %%
duplicated_keypress_dates = df_combined_daily_keypresses[df_combined_daily_keypresses.duplicated(subset = 'keypresses', keep = False)].query('keypresses > 0').sort_values('keypresses', ascending = False)
# print(f"There have been {len(duplicated_keypress_dates)} dates that share keypresses with at least one other date. (This total does not include dates with 0 keypresses.)")
len(duplicated_keypress_dates)

# %%
duplicated_keypress_dates

# %%
# Total keypresses during MBA:

sum(df_combined_daily_keypresses.query(
    "Date >= '2020-08-20' & Date <= '2022-05-04'")['keypresses'])

# %%
df_combined_daily_keypresses

# %%
df_combined_daily_keypresses['Date']

# %% [markdown]
# ## Two different methods of visualizing this keypress data
# 
# Now that we have a DataFrame showing daily keypresses and multiple moving averages, it's time to visualize it! I'll show two methods for doing so here. First, I'll create a static graph using Matplotlib. Next, I'll use Plotly to create an interactive .html graph (along with a static .png copy of that graph.)

# %%
fig, ax = plt.subplots(figsize=[15,9])
fig.set_facecolor('white')
ax.set_ylim(0, max(df_combined_daily_keypresses['keypresses'])*1.1) # This 
# gives the y axis a bit more space to fit in the legend. Using a multiple is
# more flexible than is adding a specific number of keystrokes to the axis.)

ax.plot(df_combined_daily_keypresses['Date'], df_combined_daily_keypresses['keypresses'],label='Daily keypresses')
ax.plot(df_combined_daily_keypresses['Date'], df_combined_daily_keypresses['7_day_ma'],label='7-Day Moving Average')
ax.plot(df_combined_daily_keypresses['Date'], df_combined_daily_keypresses['28_day_ma'],label='28-Day Moving Average')
ax.plot(df_combined_daily_keypresses['Date'], df_combined_daily_keypresses['365_day_ma'],label='365-Day Moving Average')
plt.ylabel('Keypresses')
plt.legend(ncol = 2)
plt.title('Daily keypresses with 7-, 28-, and 365-day moving averages')
plt.savefig('graphs\\daily_keypresses_and_MAs.png', dpi=400)

# %% [markdown]
# A similar chart can be made using Plotly. The advantage of this chart is that, being HTML-based, it is interactive in nature. Thus, you can hover over the lines to view the values corresponding to those lines and zoom in to get a closer look at a particular section of the graph.

# %%
px_keypresses_line_chart = px.line(df_combined_daily_keypresses, x = 'Date', y = ['keypresses', '7_day_ma', '28_day_ma', '365_day_ma']) # Note that multiple y values can be passed to the line chart. 

px_keypresses_line_chart.write_html('graphs\\px_daily_keypresses_and_mas.html')

px_keypresses_line_chart 

# %% [markdown]
# Note that the above output will likely not show if you are viewing this notebook on GitHub. You'll instead need to download it to view the HTML output. (You can open it on your computer using a web browser.)

# %% [markdown]
# Plotly graphs can also be saved as static images:

# %%
image_width = 2600 # Larger widths will produce more detailed graphs but 
# smaller legends, so you may need to tweak this setting until you find a value
# that works for your own graph.
image_height = image_width * 9/16 # Preserves an HD/UHD aspect ratio

px_keypresses_line_chart.write_image('graphs\\px_daily_keypresses_and_mas_static.png', width = image_width, height = image_height, engine = 'kaleido')
# See https://plotly.com/python/static-image-export/

# %% [markdown]
# Here's a copy of this image:

# %%
Image('graphs\\px_daily_keypresses_and_mas_static.png')

# %%
test_df = df_combined_daily_keypresses.copy()

# %% [markdown]
# ## Monthly keypress totals:

# %%
df_monthly_keypresses = test_df.set_index('Date').resample('M').sum()['keypresses'].reset_index()
df_monthly_keypresses['Month'] = df_monthly_keypresses['Date'].dt.to_period('M')
df_monthly_keypresses['Year'] = df_monthly_keypresses['Date'].dt.to_period('Y')
df_monthly_keypresses.drop('Date', axis = 1, inplace = True)
df_monthly_keypresses['keypresses'] = df_monthly_keypresses.pop('keypresses')
df_monthly_keypresses

# %% [markdown]
# The following code shows my monthly keypresses from October to April during my 2-year MBA program.

# %%
fig, axes = plt.subplots(figsize=[10, 5])
fig.set_facecolor('white')
# Much of the following code was based on:
# https://matplotlib.org/3.5.0/gallery/lines_bars_and_markers/barchart.html
df_first_year = df_monthly_keypresses.query("Month >= '2020-10' & Month <= '2021-04'").copy()
df_second_year = df_monthly_keypresses.query("Month >= '2021-10' & Month <= '2022-04'").copy()
width = 0.4
x = np.arange(len(df_first_year))
axes.bar(x = x - width/2, height = df_first_year['keypresses'], label = 'First Year', width = width)
axes.bar(x = x + width/2, height = df_second_year['keypresses'], label = 'Second Year', width = width)
axes.set_xticks(x, ['Oct.', 'Nov.', 'Dec.', 'Jan.', 'Feb.', 'Mar.', 'Apr.'], rotation = 45)
plt.legend()
plt.ylabel('Keypresses (in millions)')
plt.title('Monthly keypresses (in millions) from October to April during my 2-year MBA')
plt.savefig('graphs\\monthly_keypresses_during_MBA.png', dpi=400)

# %% [markdown]
# List of tables in each WhatPulse database file:

# %%
# output = sqlalchemy_connection.execute("SELECT name FROM sqlite_schema WHERE type='table' ORDER BY name;") # From https://www.sqlite.org/faq.html#q7 
# print(output.all())
# Tables: [('application_active_hour',), ('application_bandwidth',), ('application_ignore',), ('application_uptime',), ('applications',), ('applications_upgrades',), ('computer_info',), ('country_bandwidth',), ('fact',), ('geekwindow_labels',), ('input_per_application',), ('keycombo_frequency',), ('keycombo_frequency_application',), ('keypress_frequency',), ('keypress_frequency_application',), ('keypresses',), ('last_update_time',), ('milestones',), ('milestones_log',), ('mouseclicks',), ('mouseclicks_frequency',), ('mouseclicks_frequency_application',), ('mousepoints',), ('network_interface_bandwidth',), ('network_interfaces',), ('network_interfaces_ignore',), ('network_protocol_bandwidth',), ('pending_applications_stats',), ('settings',), ('sqlite_sequence',), ('unpulsed_stats',), ('uptimes',)]

# %%
df_combined_daily_keypresses.to_csv('whatpulse_daily_keypresses.csv')

# %% [markdown]
# ## Hourly keypress stats:

# %%
hourly_keypress_db_list = []

for db_path in database_paths_list:
    hourly_keypress_db_list.append(generate_keypress_totals(database_path = db_path, level = 'hourly'))

df_hourly_keypresses = pd.concat([df for df in hourly_keypress_db_list]).reset_index(drop=True)

# As with my daily keypresess DataFrame, I'll use pivot_table() to group 
# multiple rows for the same day and hour into a single row. (These multiple 
# rows are the result of my using multiple computers during the same hour.)
df_hourly_keypresses = df_hourly_keypresses.pivot_table(index = ['day', 'hour'], values = 'keypresses', aggfunc = 'sum').reset_index().sort_values(['day', 'hour'])

df_hourly_keypresses.to_csv('data/df_combined_hourly_keypresses.csv', index = False)
df_hourly_keypresses

# %%
df_hourly_keypresses = pd.read_csv('data/df_combined_hourly_keypresses.csv')
df_hourly_keypresses['day'] = pd.to_datetime(df_hourly_keypresses['day'])

# %%
df_hourly_keypresses['day_and_hour'] = df_hourly_keypresses['day'] + pd.to_timedelta(df_hourly_keypresses['hour'], unit = 'H')
df_hourly_keypresses.set_index('day_and_hour', inplace = True)
df_hourly_keypresses

# %%
full_hourly_date_range = pd.date_range(start = first_date, end = last_date, freq = 'H')
df_hourly_keypresses = df_hourly_keypresses.reindex(full_hourly_date_range)
df_hourly_keypresses['keypresses'].fillna(0, inplace = True)
df_hourly_keypresses['keypresses'] = df_hourly_keypresses['keypresses'].astype('int')
df_hourly_keypresses['day'] = df_hourly_keypresses.index.date
df_hourly_keypresses['hour'] = df_hourly_keypresses.index.hour
df_hourly_keypresses

# %%
df_hourly_keypresses.reset_index(drop=True,inplace=True)

df_hourly_keypresses

# %%
df_hourly_keypresses[df_hourly_keypresses.duplicated(subset = ['day', 'hour'], keep = False)]

# %%
print("Most recent hourly keypress logs:\n",df_hourly_keypresses.iloc[-20:])

# %% [markdown]
# Most keypresses typed in a single hour:

# %%
df_hourly_keypresses.sort_values('keypresses', ascending = False)

# %% [markdown]
# Keypresses by hour:

# %%
df_hourly_avg_pivot = df_hourly_keypresses.pivot_table(index = 'hour', values = 'keypresses', aggfunc = 'mean').reset_index()
df_hourly_avg_pivot.rename(columns={'keypresses':'average'},inplace=True)
df_hourly_avg_pivot

# %%
px.bar(df_hourly_avg_pivot, x = 'hour', y = 'average')

# %%
df_hourly_pivot = df_hourly_keypresses.pivot_table(index = 'hour', values = 'keypresses', aggfunc = 'sum').reset_index()
df_hourly_pivot.rename(columns={'keypresses':'sum'},inplace=True)
df_hourly_pivot = df_hourly_pivot.merge(df_hourly_avg_pivot, on = 'hour')
total_keypresses = df_hourly_pivot['sum'].sum()
df_hourly_pivot['pct_of_total'] = 100* df_hourly_pivot['sum'] / total_keypresses
df_hourly_pivot

# %%
sum(df_hourly_pivot['pct_of_total']) # Making sure the percentages were calculated correctly (they should add up to 100%)

# %% [markdown]
# Using Plotly to create both interactive (.html) and static (.png) hourly keypress charts:

# %%
px_keypresses_by_hour = px.bar(df_hourly_pivot, x = 'hour', y = 'pct_of_total')
px_keypresses_by_hour.write_html(r'graphs\px_keypresses_by_hour.html')
px_keypresses_by_hour

# %%
image_width = 1920
image_height = image_width * 9/16
px_keypresses_by_hour.write_image(r'graphs\px_keypresses_by_hour_static.png', width = image_width, height = image_height, engine = 'kaleido')
Image(r'graphs\px_keypresses_by_hour_static.png')

# %%
end_time = time.time()
run_time = end_time - start_time
run_minutes = run_time // 60
run_seconds = run_time % 60
# print("Completed run at",time.ctime(end_time),"(local time)")
# print("Total run time:",'{:.2f}'.format(run_time),"second(s) ("+str(run_minutes),"minute(s) and",'{:.2f}'.format(run_seconds),"second(s))") # Only meaningful when the program is run nonstop from start to finish


# %%
input() # Keeps console window open when running the file in a command prompt.
# It's not necessary for the Jupyter Notebook, but when I export this notebook
# as a Python script and then run the script on a scheduled basis, this 
# line gives me time to read the output.
# See nosklo's response at: https://stackoverflow.com/a/1000968/13097194


