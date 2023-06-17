# %% [markdown]
# # WhatPulse Keypress Stats Analyzer
# 
# By Kenneth Burchfiel
# 
# Released under the MIT license
# 
# *(I am not affiliated with WhatPulse (https://whatpulse.org) but highly recommend trying out the program, which I've used since September 2008. You can find my online WhatPulse page here: https://whatpulse.org/KBurchfiel)*
# 
# This script allows you to perform various analyses of your WhatPulse typing stats. It does so by accessing the local WhatPulse database on your computer (along with other copies of databases you might want to include); reading this data into Pandas DataFrames, and then summarizing and graphing that data. The output of the script includes:
# 
# 1. Rolling average calculations at the 7-, 28-, and 365-day level
# 2. Percentile and rank calculations
# 3. Weekly and hourly keypress stats
# 4. Static (.png) and interactive (.html) keypress charts
# 
# By converting this notebook into a Python file and then instructing your computer to run it on an hourly basis, you can also keep track of how much you're typing during your day. Which is sort of nerdy, to be honest. But so is this whole program ;) 

# %% [markdown]
# I'll first import a number of packages that the program will use:

# %%
import time
start_time = time.time() # Allows the program's runtime to be measured
import pandas as pd
import sqlalchemy
import numpy as np
from datetime import date
import statsmodels.api as sm
from scipy.stats import percentileofscore
import plotly.express as px
import kaleido
from IPython.display import Image
import datetime

# %% [markdown]
# I'll next define the names of folders that will store various outputs, along with default settings to use when converting interactive charts to static .png files:

# %%
static_graphs_folder = 'graphs/static/'
interactive_graphs_folder = 'graphs/interactive/'
data_folder = 'data'

default_image_height = 540
default_aspect_ratio = 16 / 9 # Standard HD/UHD aspect ratio
default_image_width = default_image_height * default_aspect_ratio
default_image_scale = 5 # Creating a smaller graph (e.g. one 540 pixels 
# in height) and then scaling it helps keep the text a bit larger.

# %% [markdown]
# ## Importing Whatpulse data
# 
# In order to analyze my WhatPulse data, I'll first need to import it from my local Whatpulse SQLite database. I'll also import a copy of the Whatpulse SQLite database stored on my old laptop (so that my analysis doesn't have to be limited to my current computer's keypress data.)
# 
# You'll of course need to update the following cell with the path(s) to your own WhatPulse database(s).
# 
# Note: if you want to run this program on your own, but don't have access to a WhatPulse database, you can still run this program (assuming you've downloaded or cloned it from GitHub). Skip ahead to the line where I read in 

# %%
database_paths_list = ['C:/Users/kburc/AppData/Local/whatpulse/whatpulse.db', 
'C:/Users/kburc/D1V1/Documents/whatpulse_database_backups/a13r2_whatpulse.db',
'G:/My Drive/whatpulse_database_backups/linux_whatpulse.db'] 
# Note that the first path is to my computer's active database, 
# whereas additional paths point towards copies of databases from 
# other computers that I have used.

# %% [markdown]
# The following function analyzes each database's  table at either the daily or hourly level.

# %%
def generate_keypress_totals(database_path, level = 'daily'):
    '''The level argument can be 'daily', in which case the DataFrame
    returned by the function will show daily keypress totals, or 'hourly',
    which will preserve the hourly keypress totals in the original database.'''
    file_name = database_path.split('/')[-1] # Retrieves the final element
    # of the path (e.g. the file name)
    sqlalchemy_sqlite_engine = sqlalchemy.create_engine(
        'sqlite:///'+database_path) 
    # Based on https://docs.sqlalchemy.org/en/13/dialects/sqlite.html#connect-strings
    sqlalchemy_connection = sqlalchemy_sqlite_engine.connect()
    df_keypresses = pd.read_sql("select * from keypresses", 
    con = sqlalchemy_sqlite_engine) # Simply reads all of the data from this 
    # Capitalizing column names so that less renaming will be necessary
    # when creating graphs:
    df_keypresses.columns = [column.title() for column in df_keypresses.columns]

    # table into a Pandas DataFrame
    df_keypresses = df_keypresses.query("Day != '0000-00-00'").copy() # Removes
    # this blank date value from the database if it happens to be there
    if level == 'daily': # In this case, we'll want to combine hourly keypress
        # totals into a single row for each day
        df_keypresses = df_keypresses.pivot_table(
            index = 'Day', values = 'Count', aggfunc = 'sum')
        df_keypresses.sort_values('Day', inplace = True)
    elif level == 'hourly': # The original data is already displayed 
        # at the hourly level, so there's no need for a pivot_table() call.
        df_keypresses.sort_values(['Day', 'Hour'], inplace = True)
    else:
        raise ValueError("Unrecognized level argument passed to function.")
    df_keypresses.rename(columns={'Count':'Keypresses'},inplace=True)  
    return df_keypresses

# %% [markdown]
# I'll now run generate_keypress_totals in order to create a record of daily keypresses for both my current database and a copy of a past database. (I'll look at hourly keypress totals later on.)

# %%
keypress_databases_list = []

for path in database_paths_list: # This loop creates a DataFrame for
    # each WhatPulse database stored in database_paths_list.
    # print("Now loading:",path)
    keypress_databases_list.append(generate_keypress_totals(
        path, level = 'daily'))

# I'll now combine these tables into a single DataFrame.
df_combined_daily_keypresses = pd.concat(
    [keypress_databases_list[i] for i in range(len(keypress_databases_list))])
df_combined_daily_keypresses.sort_index(inplace=True)

# At this point, my copy of df_combined_daily_keypresses has multiple
# entries for days in which I logged keys on multiple operating systems.
# Therefore, the following line groups these entries into a single row
# for each date.
df_combined_daily_keypresses = df_combined_daily_keypresses.reset_index(
).pivot_table(index = 'Day', values = 'Keypresses', aggfunc = 'sum')
df_combined_daily_keypresses.index = pd.to_datetime(
    df_combined_daily_keypresses.index)

df_combined_daily_keypresses.to_csv('data/df_combined_daily_keypresses.csv')
df_combined_daily_keypresses

# %% [markdown]
# The following line rebuilds df_combined_daily_keypresses using a copy of the DataFrame that got exported to a .csv file earlier on. This cell allows allow you to run this script even if you don't have your own WhatPulse database.

# %%
df_combined_daily_keypresses = pd.read_csv(
    'data/df_combined_daily_keypresses.csv', index_col='Day')
# The following line makes the index compatible with
# date operations that the following code block will perform.
df_combined_daily_keypresses.index = pd.to_datetime(
df_combined_daily_keypresses.index)
df_combined_daily_keypresses

# %% [markdown]
# The following code block fills in the DataFrame with missing dates (e.g. dates in which I did not have any keypresses). I want to add in those missing dates so that I can calculate more accurate rolling keypress averages.

# %%
first_date = df_combined_daily_keypresses.index[0]
last_date = df_combined_daily_keypresses.index[-1]
full_date_range = pd.date_range(start=first_date, end = last_date) 
# https://pandas.pydata.org/docs/reference/api/pandas.date_range.html
df_combined_daily_keypresses = df_combined_daily_keypresses.reindex(
    full_date_range, fill_value=0) 
# See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reindex.html

df_combined_daily_keypresses.index.name = 'Date'
df_combined_daily_keypresses.reset_index(inplace=True) 

# %% [markdown]
# Now that I have a more complete daily keypress history, I can begin performing analyses on this data. First, I will use the rolling() function within Pandas to calculate 7-, 28-, and 365-day moving averages. Next, I will use the Series.rank() function twice to (1) calculate the percentile of each keypress and (2) determine each keypress's rank within the entire DataFrame.

# %%
df_combined_daily_keypresses['7_day_ma'] = df_combined_daily_keypresses[
    'Keypresses'].rolling(7).mean()
df_combined_daily_keypresses['28_day_ma'] = df_combined_daily_keypresses[
    'Keypresses'].rolling(28).mean() 
# I switched from a 30-day MA to a 28-day MA because my average keypresses vary 
# significantly by weekday, meaning that a 30-day average would be skewed by 
# the number of Saturdays and Sundays present in the data.
df_combined_daily_keypresses['365_day_ma'] = df_combined_daily_keypresses[
    'Keypresses'].rolling(365).mean()

df_combined_daily_keypresses['percentile'] = 100*df_combined_daily_keypresses[
    'Keypresses'].rank(pct=True)
# The pct=True argument generates percentile values for each keypress value.
df_combined_daily_keypresses['rank'] = df_combined_daily_keypresses[
    'Keypresses'].rank(ascending = False)
# Ascending = False instructs the function to assign the lowest number 
# (e.g. 1) to the highest value.
# If two dates are tied, the rank may end in a 0.5. That's why these ranks 
# appear as floats but not integers. 
df_combined_daily_keypresses

# %% [markdown]
# Next, I'll add in weekdays using the Series.map() function within Pandas.

# %%
weekday_dict = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',
5:'Saturday',6:'Sunday'}
# weekday numbers in Python begin with 0 for Monday and end with 6 for Sunday. 
# See https://docs.python.org/3/library/datetime.html#datetime.date.weekday
weekday_dict

# %% [markdown]
# The following cell adds a 'Weekday' column to the DataFrame by (1) calculating the numerical weekday values for each date in the 'Date' column, then (2) using weekday_dict to map these numerical values to the weekday names. 

# %%
df_combined_daily_keypresses['Weekday'] = df_combined_daily_keypresses[
    'Date'].dt.weekday.map(weekday_dict)

# %% [markdown]
# Here are my daily keypress statistics for the last 10 days. Note the presence of the moving average, percentile, rank, and weekday columns.

# %%
df_combined_daily_keypresses.tail(10)

# %% [markdown]
# # Data Analysis

# %% [markdown]
# I'll start my data analysis by calculating some summary statistics. In the output below, 'count' shows the number of days since the earliest entry in my database, rather than the number of days for which I have keypress data (as the table also includes days without any keypresses).

# %%
df_combined_daily_keypresses['Keypresses'].describe()

# %% [markdown]
# Next, I'll calculate my top 50 daily keypress totals. Note that the top-ranked date has a rank of 1 and a percentile of 100.

# %%
df_max_keypresses = df_combined_daily_keypresses.sort_values(
    'Keypresses', ascending = False).reset_index(drop=True)
df_max_keypresses.insert(0, 'Rank', df_max_keypresses.index+1)
keypress_difference_list = [
    df_max_keypresses.iloc[i]['Keypresses'] - df_max_keypresses.iloc[i+1][
        'Keypresses'] for i in range(len(df_max_keypresses) -1 )]
# The above list comprehension calculates the difference 
# between each row and the row below it. This isn't possible for the final row,
# so I'll instead append a np.Nan to it.)
keypress_difference_list.append(np.NaN)
df_max_keypresses['difference_from_lower_rank'] = keypress_difference_list
df_max_keypresses.head(50)

# %% [markdown]
# Next, I'll create a visualization of these 50 dates. I will use Plotly instead of Matplotlib so that I can create both interactive (.html) and static (.png) versions of each chart. The static versions are easier to share, but the interactive versions are easier to analyze, as you can hover over the chart to get more information and zoom in on areas of particular interest.
# 
# First, I'll create a function that will make it easier to save .html and .png versions of these charts:

# %%
def save_chart(fig, file_name, 
interactive_graphs_folder = interactive_graphs_folder, 
static_graphs_folder = static_graphs_folder):
    '''Saves a Plotly figure as a .HTML and .PNG file.
    fig: The Plotly figure to save.
    file_name: The filename to use. Don't include the extension.'''
    fig.write_html(
        interactive_graphs_folder+'/'+file_name+'.html')
    # Saving the interactive chart to a .png file:
    fig.write_image(
        static_graphs_folder+'/'+file_name+'.png',
        width = default_image_width, height = default_image_height,
        scale = default_image_scale)

# %%
# Generating the interactive chart:
fig_max_keypresses = px.bar(df_max_keypresses.head(50), 
x = 'Rank', y = 'Keypresses', title = 'Top 50 Daily Keypress Totals', 
text = 'Keypresses')

save_chart(fig_max_keypresses, 'top_50_keypresses')

# %% [markdown]
# Here's a look at the chart:
# 
# *Note: this and other .html-based charts will probably not display for you within GitHub. In order to view them, you will need to download the files from GitHub (e.g. by cloning them) and view them on your computer.*

# %%
fig_max_keypresses

# %% [markdown]
# Here's a copy of the static chart in case the .html chart above did not load for you: (I'll also display static versions of later charts as well.)

# %%
Image(static_graphs_folder+'top_50_keypresses.png')

# %% [markdown]
# ## Keypress percentile data:
# 
# First, I'll calculate the keypress totals equal to the 90th through 100th percentiles (in 1-percentile increments).

# %%
df_combined_daily_keypresses['Keypresses'].describe(
    percentiles=np.linspace(0.9,1,11))[5:-1]
# The first four rows and final row provide additional descriptive statistics,
# so we can get rid of them
# by adding [4:-1] to the end of this line.
# There is probably a more elegant solution that retrieves only percentile
# data, but this option is fairly straightforward.

# %% [markdown]
# Scipy's percentileofscore() function can be used to calculate the percentile corresponding to a specific keypress total. For instance, here's the percentile for a day with only a single keypress: (The percentile may be higher than you'd expect due to the presence of many days with 0 keypresses)

# %%
percentileofscore(df_combined_daily_keypresses['Keypresses'], 1)

# %% [markdown]
# Next, I'll create a DataFrame showing the keypresses corresponding to every 5th percentile.

# %%
df_percentiles = df_combined_daily_keypresses['Keypresses'].describe(
percentiles=np.linspace(0,1,21))[4:-1].reset_index().sort_values(
    'Keypresses', ascending = False).reset_index(drop=True).rename(
        columns={'index':'percentile'})
# Calling reset_index() twice appears inefficient, but it makes it easier
# to sort by a particular value (in this case, keypresses).
keypress_difference_list = [df_percentiles.iloc[
    i, 1] - df_percentiles.iloc[i+1, 1] for i in range(len(df_percentiles) -1 )]
# This list comprehension calculates the difference between each row and 
# the row below it. This isn't possible for the final row,
# so we'll instead append a np.Nan to it.)
keypress_difference_list.append(np.NaN)
df_percentiles['difference_from_lower_percentile'] = keypress_difference_list
df_percentiles

# %% [markdown]
# The following chart shows the difference in keypresses between the different percentiles on this list.

# %%
px.bar(df_percentiles, x = 'percentile', 
       y = 'difference_from_lower_percentile')

# %% [markdown]
# And here's a histogram that shows the frequency of different keypress ranges:

# %%
fig_keypress_hist = px.histogram(df_combined_daily_keypresses, 
x = 'Keypresses', nbins = 40, text_auto = True, 
title = 'Histogram of Daily Keypress Totals')
# See https://plotly.com/python/histograms/
fig_keypress_hist.update_layout(bargap=0.1, yaxis_title = 'Days')
# https://plotly.com/python/histograms/
save_chart(fig_keypress_hist, 'daily_keypress_histogram')
fig_keypress_hist

# %%
Image(static_graphs_folder+'daily_keypress_histogram.png')

# %% [markdown]
# I'll now create a pivot table that shows my average keypresses per weekday:

# %%
df_weekday_pivot = df_combined_daily_keypresses.pivot_table(
    index = 'Weekday', values = 'Keypresses', aggfunc = 'mean').sort_values(
    'Weekday', ascending = False).reset_index()
# Adding in a Weekday_Order column so that 
# weekdays can be sorted chronologically:
df_weekday_pivot['Weekday_Order'] = df_weekday_pivot['Weekday'].map(
{'Sunday':0,'Monday':1,"Tuesday":2,"Wednesday":3,
"Thursday":4,"Friday":5,"Saturday":6})
df_weekday_pivot

# %%
df_weekday_pivot.sort_values('Weekday_Order', inplace = True)
df_weekday_pivot

# %% [markdown]
# Graphing my average keypresses per weekday:

# %%
fig_keypresses_by_weekday = px.bar(df_weekday_pivot, x = 'Weekday', 
y = 'Keypresses', text_auto = '.0f', color = 'Keypresses', 
color_continuous_scale = 'PrGn',
title = 'Average Keypresses by Weekday')
# For text_auto, see: 
# https://plotly.com/python-api-reference/generated/plotly.express.bar

# For color scale options and usage, 
# visit https://plotly.com/python/builtin-colorscales/
fig_keypresses_by_weekday.update_traces(marker_line_color='rgb(0,0,0)', 
                                        marker_line_width=0.5, opacity=1)
fig_keypresses_by_weekday

save_chart(fig_keypresses_by_weekday, file_name = 'keypresses_by_weekday')

# %%
Image(static_graphs_folder+'keypresses_by_weekday.png')

# %% [markdown]
# ### Total keypresses since first date in DataFrame:

# %%
print("Total keypresess since", str(
df_combined_daily_keypresses.iloc[0]['Date'])+":",'{:,}'.format(
sum(df_combined_daily_keypresses['Keypresses'])))

# %% [markdown]
# ### Keypresses over the past 50 days:

# %%
df_combined_daily_keypresses.tail(50) # Last 50 days

# %% [markdown]
# The following cell outputs various keypress statistics. When this script is run hourly, these statistics (along with other ones) will then appear in the console terminal.

# %%
days_with_data = len(df_combined_daily_keypresses)
#  The following column cell shows the ranks immediately above the ranks for the most recent day.
keypresses_today = df_combined_daily_keypresses.iloc[-1]['Keypresses']
percentile_today = df_combined_daily_keypresses.iloc[-1]['percentile']
rank_today = df_combined_daily_keypresses.iloc[-1]['rank']
print("Ranks are out of", days_with_data, "days.")
print(f"Today's keypresses: {keypresses_today}")
print(f"Your keypress totals yesterday and 7, 28, and 365 days ago were \
{df_combined_daily_keypresses.iloc[-2]['Keypresses']}, \
{df_combined_daily_keypresses.iloc[-8]['Keypresses']}, \
{df_combined_daily_keypresses.iloc[-29]['Keypresses']}, \
and {df_combined_daily_keypresses.iloc[-366]['Keypresses']}, respectively.")
# If your keypresses today are higher than these values, the moving averages
# associated with those values will increase.
print(f"Today's percentile: {round(percentile_today, 3)}")
print(f"Today's rank: {rank_today} \
(in front of {days_with_data - rank_today} days)")

# %% [markdown]
# Days ranked just ahead of today (along with today's rank):

# %%
df_days_with_higher_keypresses = df_combined_daily_keypresses.sort_values(
    'rank').query("rank <= @rank_today").tail(11)
keypress_difference_list = [df_days_with_higher_keypresses.iloc[i][
    'Keypresses'] - df_days_with_higher_keypresses.iloc[i+1][
        'Keypresses'] for i in range(len(df_days_with_higher_keypresses) -1 )]
keypress_difference_list.append(np.NaN)
df_days_with_higher_keypresses[
    'diff_from_following_day'] = keypress_difference_list
df_days_with_higher_keypresses[
    'diff_from_current_day'] = df_days_with_higher_keypresses[
        'Keypresses'] - df_days_with_higher_keypresses.iloc[-1]['Keypresses']
df_days_with_higher_keypresses

# %% [markdown]
# Looking for days with identical non-zero keypress totals:

# %%
duplicated_keypress_dates = df_combined_daily_keypresses[
df_combined_daily_keypresses.duplicated(
subset = 'Keypresses', keep = False)].query('Keypresses > 0').sort_values(
'Keypresses', ascending = False)
len(duplicated_keypress_dates)

# %%
duplicated_keypress_dates

# %%
df_combined_daily_keypresses

# %%
df_combined_daily_keypresses['Date']

# %% [markdown]
# ## Plotting Chronological Keypress Data
# 
# Now that we have a DataFrame showing daily keypresses and multiple moving averages, it's time to visualize it! The advantage of creating this chart within Plotly is that, being HTML-based, it is interactive in nature. Thus, you can hover over the lines to view the values corresponding to those lines and zoom in to get a closer look at a particular section of the graph. As before, though, this graph can also be saved as a static image.

# %%
fig_keypresses_line_chart = px.line(df_combined_daily_keypresses, x = 'Date', 
y = ['Keypresses', '7_day_ma', '28_day_ma', '365_day_ma'],
labels = {'variable':'Metric','value':'Keypresses'}, title = 
'Daily Keypresses and 7/28/365-day Moving Averages') 
# Note that multiple y values can be passed to the line chart. 

save_chart(fig_keypresses_line_chart, 'px_daily_keypresses_and_mas')

fig_keypresses_line_chart

# %%
Image(static_graphs_folder+'px_daily_keypresses_and_mas.png')

# %% [markdown]
# ## Monthly keypress totals:

# %%
df_monthly_keypresses = df_combined_daily_keypresses.copy().set_index(
    'Date').resample('M').sum()['Keypresses'].reset_index()
df_monthly_keypresses['Month'] = df_monthly_keypresses['Date'].dt.to_period('M')
df_monthly_keypresses['Year'] = df_monthly_keypresses['Date'].dt.to_period('Y')
df_monthly_keypresses.drop('Date', axis = 1, inplace = True)
df_monthly_keypresses['Keypresses'] = df_monthly_keypresses.pop('Keypresses')
df_monthly_keypresses

# %% [markdown]
# # Saving the updated version of this DataFrame to a .csv file:

# %%
df_combined_daily_keypresses.to_csv(
    'data/df_combined_daily_keypresses_updated.csv')

# %% [markdown]
# ## Hourly keypress stats:
# 
# In order to calculate hourly keypress statistics, we'll need to create a new DataFrame that aggregates keypresses by hour instead of by day.

# %%
hourly_keypress_db_list = []

for db_path in database_paths_list:
    hourly_keypress_db_list.append(generate_keypress_totals(
        database_path = db_path, level = 'hourly'))

df_hourly_keypresses = pd.concat([
    df for df in hourly_keypress_db_list]).reset_index(drop=True)

# As with my daily keypresess DataFrame, I'll use pivot_table() to group 
# multiple rows for the same day and hour into a single row. (These multiple 
# rows are the result of my using multiple computers during the same hour.)
df_hourly_keypresses = df_hourly_keypresses.pivot_table(index = [
    'Day', 'Hour'], values = 'Keypresses', aggfunc = 'sum').reset_index().sort_values(['Day', 'Hour'])

df_hourly_keypresses.to_csv(
    'data/df_combined_hourly_keypresses.csv', index = False)
df_hourly_keypresses

# %% [markdown]
# Recreating the DataFrame from a .csv file so that the following cells can be run by users who don't yet have their own WhatPulse database:

# %%
df_hourly_keypresses = pd.read_csv('data/df_combined_hourly_keypresses.csv')


# %%
df_hourly_keypresses['Day'] = pd.to_datetime(df_hourly_keypresses['Day'])
# Creating a combined day/hour column:
df_hourly_keypresses['Day_and_Hour'] = df_hourly_keypresses[
    'Day'] + pd.to_timedelta(df_hourly_keypresses['Hour'], unit = 'H')
df_hourly_keypresses.set_index('Day_and_Hour', inplace = True)
df_hourly_keypresses

# %% [markdown]
# The following cells add in hours with 0 keypresses (of which there are many!). In order to include current/previous hours for the current day in my results, I'll add in keypresses up to the start of the next day (i.e. midnight), then limit the results so that they don't extend beyond the current hour.

# %% [markdown]
# Calculating tomorrow's date:

# %%
last_date_for_hourly_keypress_log = last_date + datetime.timedelta(days = 1)
last_date_for_hourly_keypress_log

# %%
pd.Timestamp.now()

# %% [markdown]
# Adding hours without keypresses to the DataFrame:

# %%
full_hourly_date_range = pd.date_range(start = first_date, 
end = last_date_for_hourly_keypress_log, freq = 'H')
df_hourly_keypresses = df_hourly_keypresses.reindex(full_hourly_date_range)
df_hourly_keypresses['Keypresses'].fillna(0, inplace = True)
df_hourly_keypresses['Keypresses'] = df_hourly_keypresses[
    'Keypresses'].astype('int')
# Retrieving date and hour values from the index:
df_hourly_keypresses['Day'] = df_hourly_keypresses.index.date
df_hourly_keypresses['Hour'] = df_hourly_keypresses.index.hour
df_hourly_keypresses

# %% [markdown]
# Limiting the results to the period leading up to the current hour:

# %%
df_hourly_keypresses = df_hourly_keypresses[
    df_hourly_keypresses.index < pd.Timestamp.now()].copy()
df_hourly_keypresses.reset_index(drop=True,inplace=True)

# %% [markdown]
# Calculating rolling 24-hour keypress totals:

# %%
df_hourly_keypresses['keypresses_over_last_24_hours'] = df_hourly_keypresses[
    'Keypresses'].rolling(24).sum()
df_hourly_keypresses

# %% [markdown]
# Printing out recent hours with keypresses: (This data will also appear on the terminal window when the program is run automatically, which allows you to track your recent productivity.)

# %%
print("Keypresses over the last 25 hours (excluding hours \
with 0 keypresses):\n",df_hourly_keypresses.iloc[-25:].query("Keypresses > 0"))
# Hours with 0 keypresses are removed in order to give the console output more
# space to fit on a single line.

# %% [markdown]
# Keypresses for the last 48 hours (including hours with 0 keypresses, now that they have been added to our table):

# %%
df_hourly_keypresses.iloc[-48:]

# %% [markdown]
# Making sure that all rows with the same day and hour (e.g. from multiple WhatPulse databases) have been merged into the same row:

# %%
df_hourly_keypresses[df_hourly_keypresses.duplicated(
    subset = ['Day', 'Hour'], keep = False)]

# %% [markdown]
# Most keypresses typed in a single hour within the entire dataset:

# %%
df_hourly_keypresses.sort_values('Keypresses', ascending = False).head(50)

# %% [markdown]
# Average keypresses by hour:

# %%
df_hourly_pivot = df_hourly_keypresses.pivot_table(index = 'Hour', 
values = 'Keypresses', aggfunc = ['mean', 'sum']).reset_index()
df_hourly_pivot.columns = df_hourly_pivot.columns.to_flat_index()
# At this point, the columns will read: "(hour, ),	(mean, Keypresses), and
# (sum, Keypresses)". We could convert them to regular columns via a loop,
# but since there are only 3, the simplest solution is to simply rename them
# as follows:
df_hourly_pivot.columns = ['Hour', 'Average Keypresses', 'Keypresses']

# Determining the percentage of total keypresses typed each hour:
total_keypresses = df_hourly_pivot['Keypresses'].sum()
df_hourly_pivot['pct_of_total'] = 100* df_hourly_pivot[
    'Keypresses'] / total_keypresses
df_hourly_pivot


df_hourly_pivot

# %%
sum(df_hourly_pivot['pct_of_total']) 
# Making sure the percentages were calculated correctly 
# (they should add up to 100%)

# %% [markdown]
# ## Plotting average keypresses by hour:
# 
# Not surprisingly, my hourly keypress averages are highest during the workday and lowest in the middle of the night, although I've been pretty active in the late evening hours also. (More on this below.)

# %%
fig_hourly_keypresses = px.bar(df_hourly_pivot, x = 'Hour', 
y = 'Average Keypresses', text_auto = '.0f', 
title = 'Average Keypresses by Hour')
save_chart(fig_hourly_keypresses, 'average_keypresses_by_hour')
fig_hourly_keypresses

# %%
Image(static_graphs_folder+'average_keypresses_by_hour.png')

# %%
df_hourly_keypresses['Day'][0]

# %% [markdown]
# Saving the updated version of df_hourly_keypresses to a .csv file:

# %%
df_hourly_keypresses.to_csv('data/df_combined_hourly_keypresses_updated.csv')

# %% [markdown]
# ## Data analysis question: Did marriage change my typing patterns?
# 
# I got married in April 2023, and I suspected that my keypress distributions as a married man might skew earlier than they did when I was a bachelor. I decided to investigate this graphically by creating subsets of df_hourly_keypresses that contained pre-marriage and post-marriage datasets, then comparing them via a grouped bar chart.

# %% [markdown]
# Setting datetime.date() values that will be used to filter df_hourly_keypresses:

# %%
post_mba_work_start_date = datetime.date(2022, 6, 21) # I began my current
# full-time work in June 2022 after finishing my MBA. I chose to limit the
# dataset to this date range so that my results wouldn't be influenced
# by my time as an MBA student (which featured more irregular computer hours).
last_day_before_marriage = datetime.date(2023, 4, 14)
day_after_honeymoon = datetime.date(2023, 4, 29) # I didn't type much at all
# on my honeymoon, so I excluded this period from my analysis in order not
# to skew the average keypress totals downward. 

# %%
df_hourly_keypresses_pre_marriage = df_hourly_keypresses.query(
"Day > @post_mba_work_start_date & Day <= @last_day_before_marriage"
).pivot_table(index = 'Hour', values = 'Keypresses', aggfunc = 'mean').reset_index()
df_hourly_keypresses_pre_marriage['Period'] = 'Before Marriage'

df_hourly_keypresses_post_marriage = df_hourly_keypresses.query(
    "Day >= @day_after_honeymoon").pivot_table(
        index = 'Hour', values = 'Keypresses', aggfunc = 'mean').reset_index()
df_hourly_keypresses_post_marriage['Period'] = 'After Marriage'

# Combining these two DataFrames together:
df_hourly_keypresses_by_period = pd.concat([df_hourly_keypresses_pre_marriage,
df_hourly_keypresses_post_marriage])

df_hourly_keypresses_by_period


# %% [markdown]
# My daily keypress counts have increased slightly since getting married (at least when the honeymoon isn't taken into account):

# %%
df_hourly_keypresses_by_period.pivot_table(
    index = 'Period', values = 'Keypresses', aggfunc = 'sum')

# %% [markdown]
# However, as the following chart shows, the hourly distribution of these keypresses has changed significantly. I'm now typing much less late at night and am getting more keypresses in earlier in the day.

# %%
fig_keypresses_by_period = px.bar(df_hourly_keypresses_by_period, x = 'Hour', 
y = 'Keypresses', color = 'Period', barmode = 'group', text_auto = '.0f', 
title = 'Average Keypresses by Hour Before and After Getting Married')
save_chart(fig_keypresses_by_period, 'keypresses_before_and_after_marriage')
# See https://plotly.com/python/bar-charts/ 
# for the use of the 'color' and 'barmode' arguments.
fig_keypresses_by_period

# %%
Image(static_graphs_folder+'keypresses_before_and_after_marriage.png')

# %%
end_time = time.time()
run_time = end_time - start_time
run_minutes = run_time // 60
run_seconds = run_time % 60

run_minutes, run_seconds


# %% [markdown]
# The input() function within the following cell keeps the console window open when running the file in a command prompt. It's not necessary for the Jupyter Notebook, but when I export this notebook as a Python script and then run the script on a scheduled basis, this line gives me time to read the output.
# See nosklo's response at: https://stackoverflow.com/a/1000968/13097194

# %%
print("The program has finished running. Press Enter to exit.") # Lets me know
# that I can now close the program after it has finished running in a console
# window. (I wouldn't want to close it while the 
# graphs are in the process of being generated.)
input() 


# %% [markdown]
# That's it for this program! I hope you enjoy using it to analyze your own WhatPulse keypress statistics.


