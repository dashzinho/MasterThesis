import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load file
file = "../usyields.csv"
df = pd.read_csv(file, index_col=0)

# fix data
columns_to_drop = ['1 Mo', '2 Mo', '4 Mo', '20 Yr', '30 Yr']
df = df.drop(columns=columns_to_drop)
df.index = pd.to_datetime(df.index, format="%m/%d/%y")
df = df.sort_index()
df = df.resample('M').last()

# Define the new column names
new_column_names = {
    '3 Mo': '3', '6 Mo': '6', '1 Yr': '12', 
    '2 Yr': '24', '3 Yr': '36', 
    '5 Yr': '60', '7 Yr': '72', 
    '10 Yr': '120'
}

df = df.rename(columns=new_column_names)
df['3'].interpolate(method='linear', inplace=True)

df.to_csv("../data.csv")


# Create a new folder if it doesn't exist
folder_name = '../Figures/Data'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

#Data visualization

#graph 1

# Evolution through time
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['3'] , label='3 months', color='C1')
plt.plot(df.index, df['6'] , label='6 months', color='C2')
plt.plot(df.index, df['12'] , label='12 months', color='C3')
plt.plot(df.index, df['24'] , label='24 months', color='C4')
plt.title(f'Par yields evolution for smaller maturities')
plt.xlabel('Date')
plt.ylabel('Yield')
subset_dates = df.index[::60]
plt.xticks(subset_dates, rotation=45)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(folder_name, 'yields_evolution_1.png'), bbox_inches='tight') 
plt.show()

#graph2

plt.figure(figsize=(10, 6))
plt.plot(df.index, df['36'] , label='36 months', color='C5')
plt.plot(df.index, df['60'] , label='60 months', color='C6')
plt.plot(df.index, df['72'] , label='72 months', color='C7')
plt.plot(df.index, df['120'] , label='120 months', color='C8')
plt.title(f'Par yields evolution for higher maturities')
plt.xlabel('Date')
plt.ylabel('Yield')
subset_dates = df.index[::60]
plt.xticks(subset_dates, rotation=45)
plt.xticks(rotation=45) 
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(folder_name, 'yields_evolution_2.png'), bbox_inches='tight')
plt.show()

#graph3

# List of specific dates
specific_dates = ['1990-01-31', '2000-01-31', '2010-01-31','2021-12-31' ] 

for date in specific_dates:
    
    plt.figure(figsize=(10, 6))
  
    specific_date_df = df.loc[date]

    plt.plot(specific_date_df.index, specific_date_df.values, marker='o')
    plt.title(f'Upwards trend Yield Curve on {date}')
    plt.xlabel('Maturity')
    plt.ylabel('Yield')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.savefig(os.path.join(folder_name, f'yield_curve_{date}.png'), bbox_inches='tight')
    plt.show()


# List of specific dates
specific_dates = ['1999-01-31','2001-01-31', '2007-01-31','2019-04-30' ]

for date in specific_dates:
    
    plt.figure(figsize=(10, 6))
  
    specific_date_df = df.loc[date]

    plt.plot(specific_date_df.index, specific_date_df.values, marker='o')
    plt.title(f'Downwards trend Yield Curve on {date}')
    plt.xlabel('Maturity')
    plt.ylabel('Yield')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.savefig(os.path.join(folder_name, f'yield_curve_{date}.png'), bbox_inches='tight')
    plt.show()