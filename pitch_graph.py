#!/usr/bin/python
import matplotlib.pyplot as plt
import pandas as pd
from spotify_audio_analysis import *

# import data
data = pd.read_csv(r'C:\Users\Andrew\PycharmProjects\problem3\spotify-audio-analysis\results - results.csv')

# set columns to look at
df = pd.DataFrame(data=data, columns=['file', 'layers', 'val_loss'])

# set rows to look at
df.set_index("file", inplace=True)
df = df.loc['pitch']

# turn layers column to strings
df['layers'] = df['layers'].astype(str)
# print result
print(df)

# only specific row
df1 = df.iloc[[0]]
print(df1)

# only specific cell [row][column-1] (row 1 of layer column)
str1 = df.iloc[0][0]
print("only specific cell\n", str1)

to_remove = "[]"
for r in to_remove:
    str1 = str1.replace(r, "")

str2 = str1.split('|')
print(str2)
lays = []
for s in str2:
    lays.append(s.split(' '))

print("\n", lays)

print(lays[0])
print(lays[0][0])
