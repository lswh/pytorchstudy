import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil


# title,excerpt,url,file_name,keyword,category
df = pd.DataFrame(columns=['title', 'excerpt', 'url', 'file_name', 'keyword', 'category'])

for filename in os.listdir('.'):
    if not filename.startswith('dataframe_'):
        continue

    idf = pd.read_csv('./' + filename)
    df = pd.concat([df, idf], ignore_index=True)


df = df.drop_duplicates(subset=['url'])

df[['title', 'category']].groupby(['category']).count().plot(kind='bar')
plt.show(block=True)


df.to_csv('text_analysis_data.csv')

# Delete existing cleaned data
try:
    shutil.rmtree('./clean_data')
except FileNotFoundError:
    pass

os.mkdir('./clean_data')

for index, row in df.iterrows():
    shutil.copy('./data/files/' + row['file_name'], './clean_data/' + row['file_name'])
