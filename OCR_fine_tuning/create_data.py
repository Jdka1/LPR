import pandas as pd


df = pd.read_csv('data/_results.csv', index_col=False)
df = df.drop(['state'], axis=1)

with open('data/_labels.txt', 'w') as f:
    for i in range(df.rows()):
        row = df.iloc[i]
        f.write(f'{row['image_path']} {row['plate_number']}\n')
