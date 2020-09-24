from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_csv('/home/akash/PycharmProjects/Reservoir/Datasets/reservoir_5.csv')

# training

kmeans = KMeans(n_clusters=5)
kmeans.fit(df[['level', 'storage', 'inflow', 'rainfall']])

# label assignment

label = kmeans.predict(df[['level', 'storage', 'inflow', 'rainfall']])

df = pd.DataFrame({'date': df['date'], 'level': df['level'], 'storage': df['storage'], 'inflow': df['inflow'], 'rainfall': df['rainfall'], 'label': label})
df.to_csv(r'/home/akash/PycharmProjects/Reservoir/Datasets/reservoir_6.csv', index=False, encoding='utf-8')
