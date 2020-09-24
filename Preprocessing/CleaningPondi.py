import pandas as pd

df = pd.read_csv('/home/akash/PycharmProjects/Reservoir/Datasets/reservoir_4.csv')
discharge = df['outflow']
new_discharge2 = []
for row in discharge:
    row = str(row)
    if row.find('+') != -1:
        sep_val = row.split(sep='+')
        row = 0
        for val in sep_val:
            row = row + float(val)
    new_discharge2.append(float(row))

df = pd.DataFrame({'date': df['date'], 'level': df['level'], 'storage': df['storage'], 'inflow': df['inflow'], 'outflow': new_discharge2, 'rainfall': df['rainfall']})
df.to_csv(r'/home/akash/PycharmProjects/Reservoir/Datasets/reservoir_5.csv', index=False, encoding='utf-8')
