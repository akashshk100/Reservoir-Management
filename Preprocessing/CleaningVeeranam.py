import pandas as pd

df = pd.read_csv('/home/akash/PycharmProjects/Reservoir/reservoir_1.csv')
discharge = df['discharge']
storage = df['storage']
inflow = df['inflow']
new_discharge = []
new_storage = []
new_inflow = []

for row in discharge:
    if row != 'Discharge in cusec':
        new_discharge.append(row)

new_discharge2 = []
new_storage2 = []
new_inflow2 = []

for row in new_discharge:
    row = str(row)
    if row.find('+') != -1:
        sep_val = row.split(sep='+')
        row = 0
        for val in sep_val:
            row = row + float(val)
    new_discharge2.append(float(row))

for i in range(0, df.__len__()):
    storage_str = df['storage'][i]
    inflow_str = df['inflow'][i]
    if storage_str != 'Storage in Mcft':
        new_storage2.append(float(storage_str))
    if inflow_str != 'inflow in cusec':
        new_inflow2.append(float(inflow_str))

new_discharge3 = []
new_storage3 = []
new_inflow3 = []

for i in range(0, new_discharge2.__len__()):
    if new_storage2[i] != 0.0 or new_inflow2[i] != 0.0 or new_discharge2[i] != 0.0:
        new_discharge3.append(new_discharge2[i])
        new_inflow3.append(new_inflow2[i])
        new_storage3.append(new_storage2[i])

df = pd.DataFrame({'storage': new_storage3, 'inflow': new_inflow3, 'discharge': new_discharge3})
df.to_csv(r'reservoir_2.csv', index=False, encoding='utf-8')
