from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd

driver = webdriver.Chrome('/home/akash/PycharmProjects/Reservoir/chromedriver')
driver.get('http://123.63.203.150/veeranam08.htm')
content = driver.page_source
soup = BeautifulSoup(content, features='html5lib')

# date = []
storage = []
inflow = []
discharge = []

table = soup.find('table')

for row in table.findAll('tr'):
    # date.append(row.find('td', attrs = {'width':'20%'}).p.span.text)
    storage.append(row.find('td', attrs={'width': '24%'}).p.span.text)
    inflow.append(row.find('td', attrs={'width': '23%'}).p.span.text)
    discharge.append(row.find('td', attrs={'width': '31%'}).p.span.text)

driver.get('http://123.63.203.150/veeranam09.htm')
content = driver.page_source
soup = BeautifulSoup(content, features='html5lib')

for table in soup.findAll('table'):
    for row in table.findAll('tr'):
        # date.append(row.find('td', attrs = {'width':'20%'}).p.span.text)
        storage.append(row.find('td', attrs={'width': '24%'}).p.span.text)
        inflow.append(row.find('td', attrs={'width': '23%'}).p.span.text)
        discharge.append(row.find('td', attrs={'width': '31%'}).p.span.text)

for i in range(10, 14):
    st = 'http://123.63.203.150/veeranam{}.htm'.format(i)
    driver.get(st)
    content = driver.page_source
    soup = BeautifulSoup(content, features='html5lib')
    for table in soup.findAll('table'):
        for row in table.findAll('tr'):
            #  date.append(row.find('td', attrs = {'width':'20%'}).p.span.text)
            storage.append(row.find('td', attrs={'width': '24%'}).p.span.text)
            inflow.append(row.find('td', attrs={'width': '23%'}).p.span.text)
            discharge.append(row.find('td', attrs={'width': '31%'}).p.span.text)

for i in range(14, 19):
    st = 'http://123.63.203.150/veeranam {}.htm'.format(i)
    driver.get(st)
    content = driver.page_source
    soup = BeautifulSoup(content, features='html5lib')
    k = 0
    for table in soup.findAll('table'):
        if k == 0:
            k += 1
        else:
            for row in table.findAll('tr'):
                j = 0
                for col in row.findAll('td'):
                    if j == 0:
                        # date.append(col.p.span.text)
                        j += 1
                    elif j == 1:
                        storage.append(col.p.span.text)
                        j += 1
                    elif j == 2:
                        inflow.append(col.p.span.text)
                        j += 1
                    else:
                        discharge.append(col.p.span.text)
                        j = 0

df = pd.DataFrame({'storage': storage, 'inflow': inflow, 'discharge': discharge})
df.to_csv(r'reservoir_1.csv', index=False, encoding='utf-8')