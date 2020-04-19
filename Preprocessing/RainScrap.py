from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as uReq, Request
my_url = 'https://weather.com/en-IN/weather/today/l/a6fef8b45e8b8e6a03bb6f146fa86efc8a3c428ceca90b547b8fb91fa612ddfa'
print(my_url)
hdr = {'User-Agent': 'Chrome/80.0'}
req = Request(my_url, headers=hdr)
page = uReq(req)
page_soup = soup(page, "html.parser")

containers = page_soup.findAll("div", {"class": "looking-ahead"})

container = containers[0]
rain = container.findAll("div", {"class": "today-daypart-precip"})


def rainscrap():
    rainList = []

    for val in range(5):
        rainList.append(rain[val].text)

    return rainList


val1 = rainscrap()
print(val1)

# print(soup.prettify(containers[0]))
