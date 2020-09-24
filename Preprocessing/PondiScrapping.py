from selenium import webdriver
from csv import writer

driver = webdriver.Chrome('/home/akash/PycharmProjects/Reservoir/chromedriver')
driver.get('http://123.63.203.150/reserve.asp')


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def fetchvalues(date):
    try:
        text_area = driver.find_element_by_xpath('/html/body/form/table/tbody/tr[2]/td/font/input[1]')
        text_area.send_keys(date)
        submit_button = driver.find_element_by_xpath('/html/body/form/table/tbody/tr[2]/td/font/input[2]')
        submit_button.click()
        level = driver.find_element_by_xpath('/html/body/form/center[2]/table/tbody[2]/tr[2]/td[4]/center/font/b').text
        storage = driver.find_element_by_xpath('/html/body/form/center[2]/table/tbody[2]/tr[2]/td[5]/center/b/font/b').text
        inflow = driver.find_element_by_xpath('/html/body/form/center[2]/table/tbody[2]/tr[2]/td[6]/center/font/b').text
        outflow = driver.find_element_by_xpath('/html/body/form/center[2]/table/tbody[2]/tr[2]/td[7]/center/font/b').text
        rainfall = driver.find_element_by_xpath('/html/body/form/center[2]/table/tbody[2]/tr[2]/td[8]/center/font/b').text
        append_list_as_row('../MiscPractice/reservoir5.csv', [date, level, storage, inflow, outflow, rainfall])
    except:
        append_list_as_row('../MiscPractice/reservoir5.csv', [date, 0.0, 0.0, 0.0, 0.0, 0.0])
        driver.get('http://123.63.203.150/reserve.asp')


date = ''
for year in range(2011, 2018):
    for month in range(1, 13):
        if year % 4 == 0 and month == 2:
            for day in range(1, 30):
                if month < 10:
                    if day < 10:
                        date = '0' + str(day) + '/' + '0' + str(month) + '/' + str(year)
                    else:
                        date = str(day) + '/' + '0' + str(month) + '/' + str(year)
                else:
                    if day < 10:
                        date = '0' + str(day) + '/' + str(month) + '/' + str(year)
                    else:
                        date = str(day) + '/' + str(month) + '/' + str(year)
                fetchvalues(date)
        elif month in [1, 3, 5, 7, 8, 10, 12]:
            for day in range(1, 32):
                if month < 10:
                    if day < 10:
                        date = '0' + str(day) + '/' + '0' + str(month) + '/' + str(year)
                    else:
                        date = str(day) + '/' + '0' + str(month) + '/' + str(year)
                else:
                    if day < 10:
                        date = '0' + str(day) + '/' + str(month) + '/' + str(year)
                    else:
                        date = str(day) + '/' + str(month) + '/' + str(year)
                fetchvalues(date)
        elif month in [4, 6, 9, 11]:
            for day in range(1, 31):
                if month < 10:
                    if day < 10:
                        date = '0' + str(day) + '/' + '0' + str(month) + '/' + str(year)
                    else:
                        date = str(day) + '/' + '0' + str(month) + '/' + str(year)
                else:
                    if day < 10:
                        date = '0' + str(day) + '/' + str(month) + '/' + str(year)
                    else:
                        date = str(day) + '/' + str(month) + '/' + str(year)
                fetchvalues(date)
        else:
            for day in range(1, 29):
                if month < 10:
                    if day < 10:
                        date = '0' + str(day) + '/' + '0' + str(month) + '/' + str(year)
                    else:
                        date = str(day) + '/' + '0' + str(month) + '/' + str(year)
                else:
                    if day < 10:
                        date = '0' + str(day) + '/' + str(month) + '/' + str(year)
                    else:
                        date = str(day) + '/' + str(month) + '/' + str(year)
                fetchvalues(date)
