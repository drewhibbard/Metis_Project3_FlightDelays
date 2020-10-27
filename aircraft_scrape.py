'''
Module for scraping airfleets.net for information about specific aircraft.
'''

import requests
from bs4 import BeautifulSoup
import re
import time
import numpy as np
import pandas as pd
import pickle
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

user_agent = {'User-agent': 'Mozilla/5.0'}

def get_links(aircraft_list,filename='links.pickle'):
    '''
    Function to obtain the links to each FAA-registered aircraft.  Will get every N-number aircraft
    the input aircraft model production list.

    Input: Base aircraft model.  Ex: A330 or b737

    Returns: links to all FAA-registered aircraft for that model.  For use in scraping information
    about each aircraft.
    '''
    final = []

    for aircraft in aircraft_list:
        print(aircraft)
        try:
            base = f'https://www.airfleets.net/listing/{aircraft}-1.htm'
            response = requests.get(base,headers=user_agent)
            first_soup = BeautifulSoup(response.content)
            n_pages = int(first_soup.find(class_='page2').text.split('/')[1].strip())
        except:
            continue

        for i in range(1,n_pages+1):
            print(i)
            url = f'https://www.airfleets.net/listing/{aircraft}-{i}.htm'
            resp = requests.get(url,headers=user_agent)
            soup = BeautifulSoup(resp.content)

            table = soup.find('table',class_='tab800')
            rows = [row for row in table.find_all('tr')]
            
            for row in rows[1:]:
                items = row.find_all('td')
                final += [item.find('a').get('href').strip('..') for item in items[5::6] if item.find('a').text.startswith('N')]

            time.sleep(1+np.random.uniform())

            if i%5==0:
                with open(filename,'wb') as to_write:
                    pickle.dump(final,to_write)

        with open(filename,'wb') as to_write:
            pickle.dump(final,to_write)
    

def get_tail_num(soup):
    return soup.find('h1').text.split(' - ')[-1].split(' ')[0]

def get_aircraft_manufacturer(soup):
    return soup.find('h1').text.split(' - ')[0].split()[0]

def get_aircraft_model(soup):
    return soup.find('h1').text.split(' - ')[0].split()[1]

def get_first_flight(soup):
    table = soup.find('table',style='BORDER-COLLAPSE: collapse')
    return table.find(text=re.compile('First flight')).findNext().text

def get_engine(soup):
    return soup.find(text=re.compile('Engines')).replace('\t','').strip().split('Engines')[-1]


def load_page(url):
    chromedriver = "/Applications/chromedriver"
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--incognito")

    driver = webdriver.Chrome(chromedriver,chrome_options=chrome_options)
    driver.get(url)
    time.sleep(3)
    page = driver.page_source
    driver.quit()
    return page

agents = pd.read_csv('user-agents.csv')
agents = agents.user_agent.to_list()


def get_aircraft_info(links,filename='aircraft_info.pickle'):
    '''
    Input: iterable of links obtained from function get_links.

    Returns: a list of dictionaries, with each dictionary containing the following info on each aircraft:
        tail number
        manufacturer
        model
        engine
        date of first flight
    '''

    final = []
    counter = 1
    agent = agents[0]

    for link in links:
        if counter%30==0:
            agent = agents[counter/30]
        base = 'https://www.airfleets.net'
        url = base + link
        response = requests.get(url,headers={'User-agent': agent})
        soup = BeautifulSoup(response.content)

        headers = ['tail_num','aircraft_manufacturer','aircraft_model','engine','first_flight']

        d = dict(zip(headers,[get_tail_num(soup),
        get_aircraft_manufacturer(soup),
        get_aircraft_model(soup),
        get_engine(soup),
        get_first_flight(soup)]))

        final.append(d)
        with open(filename,'wb') as to_write:
            pickle.dump(final,to_write)

        print(counter)
        counter +=1

    