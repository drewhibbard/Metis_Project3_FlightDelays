#!/usr/bin/env python
# coding: utf-8

# In[115]:

'''
This module will gather live flight data, such as airline, originating and departing airports, aircraft type, engine type, weather, etc.  It will use that data to make a probability prediction of the likelihood that flight will be delayed.  It will be deployed in a Streamlit web app.
'''


import sys
from suds import null, WebFault
from suds.client import Client
import logging
import pandas as pd
import numpy as np
import geopy.distance
import pickle
import datetime
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot
import seaborn as sns
import math
sns.set()

np.seterr(divide='ignore', invalid='ignore')


#connect to the Flightaware API
username = 'drewhibbard'
key = 'e3b96bb77a74f8670797a353cdec88fdaee2aa16'
url = 'http://flightxml.flightaware.com/soap/FlightXML2/wsdl'

logging.basicConfig(level=logging.INFO)
api = Client(url, username=username, password=key)


# to convert between IATA and ICAO airport codes
with open('data/airport_code_converter.pickle','rb') as read_file:
    airport_code_converter = pickle.load(read_file)

# to convert between IATA and ICAO airline codes
with open('data/airline_code_converter.pickle','rb') as read_file:
    airline_code_converter = pickle.load(read_file)
    
# to grab latitude and longitude of airports
with open('data/airport_coord_lookup.pickle','rb') as read_file:
    airport_coord_lookup = pickle.load(read_file)

# to get the engines associated with each aircraft
with open('tail_engine_conv.pickle','rb') as read_file:
    tail_engine_convert = pickle.load(read_file)
    
# to convert the engine code to manufacturer name
with open('data/engine_manufacturer_lookup.pickle','rb') as read_file:
    engine_manufacturer_lookup = pickle.load(read_file)
    
# to grab the aircraft age
with open('data/aircraft_age_lookup.pickle','rb') as read_file:
    aircraft_age_lookup = pickle.load(read_file)
    
# to grab the airline name from IATA code
with open('data/airline_name_lookup.pickle','rb') as read_file:
    airline_name_lookup = pickle.load(read_file)
    
# various files to obtain the numerical feature variables from categorical variables

with open('data/aircraft_delay_lookup.pickle','rb') as read_file:
    aircraft_delay_lookup = pickle.load(read_file)
    
with open('data/airline_delay_lookup.pickle','rb') as read_file:
    airline_delay_lookup = pickle.load(read_file)
    
with open('data/airport_delay_lookup.pickle','rb') as read_file:
    airport_delay_lookup = pickle.load(read_file)
    
with open('data/engine_delay_lookup.pickle','rb') as read_file:
    engine_delay_lookup = pickle.load(read_file)
    
with open('data/hour_delay_lookup.pickle','rb') as read_file:
    hour_delay_lookup = pickle.load(read_file)
    
with open('data/model_delay_lookup.pickle','rb') as read_file:
    model_delay_lookup = pickle.load(read_file)
    
with open('data/month_delay_lookup.pickle','rb') as read_file:
    month_delay_lookup = pickle.load(read_file)
    
with open('data/weekday_delay_lookup.pickle','rb') as read_file:
    weekday_delay_lookup = pickle.load(read_file)
    
# coefficients from the Logistic Regression model, for lightning fast predictions
with open('data/model_coef.pickle','rb') as read_file:
    model_coefs = pickle.load(read_file)

# a scaler fit to the features
with open('data/scaler.pickle','rb') as read_file:
    scaler = pickle.load(read_file)

# because I trained on data only from certain airports, I want some sort of prediction to run no matter what
# so if a feature is unavailable for any reason, I will be able to fill in the mean
with open('data/feature_means.pickle','rb') as read_file:
    feature_means = pickle.load(read_file)




def get_flight_info(flight_number):
    '''
    Input: a flight number
    Returns: a dictionary with all information needed for delay prediction, as well as consumer-facing variables 
    such as airport names.
    '''
    
    flight_info = {}
    
    flight_details = api.service.FlightInfoEx(flight_number,1)
    fa_id = flight_details[1][0]['faFlightID']  # FlightAware's unique code needed to grab other info
    flight_info['unique_id'] = fa_id
    
    tail = api.service.AirlineFlightInfo(fa_id)['tailnumber']  # the FAA registration number
    
    flight_info['tail_num'] = tail
    flight_info['aircraft_type'] = flight_details[1][0]['aircrafttype'] # the aircraft model, such as Boeing 737
    airport = airport_code_converter[flight_details[1][0]['origin']]
    flight_info['airport_orig'] = airport
    flight_info['airport_dest'] = airport_code_converter[flight_details[1][0]['destination']]
    flight_info['airline'] = airline_code_converter[flight_details[1][0]['ident'][:3]]
    
    depart_unix_time = flight_details[1][0]['filed_departuretime']
    depart_timestamp = datetime.datetime.fromtimestamp(depart_unix_time)
    
    flight_info['month'] = depart_timestamp.month
    flight_info['hour'] = depart_timestamp.hour
    flight_info['day_of_week'] = depart_timestamp.weekday() + 1
    
    weather = api.service.MetarEx(airport,howMany=1)
    
    flight_info['snow'] = int('snow' in weather['metar'][0]['cloud_friendly'].lower())  # snow as a binary variable
    flight_info['rain'] = int('rain' in weather['metar'][0]['cloud_friendly'].lower())  # rain as a binary variable
    flight_info['wind'] = round(weather['metar'][0]['wind_speed'] * 1.151,1)  # convert from nots to mph 
    flight_info['temp_f'] = round(weather['metar'][0]['temp_air']*(9/5) +32,0)  # convert from celcius to fahrenheit
    flight_info['clouds'] = weather['metar'][0]['cloud_friendly']
    
    # obtain the lat and long coordinates of both airports from my original data, or query the API if unavailable
    try:
        port1_coords = airport_coord_lookup[flight_info['airport_orig']]
        port2_coords = airport_coord_lookup[flight_info['airport_dest']]
        flight_info['distance'] = round(geopy.distance.distance(port1_coords,port2_coords).miles,0)
    except:
        orig = api.service.AirportInfo(flight_details[1][0]['origin'])
        dest = api.service.AirportInfo(flight_details[1][0]['destination'])
        port1_coords = (orig['latitude'],orig['longitude'])
        port2_coords = (dest['latitude'],dest['longitude'])
        flight_info['distance'] = round(geopy.distance.distance(port1_coords,port2_coords).miles,0)
    
    flight_info['orig_city'] = flight_details[1][0]['originCity']
    flight_info['destination_city'] = flight_details[1][0]['destinationCity']
    
    # lookup the engine info based on tail number, or nan if tail number is not in the data source
    try:
        flight_info['engine'] = tail_engine_convert[tail]
        flight_info['engine_manufacturer'] = engine_manufacturer_lookup[flight_info['engine']]
    except:
        flight_info['engine'] = np.nan
        flight_info['engine_manufacturer'] = np.nan
    
    # same thing for the aircraft age
    try: 
        flight_info['year_plane_made'] = aircraft_age_lookup[tail]
        flight_info['aircraft_age'] = 2020 - aircraft_age_lookup[tail]
    except:
        flight_info['year_plane_made'] = np.nan
        flight_info['aircraft_age'] = np.nan
        
    # determine if the previous flight of that exact aircraft was delayed by comparing actual depart time to 
    # scheduled depart time
    
    previous_flight = api.service.FlightInfoEx(flight_number,1)
    if previous_flight[1][0]['actualdeparturetime'] > previous_flight[1][0]['filed_departuretime']:
        flight_info['previous_delay'] = 1
    else:
        flight_info['previous_delay'] = 0
    
    return flight_info




def get_prediction_variables(flight_dict):
    '''
    Input: a dictionary of flight info obtained from get_flight_info function
    Returns: an array of feature variables to run through the predictive model.
    '''
    
    # only the non-binary features
    
    # attempt to grab the relevant metric, but if unavailable for any reason, grab the mean instead
    
    features_no_scale = []
    try: 
        features_no_scale.append(flight_dict['snow'])
    except:
        features_no_scale.append(feature_means['snow_orig'])
        
    try: 
        features_no_scale.append(flight_dict['previous_delay'])
    except:
        features_no_scale.append(feature_means['previous_delay'])
        
    try: 
        features_no_scale.append(flight_dict['rain'])
    except:
        features_no_scale.append(feature_means['precip_orig'])
        
        
    features_to_scale = []
    
    try: 
        features_to_scale.append(flight_dict['distance'])
    except:
        features_to_scale.append(feature_means['distance'])
        
    try: 
        features_to_scale.append(airport_delay_lookup[flight_dict['airport_orig']])
    except:
        features_to_scale.append(feature_means['airport_delayed'])
        
    try: 
        features_to_scale.append(airline_delay_lookup[flight_dict['airline']])
    except:
        features_to_scale.append(feature_means['airline_delayed'])
        
    try: 
        features_to_scale.append(hour_delay_lookup[flight_dict['hour']])
    except:
        features_to_scale.append(feature_means['hour_delayed'])
        
    try: 
        features_to_scale.append(month_delay_lookup[flight_dict['month']])
    except:
        features_to_scale.append(feature_means['month_delayed'])
        
    try: 
        features_to_scale.append(weekday_delay_lookup[flight_dict['day_of_week']])
    except:
        features_to_scale.append(feature_means['weekday_delayed'])
        
    try: 
        features_to_scale.append(model_delay_lookup[flight_dict['aircraft_type']])
    except:
        features_to_scale.append(feature_means['model_delayed'])
        
    try: 
        features_to_scale.append(engine_delay_lookup[flight_dict['engine']])
    except:
        features_to_scale.append(feature_means['engine_delayed'])
        
    try: 
        features_to_scale.append(aircraft_delay_lookup[flight_dict['tail_num']])
    except:
        features_to_scale.append(feature_means['aircraft_delayed'])
        
    try: 
        features_to_scale.append(flight_dict['temp_f'])
    except:
        features_to_scale.append(feature_means['min_temp_orig'])
        
    try: 
        features_to_scale.append(flight_dict['wind'])
    except:
        features_to_scale.append(feature_means['avg_wing_orig'])
    
    arr_no_scale = np.array([features_no_scale])
    arr_to_scale = np.array([features_to_scale])

    scaled = scaler.transform(arr_to_scale)

    
    return np.concatenate([arr_no_scale,scaled],axis=1)



def predict(features):
    exp = sum(features[0] * model_coefs)
    return 1/(1+math.e**(-exp))


st.write('''
# Will Your Flight be Delayed?
''')


flight = st.text_input('Input your flight number, without spaces')

flight_info = get_flight_info(flight)
features = get_prediction_variables(flight_info)
prediction = predict(features)

st.markdown('''*Don't have a flight number but want to test it? Google two cities and an airline*''')
st.markdown('''*Real time data from* [FlightAware](https://flightaware.com/)''')

# if tail number is not available, alert the user

if not flight_info['tail_num']:
    st.write('''## Information on your specific aircraft is unavailable.  This will affect
    the accuracy of the prediction.''')


st.write('## Prediction: ',round(prediction*100,0), '% chance of delay.')

if flight_info['previous_delay']: 
    st.write('''### Your aircraft was delayed earlier today, which significantly increases the odds of your flight also being delayed.''')
else: 
    st.write('''### Your aircraft arrived on time to its most recent flight.  This decreases the odds of your flight being delayed.''')
    


st.sidebar.write('### Flying with: ',airline_name_lookup[flight_info['airline']])
st.write('### ',round(airline_delay_lookup[flight_info['airline']]*100,0),
           '% of flights with this airline are delayed.')

st.sidebar.write('### Origin: ',flight_info['orig_city'])

try:
    st.write('### ',round(airport_delay_lookup[flight_info['airport_orig']]*100,0),
               '% of flights out of this airport are delayed')
except:
    pass

st.sidebar.write('### Destination: ',flight_info['destination_city'])
st.sidebar.write('Total Distance: ',flight_info['distance'],' miles')

st.sidebar.write('### Current Conditions:')
st.sidebar.write(flight_info['temp_f'],' degrees F')
st.sidebar.write(flight_info['clouds'])
st.sidebar.write('Wind: ',flight_info['wind'],' mph')

if flight_info['rain']:
    st.write('### 26% of flights are delayed when raining.')
elif flight_info['snow']:
    st.write('### 38% of flights are delayed when snowing')
elif flight_info['temp_f'] <=32:
    st.write('### De-icing is common at this temperature.  This could increase your chances of delay.')
else:
    st.write("### The weather should not affect your aircraft's arrival.")
    

if flight_info['aircraft_type'][0] == 'B': 
    aircraft_manufacturer = 'Boeing'
    model = flight_info['aircraft_type'][-3:]
elif flight_info['aircraft_type'][0] == 'A': 
    aircraft_manufacturer = 'Airbus'
else: 
    aircraft_manufacturer = ''

st.sidebar.write('### Your aircraft: ',aircraft_manufacturer, flight_info['aircraft_type'])
st.sidebar.write('Built: ', flight_info['year_plane_made'], 'Age: ',flight_info['aircraft_age'], ' years')
st.sidebar.write('Tail number: ',flight_info['tail_num'])
st.sidebar.write('Engines: ',flight_info['engine_manufacturer'])

month_lookup = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
         7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'} 

weekday_lookup = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}

st.sidebar.write('### Time Information:')
st.sidebar.write('Month: ',month_lookup[flight_info['month']])
st.sidebar.write('Day of Week: ', weekday_lookup[flight_info['day_of_week']])
st.sidebar.write('Hour of Day: ',flight_info['hour'])


if flight_info['aircraft_type'][0] == 'B':
    st.write('### Aircraft ', flight_info['aircraft_type'],' is associated with a ',
             round(model_delay_lookup[flight_info['aircraft_type'][-3:]]*100,0),'% delay rate')
else:
    st.write('### Aircraft type ', flight_info['aircraft_type'],' is associated with a ',
             round(model_delay_lookup[flight_info['aircraft_type']]*100,0),'% delay rate')
    
st.write('### Interestingly, aircraft age has no effect on flight delays.  Nor on flight safety.')
st.write('### Engine manufacturer',flight_info['engine_manufacturer'], ' is associated with a ',
        round(engine_delay_lookup[flight_info['engine']]*100,0),' % delay rate.')

st.write('### ', round(month_delay_lookup[flight_info['month']]*100,0), '% of flights during ',
         month_lookup[flight_info['month']], ' are delayed.')
st.write('### ', round(weekday_delay_lookup[flight_info['day_of_week']]*100,0), '% of flights on ',
         weekday_lookup[flight_info['day_of_week']], ' are delayed.')
st.write('### ', round(hour_delay_lookup[flight_info['hour']]*100,0), '% of flights at ',
         flight_info['hour'], ' are delayed.')

st.write('## See how your metrics compare below!')
st.image('images/flight_delay_map.png',use_column_width=True)

st.image('images/by_airline.png',use_column_width=True)

st.image('images/by_time.png',use_column_width=True)

st.image('images/byweather.png',use_column_width=True)

st.image('images/by_type.png',use_column_width=True)

