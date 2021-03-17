import pandas as pd
import os
import json
import glob

files = glob.glob('weather_/*.json')

el = []
for f in files:
    f = json.loads(open(f,'r').read())
    
    day = {
        'city': f['location']['name'],
        'day': f['forecast']['forecastday'][0]['date'],
    }
    features = ['maxtemp_c', 'mintemp_c', 'avgtemp_c', 'maxwind_kph', 'totalprecip_mm', 'avgvis_km', 'avghumidity', 'uv']

    for feature in features:
        day[feature] = f['forecast']['forecastday'][0]['day'][feature]
    day['text'] = f['forecast']['forecastday'][0]['day']['condition']['text']
    day['icon'] = f['forecast']['forecastday'][0]['day']['condition']['icon']

    el.append(day)
pd.DataFrame.from_records(el).to_csv('weather2.csv', index=False)
