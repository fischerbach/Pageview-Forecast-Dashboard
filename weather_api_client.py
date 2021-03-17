import pandas as pd

from pathlib import Path
import glob
import os

import requests
import json

from cachetools import cached, TTLCache
cache = TTLCache(maxsize=25000, ttl=365*24*60*60)

class WeatherApiClient(object):
    def __init__(self, key='', base='http://api.weatherapi.com/v1'):
        self.key = key
        self.base = f"{base}"

    @cached(cache)
    def history(self, city, dt):
        url = f'{self.base}/history.json?key={self.key}&q={city}&dt={dt}'
        response = requests.get(url)
        return response.json()


def save_day_history(client, city, dt, outdir):
    outfile = os.path.join(outdir, f'{dt}_{city}.json')

    if(outfile not in glob.glob(os.path.join(outdir, '*.json'))):
        data = client.history(city, dt)
        with open(outfile, 'w') as out:
            out.write(json.dumps(data))

weather = WeatherApiClient(key="GETKEY")
cities_data = pd.read_csv('dataset.csv')
cities_data.sort_values(by='day', inplace=True)

outdir = 'weather'
Path(os.path.join(outdir)).mkdir(parents=True, exist_ok=True)

for row in cities_data[['day','city']].values:
    try:
        save_day_history(weather, row[1], row[0], outdir)
    except:
        pass
    print(row)
