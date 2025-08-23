from skyfield.api import Topos, load
from datetime import date, timedelta
from skyfield import almanac

import json
import os

os.chdir('../')
try:
    os.mkdir('data')
except OSError as error:
    pass
path = os.path.join(os.getcwd(), 'data')
os.chdir(path)


def generate(start_year=450,
             start_month=1,
             start_day=1,
             end_year=550,
             end_month=1,
             end_day=1):

    skyfield_data = load('de422.bsp')
    astronomical_bodies = ['sun', 'moon', 'mercury', 'mars', 'venus', 'jupiter', 'saturn']
    earth = skyfield_data["earth"]

    test_loc = earth + Topos("25.61 N", "85.16 E")

    ts = load.timescale()
    start_date = date(start_year, start_month, start_day)
    end_date = date(end_year, end_month, end_day)

    time_diff = (end_date - start_date).days

    processed_data = {
        'time': {
            'utc': [],
            'ord': []},
        'astronomical': {i: {
            'ecliptic_lon': [], 'distance': [],
            'phase_portion': [], 'arc_min': []} if i == 'moon' else {
            'ecliptic_lon': [], 'distance': [], 'arc_min': []} if i == 'sun' else {
            'ecliptic_lon': [], 'distance': []} for i in astronomical}}

    for n in range(time_diff):
        t = start_date + timedelta(n)
        time = ts.utc(t.year, t.month, t.day)

        processed_data['time']['utc'].append(str(t))
        processed_data['time']['ord'].append(t.toordinal())

        phase = almanac.fraction_illuminated(skyfield_data, 'moon', time)
        processed_data['astronomical']['moon']['phase_portion'].append(phase)

        for a in astronomical:
            if a == 'sun' or a == 'moon':
                planet = skyfield_data[a]
            else:
                planet = skyfield_data[a + ' barycenter']
            observe = test_loc.at(time).observe(planet).apparent()
            lat, lon, dist = observe.ecliptic_latlon()

            astro = processed_data['astro'][a]
            if a == 'sun' or a == 'moon':
                astro['arc_min'].append(lon.arcminutes())
            astro['ecliptic_lon'].append(lon.radians)
            astro['distance'].append(dist.au)

    try:
        with open('processed_data.json', 'w') as file:
            json.dump(processed_data, file, indent=3)
    except IOError:
        print("I/O error")


