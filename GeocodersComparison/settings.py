# -*- coding: utf-8 -*-
"""
@author: Cat Chenal
@module: settings.py
Note: The number of geocoding services in this comparison is fixed at 4.
      They can be changed by changing the names in the 'geocs' list.
      Such a change would require:
      . Saving the new API Keys in the .env file (saved alongside settings.py);
      . Creating the approriate variable(s) for the required API key(s) in
        this module, to be retrieved by GeocodersComparison on import;
      . Ensuring the new geocoders are callable via geopy;
      . Amending get_geodata() in case the output cannot be parsed with the
        current patterns.
"""
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from gc4utils import get_geo_file


geocs = ['Nominatim', 'GoogleV3', 'ArcGis', 'AzureMaps']
colors_dict = dict(zip(geocs, ['red', 'green', 'blue', 'cyan']))

query_lst = ['New York City, NY, USA',
             "Cleopatra's needle, Central Park, New York, NY, USA",
             'Bronx county, NY, USA',
             'Kings county, NY, USA',
             'New York county, NY, USA',
             'Queens county, NY, USA',
             'Richmond county, NY, USA',
             'Boston, MA, USA']

DIR_GEO = os.path.join(Path('.'), 'geodata')
DIR_IMG = os.path.join(Path('.'), 'images')
DIR_SHP = os.path.join(Path('.'), 'geodata', 'shapefiles')
DIR_HTML = os.path.join(Path('.'), 'geodata', 'html_frames')


def check_gecoders_data(geocoders, geo_dir):
    ok = True
    for geoc in geocoders:
        out = 'geodata_' + geoc[:3] + '.json'
        try:
            ok = ok and get_geo_file(out, file_check_only=True)
        except Exception():
            print('Exception on:' + out)
            ok = False
            break

    return ok


def load_env():
    """
    Find .env automacically by walking up directories until it's found
    """
    dotenv_path = find_dotenv()

    if dotenv_path == '':
        msg = "Local '.env' file not found:\n \
               Keys for GoogleV3 and AzureMaps APIs are needed for geocoding\n \
               Are needed if no geocoding data is found locally."
        #raise EnvironmentError( msg )
        print(msg)
    else:
        # load up the entries as environment variables
        load_dotenv(dotenv_path)

    return


load_dotenv()

GOOGLE_KEY = os.getenv("GOO_GEO_API_1")
AZURE_KEY = os.getenv("AZ_KEY_1")


local_data_found = True

if (GOOGLE_KEY is None) or (GOOGLE_KEY is None):
    local_data_found = check_gecoders_data(geocs, DIR_GEO)
    if local_data_found:
        print("The geocoding API keys could not be loaded from the\n \
               environment, but local data was found in {}.".format(DIR_GEO))
    else:
        print("The geocoding API keys could not be loaded from the\n \
               environment and no local data was found in {}.\n \
               Please, setup a .env file containing KEY=value pairs\n \
               to fix this problem.".format(DIR_GEO))

        GOOGLE_KEY = ''
        AZURE_KEY = ''
