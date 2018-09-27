# settings.py

import os
from dotenv import load_dotenv, find_dotenv


def load_env():
    """
    Find .env automacically by walking up directories until it's found
    """
    dotenv_path = find_dotenv()

    if dotenv_path == '':
        msg = "Local '.env' file not found:\n \
               Keys for GoogleV3 and AzureMaps APIs are needed for geocoding.\n \
               To proceed, create an .env file (KEY=value format) and reload."
        raise EnvironmentError( msg )
    else:
        # load up the entries as environment variables
        load_dotenv(dotenv_path)

    return


load_dotenv()

GOOGLE_KEY = os.getenv("GOO_GEO_API_1")
AZURE_KEY = os.getenv("AZ_KEY_1")
