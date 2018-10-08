# -*- coding: utf-8 -*-
"""
@author: Cat
@module: gc4utils
Mostly file IO;
"""
import os


def get_file_age(fullfilename):
    """
    To be called after a file existence check.
    Returns a string.
    """
    from datetime import datetime
    if not isinstance(fullfilename, str):
        return ''

    diff = datetime.utcnow()
    try:
        diff -= datetime.utcfromtimestamp(os.stat(fullfilename).st_atime)
    except:
        return ''

    dd = str(diff).split(':')
    info = ': {}, {}h {}m {:.0f}s old'.format( fullfilename,
                                                         dd[0], dd[1],
                                                         float(dd[2]))
    return info


def get_geo_file(geofile, file_check_only=False, show_info=True):
    """Loads a local geo json file data in a dict if:
        1. file_check_only == False;
        2. file has a json extension;
        3. file is found.
       Parameters
       ----------
       :param geofile: The file full name <path, name, extension>.
       :param file_check_only: Output is bool, not data dict.
       :param show_info: Show local file age info.

    """
    d = ''
    if file_check_only:
        return os.path.exists(geofile)
    else:
        if os.path.exists(geofile):
            if show_info:
                print('Found {}'.format(get_file_age(geofile)))

            import pandas as pd

            d = pd.read_json(geofile).to_dict()
            return d
        else:
            return None


def save_file(outname, ext, s, replace=True):
    outfile = outname + '.' + ext

    if replace:
        if os.path.exists(outfile):
            os.remove(outfile)

    if isinstance(s, dict):
        import json

        with open(outfile, 'w') as f:
            f.write(json.dumps(s))
    else:
        if len(s):
            with open(outfile, 'w') as f:
                f.write(s)
    return
