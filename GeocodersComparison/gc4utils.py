# -*- coding: utf-8 -*-
"""
@author: Cat
@module: gc4utils
Mostly file IO;
"""
__author__ = 'catchenal@gmail.com'

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

    if os.path.exists(geofile):
        if show_info:
            print('Found {}'.format(get_file_age(geofile)))

        import pandas as pd

        return pd.read_json(geofile).to_dict()
    
    else:
        return None


def save_file(fname, ext, s, replace=True):
    # check if fname has an extension:
    try:
        i = fname.index('.' , -6)
        outfile = fname[:i] + '.' + 'json'
    except:
        outfile = fname + '.' + 'json'
    
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


def is_lab_notebook():
        import re
        import psutil
        
        return any(re.search('jupyter-lab-script', x)
                   for x in psutil.Process().parent().cmdline())
                   
def check_notebook():
    """
    Util to check a Jupyter notebook environment:
    a markdown cell in a jupyter lab notebook cannot render
    variables: the cell text needs to be created with
    IPython.display.Markdown.
    """

    if is_lab_notebook():
        # need to use Markdown if referencing variables:
        from IPython.display import Markdown
               
        msg = "This is a <span style=\"color:red;\">JupyterLab notebook \
              </span>: Use `IPython.display.Markdown()` if referencing variables; \
              {{var}} does not work."
        return Markdown('### {}'.format(msg))
    

def format_with_bold(s_format):
    """
    Returns the string with all placeholders preceded by '_b' 
    replaced with a bold indicator value (ANSI escape code).
    
    :param: s_format: a string format; 
            if contains '_b{}b_' this term gets bolded.
    :param: s: a string or value
    
    :note 1: '... _b{}; something {}b_ ...' is a valid format.
    :note 2: IndexError is raised using the returned format only when
            the input tuple length < number of placeholders ({});
            it is silent when the later are greater (see Example).
    :TODO: Do same for _f{}f_: to frame a text.
    
    :Example:
    # No error:
    fmt = 'What! _b{}b_; yes: _b{}b_; no: {}.'
    print(format_with_bold(fmt).format('Cat', 'dog', 3, '@no000'))
    # IndexError:
    print(format_with_bold(fmt).format('Cat', 'dog'))
    """

    # Check for paired markers:
    if s_format.count('_b') != s_format.count('b_'):
        err_msg1 = "Bold indicators not paired. Expected '_b{}b_'."
        raise LookupError(err_msg1)
    
    # Check for start bold marker:
    b1 = '_b'
    i = s_format.find(b1 + '{')
    
    # Check marker order: '_b' past 'b_'?:
    if i > s_format.find('}' + 'b_'):
        err_msg2 = "Starting bold indicator not found. Expected '_b{}b_'."
        raise LookupError(err_msg2)
        
    while i != -1:
        
        # Check for trailing bold marker:
        b2 = 'b_'
        j = s_format.find('}' + b2)
        
        if j != -1:
            s_format = s_format.replace(b1, '\033[1m')
            s_format = s_format.replace(b2, '\033[0m')
        else:
            err_msg3 = "Trailing bold indicator not found. Expected '_b{}b_'."
            raise LookupError(err_msg3)
            
        i = s_format.find(b1 + '{')
    
    return s_format


AS_OF = 'April 2019'
def caveat_codor():
    import sys
    from IPython.display import Markdown

    mysys = '{} | {}<br>As of:  {}'.format(sys.version,
                                           sys.platform,
                                           str(AS_OF))
    msg = "The code and information herein is valid given my "
    msg += "understanding and this environment:<br>"
    return Markdown(msg + mysys)
