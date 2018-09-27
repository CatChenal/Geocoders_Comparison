# -*- coding: utf-8 -*-
"""
@author: Cat Chenal
"""
import os

import numpy as np
import pandas as pd

from collections import OrderedDict

import requests

from geopy import distance as geod
from geopy.geocoders import Nominatim
from geopy.geocoders import GoogleV3
from geopy.geocoders import ArcGIS
# from geopy.geocoders import AzureMaps
# no longer works; acessing the API w/requests

import folium
from IPython.display import display

import settings
# =============================================================================


def get_geo_file(geofile):
    """Loads a previously stored geojson file if found.
       Returns a dict.
    """
    from datetime import datetime
    d = ''
    if os.path.exists(geofile):
        diff = datetime.utcnow()
        diff -= datetime.utcfromtimestamp(os.stat(geofile).st_atime)
        dd = str(diff).split(':')
        print('Found locale file: {}, {}h {}m {:.0f}s old'.format(geofile,
                                                                  dd[0], dd[1],
                                                                  float(dd[2])))
        d = pd.read_json(geofile).to_dict()
    return d


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


def get_geodata(geocoder_to_use, query_list, use_local=True):
    """
    Wrapper function for using one of four geocoders: 'Nominatim', 'GoogleV3',
    'ArcGis', 'AzureMaps'*, to retrieve the geographical data of places in
    query_list.
    [*] called  via requests after unsolved HTTP 400 error appeared.

    Parameters
    ----------
    :param geocoder_to_use (str): to switch geocoding service
    :param query_list (list): a list of cities, places or counties
    :param use_local (bool), default=True: a local file returned if found

    Returns
    -------
    geodata (odict): odict_keys(['loc', 'box']) where
              loc=['lat', 'lon'] and box=[[NE lat, lon], [SW lat, lon]].
              Its keys are the places queried, not the entire string.

    Example
    -------
    >>> geo_Nom = get_geodata('Nominatim', ['New York City, NY, USA')

    >>> geo_Nom['New York City'].keys()
    odict_keys(['loc', 'box'])
    """

    # init checks:
    if geocoder_to_use not in geocs:
        msg = 'Function setup for these geocoders: {}, not for: {}'
        msg = msg.format(geocs, geocoder_to_use)
        raise Exception(msg)

    if (not isinstance(query_list, list)):
        msg = '"query_list" must be a list. Given is: {}'
        msg = msg.format(type(query_list))
        return TypeError(msg)

    # to check /save local file:
    out = 'geodata_' + geocoder_to_use[:3]

    if use_local:
        geodata = get_geo_file('./geodata/' + out + '.json')
        if isinstance(geodata, dict):
            return geodata
        else:
            use_local = False

    if not use_local:

        tout = 5
        idx = geocs.index(geocoder_to_use)

        if idx == 0:
            g = Nominatim(user_agent='this_app', country_bias='USA',
                          timeout=tout)
        elif idx == 1:
            g = GoogleV3(api_key=GOOGLE_KEY, timeout=tout)
        elif idx == 2:
            g = ArcGIS(username=None, password=None, referer=None,
                       user_agent='this_app', timeout=tout)
        else:
            # original setup stopped working 9/12/19: unable to resolve the
            # http 400 error; reverted to request/json.
            # g = AzureMaps(subscription_key=AZURE_KEY, timeout=tout,
            #              user_agent='ths_app', domain='atlas.microsoft.com')
            url_Azure = 'https://atlas.microsoft.com/search/address/json'

        geodata = OrderedDict()

        for i, q in enumerate(query_list):
            info_d = OrderedDict()

            if 'county' in q:
                place = q.split(' county, ')[0] + ' county'
            else:
                place = q.split(', ')[0]

            if idx == 0:
                location = g.geocode(q, exactly_one=False, addressdetails=True)
            elif idx != 3:
                location = g.geocode(q)
            else:
                params = {'subscription-key': AZURE_KEY,
                          'api-version': 1.0,
                          'query': q,
                          'typeahead': False,
                          'limit': 1}
                r = requests.get(url_Azure, params=params)
                location = r.json()['results']

            if len(location):   # not sure that's a sufficient check...

                if idx == 0:
                    # pt location
                    info_d['loc'] = [float(location[0].raw['lat']),
                                     float(location[0].raw['lon'])]
                    # bounding boxes as 2 corner pts: [NE], [SW]
                    info_d['box'] = [[float(location[0].raw['boundingbox'][1]),
                                      float(location[0].raw['boundingbox'][3])],
                                     [float(location[0].raw['boundingbox'][0]),
                                      float(location[0].raw['boundingbox'][2])]]

                elif idx == 1:
                    info_d['loc'] = [location.raw['geometry']['location']['lat'],
                                     location.raw['geometry']['location']['lng']]
                    info_d['box'] = [[location.raw['geometry']['viewport']['northeast']['lat'],
                                      location.raw['geometry']['viewport']['northeast']['lng']],
                                     [location.raw['geometry']['viewport']['southwest']['lat'],
                                      location.raw['geometry']['viewport']['southwest']['lng']]]

                elif idx == 2:
                    info_d['loc'] = [location.raw['location']['y'],
                                     location.raw['location']['x']]
                    info_d['box'] = [[location.raw['extent']['ymax'],
                                      location.raw['extent']['xmax']],
                                     [location.raw['extent']['ymin'],
                                      location.raw['extent']['xmin']]]

                else:
                    info_d['loc'] = [location[0]['position']['lat'],
                                     location[0]['position']['lon']]
                    info_d['box'] = [[location[0]['viewport']['topLeftPoint']['lat'],
                                      location[0]['viewport']['btmRightPoint']['lon']],
                                     [location[0]['viewport']['btmRightPoint']['lat'],
                                      location[0]['viewport']['topLeftPoint']['lon']] ]

            geodata[place] = info_d

            # save file (overwrite=default)
            save_file('./geodata/' + out, 'json', geodata)

        return geodata


def get_paiwise_names(geocs):
    import itertools

    pair_comps = []
    for comp in itertools.combinations(geocs, 2):
        pair_comps.append('{} v. {}'.format(comp[0], comp[1]))

    return pair_comps


def compare_geocoords(geo_df, dist_units=['km', 'mi']):
    import itertools
    """
    To obtain a pairwise comparison of the geodata from exactly 4 geocoders.

    Parameters
    ----------
    :param geo_df (pandas.DataFrame): Holds the lat, lon, NE and SW data to be
    compared. DataFrame as formatted by :function:get_geodata_df().
    :param dist_units (list) : ilometers (km), miles (mi) or both.

    Returns
    ----------
    df (pandas.DataFrame): Values are the geodesic distance of the paiwise
    differences.
    """
    # input check:
    msg = __name__
    if geo_df.shape[0] != 4:
        msg += ': Expecting 4 rows of geolocation data.\n'
        msg += 'Given: {}'.format(geo_df.shape[0])
        raise Exception(msg)

    len_units = len(dist_units)
    if len_units > 2:
        msg += ': Expecting at most two units of distance, "km" and "mi".'
        msg += 'Given: {}'.format(dist_units)
        raise Exception(msg)

    both_units = (len_units == 2)
    if not both_units:
        if (len_units == 0):
            units = 'km'
        else:
            units = dist_units[0]

    pairwise_comps = get_paiwise_names(geo_df.index.tolist())
    name = geo_df.index.name

    comps_data = dict()

    # point location differences:
    i = 0
    # Found out about geopy.distance.util.pairwise, but I would still need
    # the iteration to populate comps_data, so I did not use it.
    for comp in itertools.combinations(geo_df[['lat, lon']].values, 2):
        d = geod.distance(comp[0], comp[1])

        if both_units:
            comps_data[pairwise_comps[i]] = {'Location_(km)': np.round(d.km, 6),
                                             'Location_(mi)': np.round(d.mi, 6)}
        elif (units=='km'):
            comps_data[pairwise_comps[i]] = {'Location (km)': np.round(d.km, 6)}
        else:
            comps_data[pairwise_comps[i]] = {'Location (mi)': np.round(d.mi, 6)}

        i += 1

    # box corners differences:
    i = 0
    for comp in itertools.combinations(geo_df[['NE', 'SW']].values, 2):
        NE_d = geod.distance(comp[0][0], comp[1][0])
        SW_d = geod.distance(comp[0][1], comp[1][1])

        if both_units:
            comps_data[pairwise_comps[i]].update({'NE_(km)': np.round(NE_d.km, 6),
                                                  'NE_(mi)': np.round(NE_d.mi, 6),
                                                  'SW_(km)': np.round(SW_d.km, 6),
                                                  'SW_(mi)': np.round(SW_d.mi, 6)})
        elif (units=='km'):
            comps_data[pairwise_comps[i]].update({'NE (km)': np.round(NE_d.km, 6),
                                                  'SW (km)': np.round(SW_d.km, 6)})
        else:
            comps_data[pairwise_comps[i]].update({'NE (mi)': np.round(NE_d.mi, 6),
                                                  'SW (mi)': np.round(SW_d.mi, 6)})

        i += 1

    df = pd.DataFrame(comps_data).T
    df.index.set_names(name, inplace=True)
    if both_units:
        df.columns = df.columns.str.split('_', expand=True)

    return df


def get_geodata_df(geocs, geo_dict, place):
    data = [[gd[place]['loc'],
             gd[place]['box'][0],
             gd[place]['box'][1]] for _, gd in enumerate(geo_dict)]
    df = pd.DataFrame(data, index=geocs, columns=['lat, lon', 'NE', 'SW'])
    df.index.set_names(place, inplace=True)

    return df


def compare_two_geoboxes(place1, place2, geocs, geo_dict):
    """
    Return a pandas.DataFrame with the np.allclose() results for the bounding
    boxes of place1 and place2 for each geocoder.

    Usage
    -----
    To answer the question: "How does each geocoder treat the bounding boxes of
    these two locations, identically?". This is a quantitative check usefull to
    run when two boxes visualized on a map appear to have the same coordinates.

    Parameters
    ----------
    :param place1, place2 (str): Places to compare.
    :param geocs (list): Geocoders names.
    :param geo_dicts (list): Geocoding data dictionaries.

    Return
    ------
    df (pandas.DataFrame): Geocoder name, bool(Identical box?)
    """
    data = {}
    for i, g in enumerate(geo_dict):
        data[geocs[i]] = np.allclose(g[place1]['box'], g[place2]['box'])

    df = pd.DataFrame(pd.Series(data), columns=['Identical_bounding boxes?'])
    col0 = '{} & {}:'.format(place1, place2)
    df.columns = df.columns.str.split('_', expand=True)
    df.index.name = col0

    return df


def compare_location_with_geobox(places, geocs, geo_dicts, show_values=False):
    """
    Usage
    -----
    To answer the question: "Are the point coordinates for a place identical to
    the center of its bounding box?" for all geocoders in the comparison.
    Note: the earth is plat here, no geodesic distance.
    """
    df_lst = []

    for i, g in enumerate(geo_dicts):
        geo = geocs[i]
        data_p = OrderedDict()

        for p, v in g.items():
            # p: places, v: geoms
            box_ctr = [(v['box'][0][0] + v['box'][1][0])/2,
                       (v['box'][0][1] + v['box'][1][1])/2]
            boo = np.allclose(np.round(v['loc'], 6),
                              np.round(box_ctr, 6),
                              atol=1e-06)

            if show_values:
                data_p[p] = {geo+'_location': np.round(v['loc'], 6),
                             geo+'_box center': np.round(box_ctr, 6),
                             geo+'_same?': boo}
            else:
                data_p[p] = {geo: boo}

        df_lst.append(pd.DataFrame(data_p).T)

    df = pd.concat(df_lst, axis=1, sort=False)

    if show_values:
        df.columns = df.columns.str.split('_', expand=True)
    else:
        df.index = places

    df.index.name = 'Location is box center?'

    return df


def get_map(geo, zoom=14, map_style='Stamen Toner'):
    g = np.array(geo)
    mean_lat = g[..., 0].mean()
    mean_lon = g[..., 1].mean()

    m = folium.Map([mean_lat, mean_lon], tiles=map_style, zoom_start=zoom)
    return m


def add_box_and_markers(mapobj, gdf, colors_d):
    """
    Add location markers and a feature group, which provides an "interactive
    legend",
    """
    place = r'{}'.format(gdf.index.name.replace("'", "\\'"))  # for apostrophes

    for i, row in gdf.iterrows():
        # i is a string index
        grp1_name = '<span style=\\"color:' + colors_d[i] + ';\\">' + i
        grp1_name += '</span>'
        grp1 = folium.FeatureGroup(grp1_name)

        tip = "{}, {}: {}".format(i, place, row['lat, lon'])

        folium.CircleMarker([row['lat, lon'][0], row['lat, lon'][1]],
                            radius=10,
                            color='blue',
                            weight=1,
                            fill=True,
                            fill_color=colors_d[i],
                            fill_opacity=0.3,
                            tooltip=tip).add_to(grp1)

        grp1.add_to(mapobj)

        grp2_name = '<span style=\\"color:' + colors_d[i] + ';\\">' + i
        grp2_name += '</span>'
        grp2 = folium.FeatureGroup(grp2_name)

        p1 = row.NE
        p3 = row.SW
        p2 = [p3[0], p1[1]]  # SE
        p4 = [p1[0], p3[1]]  # NW

        # 4 lines to a box:
        folium.PolyLine(locations=[p1, p2], color=colors_d[i]).add_to(grp2)
        folium.PolyLine(locations=[p2, p3], color=colors_d[i]).add_to(grp2)
        folium.PolyLine(locations=[p3, p4], color=colors_d[i]).add_to(grp2)
        folium.PolyLine(locations=[p4, p1], color=colors_d[i]).add_to(grp2)

        grp2.add_to(mapobj)

    folium.map.LayerControl('topright', collapsed=False).add_to(mapobj)

    return mapobj


def add_markers(mapobj, gdf, colors_d):
    """
    Add location markers and a feature group, which provides an "interactive
    legend",
    """
    place = r'{}'.format(gdf.index.name.replace("'", "\\'"))  # for apostrophes

    for i, row in gdf.iterrows():
        # i is a string index
        grp1_name = '<span style=\\"color:' + colors_d[i] + ';\\">' + i
        grp1_name += '</span>'
        grp1 = folium.FeatureGroup(grp1_name)

        tip = "{}, {}: {}".format(i, place, row['lat, lon'])

        folium.CircleMarker([row['lat, lon'][0], row['lat, lon'][1]],
                            radius=10,
                            color='blue',
                            weight=1,
                            fill=True,
                            fill_color=colors_d[i],
                            fill_opacity=0.3,
                            tooltip=tip).add_to(grp1)

        grp1.add_to(mapobj)

    folium.map.LayerControl('topright', collapsed=False).add_to(mapobj)

    return mapobj


def add_boxes(mapobj, gdf, colors_d):
    """
    Draw bounding box on map
    """
    for i, row in gdf.iterrows():
        p1 = row.NE
        p3 = row.SW
        p2 = [p3[0], p1[1]]  # SE
        p4 = [p1[0], p3[1]]  # NW

        grp_name = '<span style=\\"color:' + colors_d[i] + ';\\">' + i
        grp_name += '</span>'
        grp = folium.FeatureGroup(grp_name)

        # 4 lines to a box:
        folium.PolyLine(locations=[p1, p2], color=colors_d[i]).add_to(grp)
        folium.PolyLine(locations=[p2, p3], color=colors_d[i]).add_to(grp)
        folium.PolyLine(locations=[p3, p4], color=colors_d[i]).add_to(grp)
        folium.PolyLine(locations=[p4, p1], color=colors_d[i]).add_to(grp)

        grp.add_to(mapobj)

    folium.map.LayerControl('topright', collapsed=False).add_to(mapobj)

    return mapobj


def map_geos(df, colors_dict, geo_type='location', zoom=14, **kwargs):
    """
    **kwargs: allows for map_style = '<new style>' to change the default
              'Stamen Toner' default map_style.
    """
    if df is None:
        msg = 'map_geos():D ataFrame not set: to populate df,\n'
        msg += 'run show_data(geocs, geo_dicts, place) for a specific place.'
        raise TypeError( msg)

    locs = df['lat, lon'].tolist()

    if 'map_style' in kwargs:
        map_x = get_map(locs, zoom=zoom, map_style=kwargs.get('map_style'))
    else:
        map_x = get_map(locs, zoom=zoom)

    if geo_type == 'location':
        add_markers(map_x, df, colors_dict)
    else:
        add_boxes(map_x, df, colors_dict)

    name = 'map'
    if not (df.index.name is None):
        name = df.index.name.replace(' ', '_')
    name += '_' + geo_type + '.html'
    map_x.save('./html/' + name)

    return map_x


# Dataframe styling functions:
# These are not applied by the function that creates the df
# because the final object would be of type "Styler" and no longer open to
# DataFrame operations.
#
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: lightpink' if v else '' for v in is_max]


def highlight_min(s):
    is_min = s == s.min()
    return ['background-color: palegreen' if v else '' for v in is_min]


def center_hdr(df):
    # also centers the index column???
    align_hdr = [dict(selector="th", props=[('text-align', 'center'),
                                            ('background-color', '#f7f7f9')])]
    df = df.style.set_table_styles(align_hdr)
    return df


def with_style(df):
    cap = 'Four geocoders coordinates pairwise difference comparison'
    align_hdr_left = [dict(selector="th",
                           props=[('text-align', 'left'),
                                  ('background-color', '#f7f7f9')])]

    df = df.style.set_table_styles(align_hdr_left)\
                 .apply(highlight_min)\
                 .apply(highlight_max)\
                 .format("{:.6f}")\
                 .set_caption(cap)

    return df


def show_data(geocs, geo_dicts, place, show=True):

    # geocoder results:
    df = get_geodata_df(geocs, geo_dicts, place)
    if show:
        display(center_hdr(df))

    # Differences as distances:
    dist_df = compare_geocoords(df)
    if show:
        display(with_style(dist_df))

    return df, dist_df


def get_geo_dist_heatmap(places, unit='km', save_fig=True):
    """To show the paiwise geodistance comparison in 3 heatmaps for
       Lcation, NE corner, SW corner.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("notebook", font_scale=1., rc={"lines.linewidth": 1.})

    dist_frames_d = {}

    for p in places:
        # Get the pairwise distance dataframe, second output of show_data():
        dist_frames_d[p] = show_data(geocs, geo_dicts, p, show=False)[1]

    combined_df = pd.DataFrame()

    for k, df in dist_frames_d.items():
        drop_unit = '(mi)'
        if unit == 'mi':
            drop_unit = '(km)'

        new = df.T.unstack(level=0).drop(drop_unit, axis=0)
        new.columns.set_names(['geocoders', 'geom'], inplace=True)
        new.index = [k]
        combined_df = pd.concat([combined_df, new])

    combined_df = combined_df.T
    combined_df.reset_index(inplace=True)

    # for sorting by geom location, NE, SW
    combined_df['sort_geo'] = ([g.split('v. ')[1]
                               for g in combined_df['geocoders']])

    combined_df.sort_values(by=['geom', 'sort_geo'], inplace=True)
    combined_df.set_index(combined_df.geocoders + '|' + combined_df.geom,
                          inplace=True)

    names = combined_df.geocoders.unique()

    combined_df.drop(['geocoders', 'geom', 'sort_geo'], axis=1, inplace=True)

    # Split into 3 to show corresponding heatmaps:
    locs_df = combined_df.loc[combined_df.index.str.endswith('Location')]
    NE_df = combined_df.loc[combined_df.index.str.endswith('NE')]
    SW_df = combined_df.loc[combined_df.index.str.endswith('SW')]

    # Plot heamap with seaborn:
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 8), sharey=True)

    dfs = {0: locs_df, 1: NE_df, 2: SW_df}
    which_type = {0: 'Location', 1: 'NE corner', 2: 'SW corner'}

    # To center the color map:
    my_max_acceptable_difference = 5  # km
    if unit == 'mi':
        my_max_acceptable_difference = my_max_acceptable_difference * 0.62

    for i, a in enumerate(ax):

        df = dfs[i]

        sns.heatmap(df, ax=a,
                    annot=True, fmt=".2f",
                    linewidths=0.5,
                    cmap='coolwarm',
                    center=my_max_acceptable_difference,
                    square=True,
                    cbar=False)

        a.set_yticklabels(names, fontweight='bold', fontsize=12)
        a.set_title('{}\n'.format(which_type[i]),
                    fontweight='bold', fontsize=14)
        a.set_ylabel('')
        if i == 1:
            a.annotate(s='Geodesic distance difference ({}):'.format(unit),
                       xy=[0.5, 1.2], xycoords='axes fraction',
                       ha="center", fontweight='bold', fontsize=14)

    fig.tight_layout()

    # if not save, show:
    if save_fig:
        plt.savefig('./images/Heatmap_sns_geodist_difference_' + unit + '.png',
                    format='png',
                    orientation='landscape',
                    bbox_inches='tight')
        return
    else:
        plt.show()

    # return

def df_to_pic(data, col_width=3., row_height=0.625, font_size=14,
                     header_color='w', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, tbl_name='', wide_col1=False, **kwargs):
    """
    Adapted from S.O. #39358752
    """

    # header_color='#40466e'
    import six          # iteritems

    if ax is None:
        W, H = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height]);
        fig, ax = plt.subplots(figsize=(W, H)); #size);
        ax.axis('off');
        #ax.spines['right'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        plt.xticks([]);
        plt.yticks([])

        if wide_col1:
            # first col has no label:
            cols = list(cat_count_T.columns)
            cols[0]='          '
            mpl_table = ax.table( cellText=data.values,
                                  colLabels=cols,
                                  bbox=bbox, **kwargs );
        else:
            mpl_table = ax.table( cellText=data.values, colLabels=data.columns, bbox=bbox, **kwargs)

    if wide_col1:
        def_ratio = W/len(data.columns)
        added = 0.05
        offset = added/(len(data.columns)-1)
        # width of cells. to accommodate long name in descriptive 1st col
        w_0 = W/(len(data.columns)-1)  # col 1
        w_n = round((W - w_0)/(len(data.columns)-1), 2)
    else:
        w_0 = col_width
        w_n = col_width

    mpl_table.auto_set_font_size(False);
    mpl_table.set_fontsize(font_size);

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color);
        if wide_col1:
            if k[1] > 0:
                cell.set_width(w_n)
            else:
                cell.set_width(w_0)
            #or k[1]
        if k[0] == 0 or k[1] < header_columns:
            cell.set_edgecolor(edge_color);
            cell.set_text_props(weight='bold', color='k');
        else:
            cell.set_edgecolor(None);
            if k[1] == 0:
                cell.set_text_props(weight='bold', color='k');

            cell.set_facecolor(row_colors[k[0]%len(row_colors)-1 ]);

    if len(tbl_name):
        plt.savefig(tbl_name+'.png', format='png',
                    transparent=True, bbox_inches='tight',  pad_inches=0.125)
        #dpi=300,

    return ax;

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#        The following code will be executed on import:

# Call to load API keys from environment file if found:
print('\nFetching API keys from environment file if found.\n')
settings.load_env()
GOOGLE_KEY = settings.GOOGLE_KEY
AZURE_KEY = settings.AZURE_KEY

if (len(GOOGLE_KEY) & len(AZURE_KEY)):
    print('GOOGLE_KEY & AZURE_KEY successfully assigned from envir. file.\n')


# Load the geocoder list in the namespace:
geocs = ['Nominatim', 'GoogleV3', 'ArcGis', 'AzureMaps']
colors_dict = dict(zip(geocs, ['red', 'green', 'blue', 'cyan']))
print('Geocoders in use, var geocs:', geocs)

query_lst = ['New York City, NY, USA',
             "Cleopatra's needle, Central Park, New York, NY, USA",
             'Bronx county, NY, USA',
             'Kings county, NY, USA',
             'New York county, NY, USA',
             'Queens county, NY, USA',
             'Richmond county, NY, USA',
             'Boston, MA, USA']
print('\nPlaces queried, var query_lst:', query_lst)

print('\nFetching geodata...')
geo_Nom = get_geodata(geocs[0], query_lst)
geo_Goo = get_geodata(geocs[1], query_lst)
geo_Arc = get_geodata(geocs[2], query_lst)
geo_Azu = get_geodata(geocs[3], query_lst)


geo_dicts = [geo_Nom, geo_Goo, geo_Arc, geo_Azu]
print('\nAll geodata variables gathered into list geo_dicts.\n')

places = list(geo_dicts[0].keys())
msg = 'The var places will be used for retrieving the geodata '
msg += 'and the distance comparison for a particular place:\n'
print(msg, places)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<