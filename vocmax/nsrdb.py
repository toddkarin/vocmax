
import numpy as np
import pandas as pd
import glob
import os
import webbrowser
import time

# import sys
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import pytz

def make_lat_long_grid(lat_lims=[-124,-66], lon_lims=[25, 47], lat_step=1, lon_step=1 ):
    """
    Make a lat/long grid pairs for the coordinates specified. Note that the
    end limit point is typically not included in the resultant grid.

    Example
        Make a latitude longitude grid:

         > make_lat_long_grid(lat_lims=[-124,-66], lon_lims=[25, 47], lat_step=1, lon_step=1 )





    """

    lat_flat = np.arange( np.min(lat_lims), np.max(lat_lims), lat_step)
    lon_flat = np.arange( np.min(lon_lims), np.max(lon_lims), lon_step)


    lat = np.zeros(len(lat_flat)*len(lon_flat))
    lon = np.zeros(len(lat_flat) * len(lon_flat))
    n=0
    for j in range(len(lat_flat)):
        for k in range(len(lon_flat)):
            lat[n], lon[n] = lat_flat[j], lon_flat[k]
            n=n+1



    return {'lat':lat, 'lon':lon, 'num':len(lat)}




def inspect_database(root_path):
    """Build database for NSRDB files

    Build a lat/long and year list for NSRDB csv files in a data folder.
    Folders are searched recursively (folders within folders are okay). This
    is a fast way to inspect a set of data files and build a database of file
    path, latitude, longitude and year.

    File names must be of the form 'locationid_lat_lon_year.csv'. For
    example, '14189_18.81_-155.94_2000.csv'.

    Examples
    --------
    inspect_database('data_folder')


    Parameters
    ----------
    root_path

    Returns
    -------
    filedata
        pandas DataFrame containing information on files in the root_path..

    """

    import fnmatch
    import os

    # root_path = 'around_fairfield'
    pattern = '*.csv'


    filedata = pd.DataFrame(columns=['lat','lon','year','filepath'])
    filename_list = []
    filename_fullpath = []
    location_id = []
    lat = []
    lon = []
    year = []

    # Cycle through files in directory, extract info from filename without opening file.
    # Note this would break if NREL changed their naming scheme.
    for root, dirs, files in os.walk(root_path):
        for filename in fnmatch.filter(files, pattern):

            temp = filename.split('_')

            filename_list.append(filename)
            filename_fullpath.append(os.path.join(root, filename))
            location_id.append(int(temp[0]))
            lat.append(float(temp[1]))
            lon.append(float(temp[2]))
            year.append(int(temp[3][0:-4]))

    # Create a DataFrame
    filedata = pd.DataFrame.from_dict({
        'location_id': location_id,
        'lat': lat,
        'lon': lon,
        'year': year,
        'filename': filename_list,
        'fullpath': filename_fullpath})


    filedata = filedata.sort_values(by='location_id')

    # Redefine the index.
    filedata.index = range(filedata.__len__())
    return filedata




def inspect_compressed_database(glob_str):
    """
    Build filename list from directory.

    Examples

    glob_str = '/Users/toddkarin/Documents/NSRDB_compressed/*'
    filedata = nsrdbtools.inspect_compressed_database(glob_str)


    Returns
    -------

    """


    location_id = []
    lat = []
    lon = []


    # filename = get_s3_files()
    # base_dir = '/Users/toddkarin/Documents/NSRDB_compressed/*'
    filename = glob.glob(glob_str)


    # Extract location id, lat and lon.
    for key in filename:
        if key.endswith('.npz'):

            path_parts = os.path.split(key)

            filename_parts = path_parts[-1].split('_')

            location_id.append(int(filename_parts[0]))
            lat.append(float(filename_parts[1]))
            lon.append(float(filename_parts[2][0:-4]))


    # Create a DataFrame
    filedata = pd.DataFrame.from_dict({
        'location_id': location_id,
        'lat': lat,
        'lon': lon,
        'filename': filename,
    })

    # Redefine the index.
    filedata.index = range(filedata.__len__())


    return filedata





def inspect_pickle_database(root_path):
    """Build database for NSRDB files

    Build a lat/long and year list for NSRDB pickled data.

    Examples
    --------
    inspect_pickle_database('data_folder')


    Parameters
    ----------
    root_path

    Returns
    -------
    filedata
        pandas DataFrame containing information on files in the root_path..

    """

    import fnmatch
    import os

    # root_path = 'around_fairfield'
    pattern = '*weather.pkl'


    # filedata = pd.DataFrame(columns=['lat','lon','type','filepath'])
    weather_filename = []
    weather_fullpath = []
    info_filename = []
    info_fullpath = []
    location_id = []
    lat = []
    lon = []
    type = []

    # Cycle through files in directory, extract info from filename without opening file.
    # Note this would break if NREL changed their naming scheme.
    for root, dirs, files in os.walk(root_path):
        for filename in fnmatch.filter(files, pattern):

            temp = filename.split('_')

            weather_filename.append(filename)
            weather_fullpath.append(os.path.join(root, filename))
            location_id.append(int(temp[0]))
            lat.append(float(temp[1]))
            lon.append(float(temp[2]))
            type.append(temp[3][0:-4])

            info_filename.append(filename[0:-11] + 'info.pkl')
            info_fullpath.append(os.path.join(root, filename)[0:-11] + 'info.pkl')

    # Create a DataFrame
    filedata = pd.DataFrame.from_dict({
        'location_id': location_id,
        'lat': lat,
        'lon': lon,
        'type': type,
        'weather_filename': weather_filename,
        'weather_fullpath': weather_fullpath,
        'info_filename': info_filename,
        'info_fullpath': info_fullpath,
    })


    filedata = filedata.sort_values(by='location_id')

    # Redefine the index.
    filedata.index = range(filedata.__len__())
    return filedata



def import_csv(filename):
    """Import an NSRDB csv file.

    The function (df,info) = import_csv(filename) imports an NSRDB formatted
    csv file

    Parameters
    ----------
    filename

    Returns
    -------
    df
        pandas dataframe of data
    info
        pandas dataframe of header data.
    """

    # filename = '1ad06643cad4eeb947f3de02e9a0d6d7/128364_38.29_-122.14_1998.csv'

    info_df = pd.read_csv(filename, nrows=1)
    info = {}
    for p in info_df:
        info[p] = info_df[p].iloc[0]

    # See metadata for specified properties, e.g., timezone and elevation
    # timezone, elevation = info['Local Time Zone'], info['Elevation']

    # Return all but first 2 lines of csv to get data:
    df = pd.read_csv(filename, skiprows=2)

    # Set the time index in the pandas dataframe:
    year=str(df['Year'][0])


    if np.diff(df[0:2].Minute) == 30:
        interval = '30'
        info['interval_in_hours']= 0.5
        df = df.set_index(
          pd.date_range('1/1/{yr}'.format(yr=year), freq=interval + 'Min',
                        periods=60*24*365 / int(interval)))
    elif df['Minute'][1] - df['Minute'][0]==0:
        interval = '60'
        info['interval_in_hours'] = 1
        df = df.set_index(
            pd.date_range('1/1/{yr}'.format(yr=year), freq=interval + 'Min',
                          periods=60*24*365 / int(interval)))
    else:
        print('Interval not understood!')

    df.index = df.index.tz_localize(
        pytz.FixedOffset(float(info['Time Zone'] * 60)))

    return (df, info)

# df, info = import_csv('nsrdb_1degree_uv/104_30.97_-83.22_tmy.csv')

def import_sequence(folder):
    """Import and append NSRDB files in a folder

    Import a sequence of NSRDB files, data is appended to a pandas dataframe.
    This is useful for importing all years of data from one folder.

    Parameters
    ----------
    folder
        directory containing files to import.

    Returns
    -------
    df
        pandas dataframe of data
    info
        pandas dataframe of header data for last file imported.
    """

    # Get all files.
    files = glob.glob(os.path.join(folder, '*.csv'))

    if len(files)==0:
        raise ValueError('No input files found in directory')
    files.sort()
    df = pd.DataFrame()
    for f in files:
        print(f)
        (df_temp,info) = import_csv(f)

        df = df.append(df_temp)

    info['timedelta_in_years'] = (df.index[-1] - df.index[0]).days/365

    return (df,info)

def combine_csv(files):
    """

    Combine multiple files into one dataframe. Note files must be in time
    sequential order.


    :param files:
    :return:
    """

    df = pd.DataFrame()
    for f in files:
        df_temp, info = import_csv(f)

        df = df.append(df_temp)

    info['timedelta_in_years'] = (df.index[-1] - df.index[0]).days / 365

    return (df, info)


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches


def build_nsrdb_link_list(filename):
    """

    Example
        url_list = build_nsrdb_link_list('link_list.txt')

    see also: download_nsrdb_link_list

    Parameters
    ----------
    filename
        text file containing file list to import. can be "copy/pasted" rough
        from gmail.

    Returns
    -------
    url_list
        List of url's to open


    """

    # filename = 'link_list.txt'
    with open(filename, 'r') as content_file:
        content = content_file.read()

    content.replace('\n','')
    url_start = list(find_all(content,'https://maps.nrel.gov/api/'))
    url_end = list(find_all(content,'.zip'))

    url_list = [None] * len(url_start)
    for j in range(len(url_list)):
        url_list[j] = content[url_start[j]:url_end[j]] + '.zip'





    return url_list



def download_nsrdb_link_list(url_list, sleep=0.2):
    """
    This simple script opens a list of urls for downloading files.

    Example:
        downlaod_nsrdb_link_list(url_list)

    Parameters
    ----------
    url_list
        list of urls to open.
    sleep
        Wait time between opening each url
    """
    for j in range(len(url_list)):
        webbrowser.open(url_list[j])
        time.sleep(sleep)




def load_npz(filename):
    """
    Load npz file from a local file

    Parameters
    ----------
    filename

    Returns
    -------

    """
    #
    data = {}
    with np.load(filename) as arr:
        for var in list(arr.keys()):
            data[var] = arr[var]
    return data


def get_local_weather_data(filename):
    """

    Load a local compressed weather datafile.

    Parameters
    ----------
    filename

    Returns
    -------

    """


    data = load_npz(filename)
    return build_weather_info(data)



def build_weather_info(info):
    """

    Parameters
    ----------
    info

    Returns
    -------

    """

    for f in info:
        try:
            if info[f].dtype == np.dtype('<U5'):
                info[f] = str(info[f])
            elif info[f].dtype == np.dtype('<U6'):
                info[f] = str(info[f])
            elif info[f].dtype == np.dtype('int64'):
                info[f] = int(info[f])
            elif info[f].dtype == np.dtype('float64'):
                info[f] = float(info[f])


        except:
            print(f)


    weather = pd.DataFrame.from_dict({
        'year': info['year'],
        'month': info['month'],
        'day': info['day'],
        'hour': info['hour'],
        'minute': info['minute'],
        'dni': info['dni'],
        'ghi': info['ghi'],
        'dhi': info['dhi'],
        'temp_air': info['temp_air'],
        'wind_speed': info['wind_speed'],
    }
    )

    weather.index = pd.to_datetime(
        pd.DataFrame.from_dict({
            'year': info['year'],
            'month': info['month'],
            'day': info['day'],
            'hour': info['hour'],
            'minute': info['minute'],
        })
    )

    weather.index = weather.index.tz_localize(
        pytz.FixedOffset(float(info['local_time_zone'] * 60)))

    # Remove long vectors from info.
    for f in list(info.keys()):
        if type(info[f]) == type(np.array([0])):
            del info[f]


    return weather, info



def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate Haversine distance in km between two locations.

    Parameters
    ----------
    lat1 : numeric
        latitude of first point, in degrees.
    lon1 : numeric
        longitude of first point, in degrees.
    lat2 : numeric
        latitude of second point, in degrees.
    lon2 : numeric
        longitude of second point, in degrees.

    Returns
    -------
    numeric: Haversine distance in km.

    """
    p = 0.017453292519943295
    a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p)*np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p)) / 2
    return 12742 * np.arcsin(np.sqrt(a))

def arg_closest_point(lat_point, lon_point, lat_list, lon_list):
    """
    Calculate the index of the closest point in the list of coordinates (
    lat_list, lon_list) to the point (lat_point, lon_point). Uses Haversine
    distance formula to calculate the distance.

    Parameters
    ----------
    lat_point : numeric
        latitude of point to search for, in degrees
    lon_point : numeric
        longitude of point to search for, in degrees.
    lat_list : array
        list of latitudes to search within, in degrees.
    lon_list : array
        list of longitudes to search within, in degrees. Must be the same size
        as lat_list

    Returns
    -------
        numeric : distance
    """
    return np.argmin(
        haversine_distance(np.array(lat_list), np.array(lon_list),
                           lat_point, lon_point))




#
#
# def haversine_distance(lat1, lon1, lat2, lon2):
#     p = 0.017453292519943295
#     a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p)*np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p)) / 2
#     return 12742 * np.arcsin(np.sqrt(a))

def closest_degrees(lat_find, lon_find, lat_list, lon_list):

    distance = np.sqrt( (lat_find-lat_list)**2 + (lon_find-lon_list)**2 )
    closest_index = np.argmin(np.array(distance))
    distance_in_degrees = distance[closest_index]

    return (closest_index, distance_in_degrees)



def find_closest_datafiles(lat,lon,filedata):
    """
    Finds the closest location to lat,lon in the filedata.

    :param lat:
    :param lon:
    :param filedata:
    :return:
    """
    closest_index = arg_closest_point(lat, lon,filedata['lat'],filedata['lon'])

    closest_location_id = filedata['location_id'][closest_index]
    # closest_lat = filedata['lat'][closest_index]
    # closest_lon = filedata['lon'][closest_index]

    closest_filedata = filedata[filedata['location_id']==closest_location_id]

    return closest_filedata
