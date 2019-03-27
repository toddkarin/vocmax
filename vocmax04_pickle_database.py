"""

This script is used for 'pickling' a set of NSRDB files in order to get
faster load times. This script reads in all years of NSRDB files for a given
location and combines them into a single compressed data file and associated
info file.

toddkarin

"""

import nsrdbtools
import numpy as np
import os
import pandas as pd

# Directory to inspect for getting weather data.
data_dir = '/Users/toddkarin/Documents/NSRDB/'

# Directory to put pickled files.
pickle_dir = '/Users/toddkarin/Documents/NSRDB_pickle/'

# Build dataframe of file info.
filedata = nsrdbtools.inspect_database(data_dir)
print(filedata.info())

# NSRDB data has one csv file for each year for each location.
# Find all unique locations in the dataset.
unique_locs = list(set(filedata['location_id']))

# Loop through locations.

# for j in [0]:
for j in range(len(unique_locs)):

    # Print progress
    print(j/len(unique_locs)*100)

    # Find all datafiles at a given location
    filedata_curr = filedata[filedata['location_id']==unique_locs[j]]

    # Sort by year
    filedata_curr = filedata_curr.sort_values('year')

    # Get the different parts of the filename
    fname_parts = filedata_curr.filename.to_list()[0].split('_')

    # Name the output files.
    data_filename = fname_parts[0] + '_' + fname_parts[1] + '_' + fname_parts[2] + '_weather.pkl'
    info_filename = fname_parts[0] + '_' + fname_parts[1] + '_' + fname_parts[2] + '_info.pkl'

    # Full path of output files
    data_fullpath = os.path.join(pickle_dir, data_filename)
    info_fullpath = os.path.join(pickle_dir, info_filename)


    if os.path.isfile(data_fullpath) and os.path.isfile(info_fullpath):
        # Skip over any files that already exist.
        print('file already exists')
    else:

        # If compressed files don't exist, read in filedata
        df, info = nsrdbtools.combine_csv(filedata_curr['fullpath'])

        # Rename fields
        weather = pd.DataFrame.from_dict({
            'dni': df['DNI'].astype(np.int16),
            'dhi': df['DHI'].astype(np.int16),
            'ghi': df['GHI'].astype(np.int16),
            'temp_air': df['Temperature'].astype(np.int8),
            'wind_speed': df['Wind Speed'].astype(np.float16)})

        # Do some data extraction and save to the info file.
        info['Min_temp_air'] = weather['temp_air'].min()
        info['Min_temp_air_ghi>150'] = weather['temp_air'][weather['ghi']>150].min()

        # Save the weather data file.
        weather.to_pickle( data_fullpath,compression='xz')

        # # Check
        # weather2 = nsrdbtools.import_weather_pickle(data_fullpath)
        # if np.max(np.abs(weather2['temp_air']-weather['temp_air']))>0:
        #     Exception('Save error')

        # Save the info file.
        info.to_pickle(info_fullpath)