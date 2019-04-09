"""

This script is used for compressing a set of NSRDB files in order to get
faster load times. This script is used to compress files, which are then
uploaded onto AWS S3.

All years of NSRDB files for a given location are read in and combined into a
single compressed data file and associated info file.

toddkarin

"""

import nsrdbtools
import numpy as np
import os
import pandas as pd

# Directory to inspect for getting weather data.
data_dir = '/Users/toddkarin/Documents/NSRDB/'

# Directory to put pickled files.
pickle_dir = '/Users/toddkarin/Documents/NSRDB_compressed/'

# Build dataframe of file info.
filedata = nsrdbtools.inspect_database(data_dir)


# NSRDB data has one csv file for each year for each location.
# Find all unique locations in the dataset.
unique_locs = list(set(filedata['location_id']))

print('Number locations: ' + str(len(unique_locs)))
# Loop through locations.

# for j in range(10):
for j in range(len(unique_locs)):

    # Print progress
    print('Iteration: {:.0f}, Percent done: {:.4f}'.format(j,j/len(unique_locs)*100))

    # Find all datafiles at a given location
    filedata_curr = filedata[filedata['location_id']==unique_locs[j]]

    # Sort by year
    filedata_curr = filedata_curr.sort_values('year')

    # Get the different parts of the filename
    fname_parts = filedata_curr.filename.to_list()[0].split('_')

    # Name the output files.
    data_filename = fname_parts[0] + '_' + fname_parts[1] + '_' + fname_parts[2] + '.npz'

    # Full path of output files
    data_fullpath = os.path.join(pickle_dir, data_filename)



    if os.path.isfile(data_fullpath):
        # Skip over any files that already exist.
        print('file already exists')
    else:

        # If compressed files don't exist, read in filedata
        df, info = nsrdbtools.combine_csv(filedata_curr['fullpath'])

        dni = np.array(df['DNI'].astype(np.int16))
        dhi = np.array(df['DHI'].astype(np.int16))
        ghi = np.array(df['GHI'].astype(np.int16))

        temp_air = np.array(df['Temperature'].astype(np.int8))
        wind_speed = np.array(df['Wind Speed'].astype(np.float16))

        year = np.array(df['Year'].astype(np.int16))
        month = np.array(df['Month'].astype(np.int8))
        day = np.array(df['Day'].astype(np.int8))
        hour = np.array(df['Hour'].astype(np.int8))
        minute = np.array(df['Minute'].astype(np.int8))

        np.savez_compressed(data_fullpath,
                            Source=info['Source'],
                            Location_ID=info['Location ID'],
                            Latitude=info['Latitude'],
                            Longitude=info['Longitude'],
                            Elevation=info['Elevation'],
                            local_time_zone=info['Local Time Zone'],
                            interval_in_hours=info['interval_in_hours'],
                            timedelta_in_years=info['timedelta_in_years'],
                            Version=info['Version'],
                            dni=dni,
                            dhi=dhi,
                            ghi=ghi,
                            temp_air=temp_air,
                            wind_speed=wind_speed,
                            year=year,
                            month=month,
                            day=day,
                            hour=hour,
                            minute=minute)


