"""
vocmaxlib

This python package calculates the maximum sting size for a photovoltaic
installation. The method is consistent with the NEC 2017 690.7 standard.

toddkarin
"""

import numpy as np
import pvlib
import pvlib.bifacial
# import nsrdbtools
# import socket
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import pandas as pd
import datetime
import glob
import pytz
from vocmax import nsrdb
import tqdm

import os
import urllib
import pytz
import sys
import os
import warnings

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

from vocmax.bifacial import pvfactors_timeseries

import glob
import vocmax

# from pvlib.bifacial import PVFactorsReportBuilder as PVFactorsReportBuilder

# Parameters entering into Voc calculation:


cec_modules = pvlib.pvsystem.retrieve_sam('CeCMod')

# Descriptions of hte various parameters used in the calculation.
explain = {
    'Voco': 'Open circuit voltage at reference conditions, in V',
    'Bvoco': 'Temperature dependence of open circuit voltage, in V/C',
    'Mbvoc': """Coefficient providing the irradiance dependence of the 
    temperature coefficient of open circuit voltage, typically assumed to be 
    zero, in V/C 

    """,
    'n_diode': 'Diode ideality factor, unitless',

    'cells_in_series': 'Number of cells in series in each module, dimensionless',

    'FD': """Fraction of diffuse irradiance arriving at the cell, typically 
    assumed to be 1, dimensionless 

    """,

    'alpha_sc': """The short-circuit current temperature coefficient of the 
    module, in A/C 

    """,

    'a_ref': """The product of the usual diode ideality factor (n_diode, 
    unitless), number of cells in series (cells_in_series), and cell thermal 
    voltage at reference conditions, in units of V. 

    """,

    'I_L_ref': """The light-generated current (or photocurrent) at reference 
    conditions, in amperes.

    """,

    'I_o_ref': """The dark or diode reverse saturation current at reference 
    conditions, in amperes. 


    """,

    'R_sh_ref': """The shunt resistance at reference conditions, in ohms.""",

    'R_s': """The series resistance at reference conditions, in ohms.""",

    'Isco': """Short circuit current at reference conditions, in amperes.""",

    'Impo': """Maximum-power current at reference conditions, in amperes.""",

    'Vmpo': """Maximum-power voltage at reference conditions, in volts.""",

    'Pmpo': """Maximum-power power at reference conditions, in watts.""",

    'Bisco': """Temperature coefficient of short circuit current, in A/C"""

}


def get_weather_data(lat, lon,
                     api_key,
                     cache_directory='cached_weather_data',
                     attributes='ghi,dhi,dni,wind_speed,air_temperature',
                     force_download=False,
                     your_name='PV Researcher',
                     your_email='researcheremail@gmail.com',
                     reason_for_use='String+Length+Design',
                     your_affiliation='Solar+Company+X',
                     join_mailing_list=False,
                     use_utc=False,
                     include_leap_year=True,
                     years=np.arange(1998, 2018.5),
                     interval='30',
                     ):
    """

    Retrieve weather data from the national solar radiation database (NSRDB).

    Description
    -----------

    df, info = get_weather_data(lat,lon,api_key) gets weather data from the
    NSRDB using the NSRDB api. Data download for a single location takes
    around 3 minutes. Once weather data is downloaded, it is stored in a
    local cache so it can be retrieved quickly. One sample point (lat=37.876,
    lon=-122.247) is provided with the function so sample data can be easily
    loaded without an api key.

    Api keys are available free of charge at https://developer.nrel.gov/signup/

    Note can only donwload data from NSRDB sequentially (not possible to
    download data using multiple scripts in parallel).

    Examples
    --------

    lat, lon = 37.876, -122.247
    # Note: Replace with your api key
    api_key = '7d5k51GGRVgNEaKeL2DQvLJgrji9gkjdfWEnvaYdg'
    df, info = vocmax.get_weather_data(lat,lon,api_key)


    Parameters
    ----------
    lat : float
        latitude of search point in fractional degrees

    lon : float
        longitude of search point in fractional degrees

    api_key : str
        api key for NSRDB, available at https://developer.nrel.gov/signup/

    cache_directory : str
        Location to stored cached data files. Default is the subdirectory
        'cached_weather_data' inside the current working directory.

    attributes : str
        comma separated list of attributes to download. Default is 'ghi,dhi,
        dni,wind_speed,air_temperature'

    force_download : bool
        If true, force downloading of weather data regardless of weather
        that particular location has already been downloaded. Default is false.

    your_name : str

        Your name for the api call, default is 'PV Researcher'

    your_email : str

        Your email for the api call, default is 'researcher.email@gmail.com'

    reason_for_use : str

        Reason for use for the api call, default is 'String Length Design'

    your_affiliation : str

        Affiliation for the api call, default is 'Solar Company X'

    join_mailing_list : bool

        Whether to join the NSRDB mailing list, default is False.

    use_utc

        Whether to download data using UTC as the time axis. default is False

    include_leap_year : bool

        Whether to include data for the day on the leap year. Default is True.

    years : list

        Numeric list of years to download data for. Default is np.arange(1998,2017.5)


    interval : str

        Time step for downloaded weather data in mintues. Options ars '60' or
        '30' (default).

    tz_localize : bool

        Weather to localize the time zone.

    Returns
    -------

    df : pandas dataframe
        Dataframe containing weather data with fields
        'year' - year of row.
        'month', 'day', 'hour', 'minute', 'dni', 'ghi', 'dhi',
       'temp_air', 'wind_speed'.

    info : dictionary
        Dictionary containting information on the weather dataset.


    """

    # Parse input
    your_name = your_name.replace(' ', '+')
    reason_for_use = reason_for_use.replace(' ', '+')
    your_affiliation = your_affiliation.replace(' ', '+')

    if join_mailing_list:
        mailing_list = 'true'
    else:
        mailing_list = 'false'

    if use_utc:
        utc = 'true'
    else:
        utc = 'false'

    if include_leap_year:
        leap_year = 'true'
    else:
        leap_year = 'false'

    # First check if data exists in cahce directory.
    if not force_download:
        search_str = os.path.join(cache_directory,
                                  '*_{:3.3f}_{:3.3f}.npz'.format(lat, lon))
        print(search_str)
        # One sample data point is provided with the package so that users don't
        # have to get an api key to try it out.
        if '{:3.3f}_{:3.3f}'.format(lat, lon) == '37.876_-122.247':
            print('getting sample data point')
            dir_path = os.path.dirname(os.path.realpath(__file__))

            df, info = nsrdb.get_local_weather_data(
                os.path.join(dir_path,
                             '123796_37.89_-122.26_search-point_37.876_-122.247.npz')
            )
            return df, info

        # Otherwise search the cache for a weather data file that has already
        # been downloaded.
        filename = glob.glob(search_str)
        if len(filename) > 0:
            # Cached weather data found, load it
            df, info = nsrdb.get_local_weather_data(filename[0])

            # TODO: Add checks that the loaded file has the same options as in the function call.
            return df, info
        else:
            # No cached weather data found.
            pass

    # Pull data from NSRDB because either force_download=True or no cached datafile found.
    print('Downloading weather data...')
    for j in tqdm.tqdm(range(len(years))):
        year = '{:.0f}'.format(years[j])

        # Declare url string
        url = 'http://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(
            year=year, lat=lat, lon=lon, leap=leap_year, interval=interval,
            utc=utc, name=your_name, email=your_email,
            mailing_list=mailing_list, affiliation=your_affiliation,
            reason=reason_for_use, api=api_key, attr=attributes)

        # file_name, urllib.request.urlretrieve(url, "testfile.txt")
        with urllib.request.urlopen(url) as f:

            # Get the data as a string.
            response = f.read().decode('utf-8')

            # Read the first few lines to get info on datafile
            info_df = pd.read_csv(StringIO(response), nrows=1)

            # Create a dictionary for the info file.
            info_iter = {}
            for p in info_df:
                info_iter[p] = info_df[p].iloc[0]

            df_iter = pd.read_csv(StringIO(response), skiprows=2)

            if np.diff(df_iter[0:2].Minute) == 30:
                interval = '30'
                info_iter['interval_in_hours'] = 0.5
            elif np.diff(df_iter[0:2].Minute) == 0:
                interval = '60'
                info_iter['interval_in_hours'] = 1
            else:
                print('Interval not understood!')

            # Set the time index in the pandas dataframe:
            year_iter = str(df_iter['Year'][0])

            df_iter = df_iter.set_index(
                pd.date_range('1/1/{yr}'.format(yr=year_iter),
                              freq=interval + 'Min',
                              periods=len(df_iter)))

            df_iter.index = df_iter.index.tz_localize(
                pytz.FixedOffset(float(info_iter['Time Zone'] * 60)))

            if j == 0:
                info = info_iter
                df = df_iter
            else:
                df = df.append(df_iter)

    info['timedelta_in_years'] = (df.index[-1] - df.index[0]).days / 365

    # Convert to int for lowering file size.
    dni = np.array(df['DNI'].astype(np.int16))
    dhi = np.array(df['DHI'].astype(np.int16))
    ghi = np.array(df['GHI'].astype(np.int16))
    temp_air = np.array(df['Temperature'].astype(np.float32))
    wind_speed = np.array(df['Wind Speed'].astype(np.float16))

    year = np.array(df['Year'].astype(np.int16))
    month = np.array(df['Month'].astype(np.int8))
    day = np.array(df['Day'].astype(np.int8))
    hour = np.array(df['Hour'].astype(np.int8))
    minute = np.array(df['Minute'].astype(np.int8))

    cache_directory = 'cached_weather_data'
    if not os.path.exists(cache_directory):
        print('Creating cache directory')
        os.mkdir(cache_directory)

    save_filename = os.path.join(cache_directory,
                                 '{}_{:3.2f}_{:3.2f}_search-point_{:3.3f}_{:3.3f}.npz'.format(
                                     info['Location ID'], info['Latitude'],
                                     info['Longitude'], lat, lon)
                                 )

    np.savez_compressed(save_filename,
                        Source=info['Source'],
                        Location_ID=info['Location ID'],
                        Latitude=info['Latitude'],
                        Longitude=info['Longitude'],
                        Elevation=info['Elevation'],
                        local_time_zone=info['Local Time Zone'],
                        leap_year=leap_year,
                        utc=utc,
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

    df.rename(columns={'Year': 'year',
                       'Month': 'month',
                       'Day': 'day',
                       'Hour': 'hour',
                       'Minute': 'minute',
                       'DNI': 'dni',
                       'GHI': 'ghi',
                       'DHI': 'dhi',
                       'Temperature': 'temp_air',
                       'Wind Speed': 'wind_speed',
                       }, inplace=True)

    # TODO: Consider reloading from the file for consistency with future iterations.
    return df, info


# def ashrae_get_data():
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#
#     # Load temperature difference data.
#     ashrae = pd.read_csv(
#         os.path.join(dir_path, 'ASHRAE2017_temperature_data.csv')
#     )
#     return ashrae


def ashrae_get_design_conditions_at_loc(lat, lon, ashrae):
    """

    Parameters
    ----------
    lat
    lon
    ashrae : dataframe

    Returns
    -------
    dataframe

        fields are

            'Latitude'

            'Longitude'

            'Extreme Annual Mean Minimum Dry Bulb Temperature' - ASHRAE
            extreme minimum dry bulb temperature, in C

    """
    # df = ashrae_get_design_conditions()

    # Calculate distance to search point.
    distance = nsrdb.haversine_distance(lat, lon, ashrae['Lat'], ashrae['Lon'])

    closest_idx = distance.idxmin()

    return ashrae.iloc[closest_idx]


def nec_correction_factor(temperature):
    """
    NEC 690.7(A)(2) correction factor from NEC2017.

    Parameters
    ----------
    temperature : numeric
        Temperature in C.

    Returns
    -------
    correction_factor : flat

    """
    is_array = isinstance(temperature, np.ndarray)
    temperature = np.array([temperature])
    f = np.zeros_like(temperature, dtype='float') + 1

    f[temperature < 25] = 1.02
    f[temperature < 20] = 1.04
    f[temperature < 15] = 1.06
    f[temperature < 10] = 1.08
    f[temperature < 5] = 1.10
    f[temperature < 0] = 1.12
    f[temperature < -5] = 1.14
    f[temperature < -10] = 1.16
    f[temperature < -15] = 1.18
    f[temperature < -20] = 1.20
    f[temperature < -25] = 1.21
    f[temperature < -30] = 1.23
    f[temperature < -35] = 1.25
    f[np.isnan(temperature)] = np.nan

    if not is_array:
        f = f[0]
    return f


def get_nsrdb_temperature_error(lat, lon,
                                number_of_closest_points=5):
    """
    Find the temperature error for a particular location.

    The NSRDB database provides temeprature data for many locations. However,
    these data are taken from the MERRA-2 dataset, and have some error
    compared to ground measurements. The temperature error depends on location.

    As a comparison, we calculated the mean minimum extreme minimum dry bulb
    temperature using NSRDB data and compared to ASHRAE data. The temperature
    difference determines the safety factor necessary for string length
    calculations.

    This function finds the closest points to a particular lat,lon coordinate
    in the ASHRAE dataset and returns the maximum temperature difference (
    NSRDB - ASHRAE) for these locations. A higher temperature difference
    means that the NSRDB is overestimating the true temperature that is
    measured at a ground station. Higher positive temperature differences
    mean that a larger safety factor should be used when calculating string
    length. The Safety factor can be calculated

    Examples
    --------

    temperature_difference =  vocmax.get_nsrdb_temperature_error(lat,lon)

    Parameters
    ----------
    lat : float
        latitude of search point in fractional degrees

    lon : float
        longitude of search point in fractional degrees

    number_of_closest_points : int
        The number of closest datapoints to find. Default is 5.

    Returns
    -------
    temperature_difference : float
        max temperature difference between NSRDB point and closest ASHRAE
        points. A positive number means that the NSRDB design temperature is
        higher than the ASHRAE design temperature. If a positive temperature
        difference is found, then an additional safety factor is suggested to
        account for this error in the NSRDB dataset.


    """

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Load temperature difference data.
    df = pd.read_pickle(
        os.path.join(dir_path, 'nsrdb_ashrae_comparison.pkl')
    )

    # Calculate distance to search point.
    distance = vocmax.nsrdb.haversine_distance(lat, lon, df['lat'], df['lon'])

    # Find the closest locations.
    distance_sort = distance.sort_values()
    closest_idx = distance_sort.index[:number_of_closest_points]

    # Calculate temperature difference
    temperature_difference = df['nsrdb-ashrae Extreme_Annual_Mean_Min_DB'].loc[
        closest_idx]

    return temperature_difference.max()


def ashrae_import_design_conditions(filename='2017DesignConditions_s.xlsx'):
    """

    Load the ASHRAE 2017 design conditions excel file. This file is NOT
    provided in vocmax, it must be purchased directly from ASHRAE and added
    to the current directory. The filename is '2017DesignConditions_s.xlsx'.
    The '_s' at the end of the filename stands for 'SI'. There is also
    another file '2017DesignConditions_p.xlsx' that contains measurements in
    imperial units, do not use this file.

    In order to use this function, purchase the weather data viewer DVD,
    version 6.0, available at:
    https://www.techstreet.com/ashrae/standards/weather-data-viewer-dvd-version-6-0?ashrae_auth_token=12ce7b1d-2e2e-472b-b689-8065208f2e36&product_id=1949790

    Importing the excel file takes around 1 minute, the data is then saved as
    a csv file with name filename + '.csv' in the current directory. This
    makes loading quick the second time.

    Parameters
    ----------
    filename : string
        Filename to import.

    Returns
    -------
    df : dataframe
        Pandas dataframe containing certain fields of the weather data file.

    """

    # filename = '2017DesignConditions_s.xlsx'

    df = pd.read_excel(filename,
                       skiprows=0,
                       sheet_name=0,
                       header=[1, 2, 3],
                       verbose=False)

    filename_out = filename + '.csv'

    df_out = pd.DataFrame(
        {'Lat': np.array(df['Lat']).flatten(),
         'Lon': np.array(df['Lon']).flatten(),
         'Country': np.array(df['Country']).flatten(),
         'Station Name': np.array(df['Station Name']).flatten(),
         'Extreme_Annual_Mean_Min_DB': np.array(
             df['Extreme Annual DB']['Mean']['Min']).flatten(),
         'Extreme_Annual_Standard Deviation_Min_DB': np.array(
             df['Extreme Annual DB']['Standard Deviation']['Min']).flatten(),
         '20-Year Return Period Extreme Min DB': np.array(
             df['n-Year Return Period Values of Extreme DB']['n=20 years'][
                 'Min']).flatten(),
         }
    )
    df_out.to_csv(filename_out, index=False)

    return df_out


def ashrae_is_design_conditions_available(
        filename='2017DesignConditions_s.xlsx'):
    return os.path.exists(filename)


def ashrae_get_design_conditions(filename='2017DesignConditions_s.xlsx'):
    """
    Get the ASHRAE design conditions data.


    Parameters
    ----------
    filename

    Returns
    -------
    df : dataframe

        Pandas dataframe containing certain fields of the ASHARE design
        conditions file



    """

    if os.path.exists(filename + '.csv'):
        df = pd.read_csv(filename + '.csv')
    elif os.path.exists(filename):
        print(
            """Importing and compressing ASHRAE design conditions excel file. Future calls will quickly call csv version. """)
        print('Found file: {}'.format(filename))
        print('Expected loading time: 1.0 minute')

        df = ashrae_import_design_conditions(filename)
    else:
        raise Exception(
            "Design conditions file '{}' not found. File must be purchased from ASHRAE and placed in current directory.".format(
                filename))

    return df


def simulate_system(weather, info, module_parameters,
                    racking_parameters, thermal_model,
                    irrad_model='perez',
                    nighttime_irradiance_addition=0
                    ):
    """

    Use the PVLIB SAPM model to calculate maximum Voc.

    Parameters
    ----------
    weather : Dataframe
        Weather data dataframe containing the columns:

        'dni': Direct Normal Irradiance (W/m^2)
        'dhi': Diffuse Horizontal Irradiance (W/m^2)
        'ghi' Global Horizontal Irradiance (W/m^2)
        'temp_air': air temperature (C)
        'wind_speed': 10 m wind speed in (m/s)

    info : dict
        Dictionary containing location information with fields:

        'Latitude': float
            latitude in degrees

        'Longitude': float
            longitude in degrees.

        Other fields may be included in info as well and will not interfere
        with operation.


    module_parameters : dict
        Dict or Series containing the below fields describing the module

        'Voco' : float
            Open circuit voltage at reference conditions.

        'Bvoco' : float
            Temperature coefficient of open circuit voltage, in Volts/C.

        'cells_in_series' : int
            Number of cells in series in the module.

        'n_diode' : float
            Diode ideality factor

        'Mbvoc' : float
            Irradiance dependence of the temperature coefficient of
            open-circuit voltage, typically assumed to be zero.

        'FD' : float
            Fraction of diffuse irradiance used by the module.

        'efficiency' : float
            Module fractional efficiency.

        'iv_model' : string
            Model for calculating Voc. Can be 'sapm', 'cec' or 'desoto'.
            TODO: Describe better.

        'aoi_model' : string
            Model for calculating the angle-of-incidence loss function. Can
            be 'no_loss' or 'ashrae'. The 'no_loss' method assumes that no
            extra reflection losses are accrued at non-normal angles of
            incidence. The 'ashrae' option uses the model in
            pvlib.pvsystem.ashraeiam

        'is_bifacial' : bool
            True if module is bifacial. Using False will force the use of
            monofacial models even if 'bifacial_model' in the
            racking_parameters input dict is set to a value.

        bifaciality_factor : float
            Number describing the efficiency of the backside of the module
            relative to the frontside. A typical values is 0.7.

    racking_parameters : dict
        dictionary describing the racking setup. Contains fields:

        'racking_type' : str
            Can be 'fixed_tilt' for a stationary PV system or 'single_axis'
            for a single axis tracker.

        'surface_tilt' : float
            If racking_type is 'fixed_tilt', specify the surface tilt in
            degrees from horizontal.

        'surface_azimuth' : float
            If racking type is 'surface_azimuth', specify the racking azimuth
            in degrees. A value of 180 degrees has the module face oriented
            due South.

        'axis_tilt' : float
            If racking_type is 'single_axis', specify the the tilt of the
            axis of rotation (i.e, the y-axis defined by axis_azimuth) with
            respect to horizontal, in decimal degrees. Standard value is 0.

        'axis_azimuth' : float
            If racking_type is 'single_axis', specify a value denoting the
            compass direction along which the axis of rotation lies. Measured
            in decimal degrees East of North. Standard value is 0.

        'backtrack' : bool
            Controls whether the tracker has the capability to ''backtrack''
            to avoid row-to-row shading. False denotes no backtrack
            capability. True denotes backtrack capability.

        'gcr' : float
            A value denoting the ground coverage ratio of a tracker system
            which utilizes backtracking; i.e. the ratio between the PV array
            surface area to total ground area. A tracker system with modules
            2 meters wide, centered on the tracking axis, with 6 meters
            between the tracking axes has a gcr of 2/6=0.333. If gcr is not
            provided, a gcr of 2/7 is default. gcr must be <=1

        bifacial_model : string

            Can be 'proportional' or 'pvfactors'. The 'proportional' bifacial
            modeling method calculates the effective irradiance on the
            frontside of the module and then assumes that the backside
            irradiance is equal to the frontside irradiance times the
            backside_irradiance_fraction times the bifaciality_factor. The
            'pvfactors' method uses bifacial modeling found in the pvfactors
            package.

        backside_irradiance_fraction : float

            For simple bifacial modeling, the backside irradiance is assumed
            to be equal to the frontside irradiance times the
            backside_irradiance_fraction. Required if using
            bifacial_model 'proportional'. Typical value is 0.3.

        pvrow_height : float.
            Height of the pv rows, measured at their center (m). Required if
            using bifacial_model 'pvfactors'.

        pvrow_width : float
            Width of the pv rows in the considered 2D plane (m). Required if
            using bifacial_model 'pvfactors'.

        albedo: float
            Ground albedo. Required if using bifacial_model 'pvfactors'.

        n_pvrows: int, default 3
            Number of PV rows to consider in the PV array. Required if
            using bifacial_model 'pvfactors'.

        index_observed_pvrow: int, default 1

            Index of the PV row whose incident irradiance will be returned.
            Indices of PV rows go from 0 to n_pvrows-1. Required if using
            bifacial_model 'pvfactors'.

        rho_front_pvrow: float, default 0.03

            Front surface reflectivity of PV rows. Required if using
            bifacial_model 'pvfactors'.

        rho_back_pvrow: float, default 0.05

            Back surface reflectivity of PV rows. Required if using
            bifacial_model 'pvfactors'.

        horizon_band_angle: float, default 15

            Elevation angle of the sky dome's diffuse horizon band (deg).
            Required if using bifacial_model 'pvfactors'.

    thermal_model : dict

        named_model : string

            If named_model is 'explicit', then use SAPM parameters defined by
            a, b, and deltaT. Otherwise named_model can be one of the
            following strings:

                ‘open_rack_cell_glassback’ (default)
                ‘roof_mount_cell_glassback’
                ‘open_rack_cell_polymerback’
                ‘insulated_back_polymerback’
                ‘open_rack_polymer_thinfilm_steel’
                ‘22x_concentrator_tracker’

        a: float

            SAPM module parameter for establishing the upper limit for
            module temperature at low wind speeds and high solar
            irradiance.

        b :float

            SAPM module parameter for establishing the rate at which the
            module temperature drops as wind speed increases (see SAPM
            eqn. 11).

        deltaT :float

            SAPM module parameter giving the temperature difference
            between the cell and module back surface at the reference
            irradiance, E0.

        open_circuit_rise : bool

            The SAPM parameters are measured for modules at maximum power
            point. At open-circuit voltage the module is warmer because less
            energy is exported as electricity. If open_circuit_rise is True
            then this temperature rise is taken into account, if False then
            it is not.

        thermal_mass : bool

            Weather to take into account the thermal mass of the modules when
            calculating temperature. Thermal mass is performed using an
            exponentially weighted moving average [Bosco2016]

        thermal_time_constant : float

            Thermal time constant of the modules, in minutes.


    irrad_model : str

        Irradiance model for determining in-plane sky diffuse irradiance
        component using the specified sky diffuse irradiance model. Default
        is 'perez'

        Sky diffuse models include:
            * isotropic (default)
            * klucher
            * haydavies
            * reindl
            * king
            * perez


    Returns
    -------

    dataframe containing simulation results. Includes the fields present in
    input 'weather' in addtion to:

        'v_oc': open circuit voltage in Volts

        'aoi': angle of incidence in degrees.

        'temp_cell': cell temeprature in C.


    References
    ----------

    [Bosco2016] N. Bosco, et al., Climate specific thermomechanical fatigue
    of flat plate photovoltaic module solder joints, Microelectronics
    Reliability (2016), http://dx.doi.org/10.1016/j.microrel.2016.03.024

    """

    # Rename the weather data for input to PVLIB.
    if np.all([c in weather.columns for c in ['dni', 'dhi', 'ghi', 'temp_air',
                                              'wind_speed', 'year', 'month',
                                              'day', 'hour', 'minute']]):
        # All colmuns are propoerly labeled, skip any relabeling.
        pass
    else:
        # Try renaming from NSRDB default values.
        weather = weather.rename(
            columns={'DNI': 'dni',
                     'DHI': 'dhi',
                     'GHI': 'ghi',
                     'Temperature': 'temp_air',
                     'Wind Speed': 'wind_speed',
                     'Year': 'year',
                     'Month': 'month',
                     'Day': 'day',
                     'Hour': 'hour',
                     'Minute': 'minute'})

    df = weather.copy()

    # Set location
    location = pvlib.location.Location(latitude=info['Latitude'],
                                       longitude=info['Longitude'])

    # Add module parameters if some aren't specified.
    module_parameters = add_default_module_params(module_parameters)

    # #
    # start_time = time.time()
    # # This is the most time consuming step
    # solar_position = location.get_solarposition(weather.index, method='nrel_numpy')
    # print( time.time()-start_time)
    #

    # Ephemeris method is faster and gives very similar results.
    solar_position = location.get_solarposition(weather.index,
                                                method='ephemeris')

    # Get surface tilt and azimuth
    if racking_parameters['racking_type'] == 'fixed_tilt':

        surface_tilt = racking_parameters['surface_tilt']
        surface_azimuth = racking_parameters['surface_azimuth']

        # idealized assumption
    elif racking_parameters['racking_type'] == 'single_axis':

        # Avoid nan warnings by presetting unphysical zenith angles.
        solar_position['apparent_zenith'][
            solar_position['apparent_zenith'] > 90] = 90

        # Todo: Check appraent_zenith vs. zenith.
        single_axis_vals = pvlib.tracking.singleaxis(
            solar_position['apparent_zenith'],
            solar_position['azimuth'],
            axis_tilt=racking_parameters['axis_tilt'],
            axis_azimuth=racking_parameters['axis_azimuth'],
            max_angle=racking_parameters['max_angle'],
            backtrack=racking_parameters['backtrack'],
            gcr=racking_parameters['gcr']
        )
        surface_tilt = single_axis_vals['surface_tilt']
        surface_azimuth = single_axis_vals['surface_azimuth']
    else:
        raise Exception('Racking system not recognized')

    # Extraterrestrial radiation
    dni_extra = pvlib.irradiance.get_extra_radiation(solar_position.index)

    airmass = location.get_airmass(solar_position=solar_position)

    # Perez is a good diffuse sky model
    total_irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt,
        surface_azimuth,
        solar_position['zenith'],
        solar_position['azimuth'],
        weather['dni'].astype('float'),
        weather['ghi'].astype('float'),
        weather['dhi'].astype('float'),
        model='perez',
        dni_extra=dni_extra,
        airmass=airmass['airmass_relative'],
        albedo=racking_parameters['albedo'])

    # Add a small irradiance during night time

    for k in total_irrad.keys():
        total_irrad[k][np.isnan(total_irrad[k])] = 0
        total_irrad[k] = total_irrad[k] + nighttime_irradiance_addition

    if racking_parameters['racking_type'] == 'fixed_tilt':
        aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
                                   solar_position['zenith'],
                                   solar_position['azimuth'])
    elif racking_parameters['racking_type'] == 'single_axis':
        aoi = single_axis_vals['aoi']
    else:
        raise Exception('Racking type not understood')
        # aoi = single_axis_vals['aoi']

    if (not 'named_model' in thermal_model) or thermal_model[
        'named_model'] == 'explicit':
        thermal_model_params = {k: thermal_model[k] for k in
                                ['a', 'b', 'deltaT']}
    else:
        temperature_model_parameters = \
        pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']

        thermal_model_params = temperature_model_parameters[
            thermal_model['named_model']]



    temperature_cell = pvlib.temperature.sapm_cell(
        poa_global=total_irrad['poa_global'],
        temp_air=weather['temp_air'],
        wind_speed=weather['wind_speed'],
        a=thermal_model_params['a'],
        b=thermal_model_params['b'],
        deltaT=thermal_model_params['deltaT'])

    # temps = pvlib.temperature.sapm_cell(total_irrad['poa_global'],
    #                                      weather['wind_speed'],
    #                                      weather['temp_air'],
    #                                      thermal_model_params)

    # if thermal_model['thermal_mass']:
    #     thermal_alpha = np.exp(-(info['interval_in_hours'] * 60) / 270)
    #

    if thermal_model['open_circuit_rise']:
        temperature_cell = weather['temp_air'] + \
                           (temperature_cell - weather['temp_air']) / (
                                       1 - module_parameters['efficiency'])

    # Spectral loss is typically very small on order of a few percent, assume no
    # spectral loss for simplicity
    spectral_loss = 1

    if not 'aoi_model' in module_parameters:
        module_parameters['aoi_model'] = 'no_loss'
    if not 'FD' in module_parameters:
        module_parameters['FD'] = 1

    # AOI loss:
    if module_parameters['aoi_model'] == 'no_loss':
        aoi_loss = 1
    elif module_parameters['aoi_model'] == 'ashrae':
        aoi_loss = pvlib.iam.ashrae(aoi,
                                    b=module_parameters['ashrae_iam_param'])
    else:
        raise Exception('aoi_model must be ashrae or no_loss')

    # Calculate effective irradiance.

    if ('is_bifacial' in module_parameters) and \
            (module_parameters['is_bifacial'] == True):
        if not 'bifacial_model' in racking_parameters:
            warnings.warn("""'bifacial_model' in racking_parameters is not 
            specified, can be 'simple' or 'pvfactors'. Defaulting to 
            'simple'.""")
            racking_parameters['bifacial_model'] = 'proportional'

        if racking_parameters['bifacial_model'] == 'proportional':

            effective_irradiance_front = calculate_effective_irradiance(
                total_irrad['poa_direct'],
                total_irrad['poa_diffuse'],
                aoi_loss=aoi_loss,
                FD=module_parameters['FD']
            )

            if not 'backside_irradiance_fraction' in racking_parameters:
                raise Exception("""Must specify 'backside_irradiance_fraction' in 
                racking_parameters for bifacial modeling. """
                                )

            effective_irradiance_back = effective_irradiance_front * \
                                        racking_parameters[
                                            'backside_irradiance_fraction'] * \
                                        module_parameters['bifaciality_factor']

            effective_irradiance = effective_irradiance_front + effective_irradiance_back
            df['effective_irradiance_front'] = effective_irradiance_front
            df['effective_irradiance_back'] = effective_irradiance_back


        elif racking_parameters['bifacial_model'] == 'pvfactors':

            total_inc_front, total_inc_back, poa_front_absorbed, poa_back_absorbed = pvfactors_timeseries(
                solar_position['azimuth'], solar_position['zenith'],
                surface_azimuth,
                surface_tilt,
                racking_parameters['axis_azimuth'],
                weather.index, weather['dni'], weather['dhi'],
                racking_parameters['gcr'],
                racking_parameters['pvrow_height'],
                racking_parameters['pvrow_width'],
                racking_parameters['albedo'],
                n_pvrows=racking_parameters['n_pvrows'],
                # fast_mode_pvrow_index=racking_parameters['fast_mode_pvrow_index'],
                index_observed_pvrow=racking_parameters['index_observed_pvrow'],
                rho_front_pvrow=racking_parameters['rho_front_pvrow'],
                rho_back_pvrow=racking_parameters['rho_back_pvrow'],
                horizon_band_angle=racking_parameters['horizon_band_angle'],
                # run_parallel_calculations=racking_parameters['run_parallel_calculations'],
                # n_workers_for_parallel_calcs=racking_parameters['n_workers_for_parallel_calcs']
            )

            effective_irradiance_front = np.nan_to_num(poa_front_absorbed)
            effective_irradiance_back = np.nan_to_num(poa_back_absorbed)

            effective_irradiance = effective_irradiance_front + effective_irradiance_back

            df['effective_irradiance_front'] = effective_irradiance_front
            df['effective_irradiance_back'] = effective_irradiance_back
        else:
            raise Exception(
                "racking_parameters['bifacial_model'] must be either 'proportional' or 'pvfactors'. ")


    else:
        # Not bifacial, i.e. monofacial module.
        effective_irradiance = calculate_effective_irradiance(
            total_irrad['poa_direct'],
            total_irrad['poa_diffuse'],
            aoi_loss=aoi_loss,
            FD=module_parameters['FD']
        )

    v_oc = sapm_voc(effective_irradiance, temperature_cell,
                    module_parameters)

    df['aoi'] = aoi
    # df['aoi_loss'] = aoi_loss
    df['temp_cell'] = temperature_cell
    df['temp_air'] = weather['temp_air']
    df['effective_irradiance'] = effective_irradiance
    df['v_oc'] = v_oc
    df['surface_tilt'] = surface_tilt
    df['surface_azimuth'] = surface_azimuth
    df['solar_zenith'] = solar_position['apparent_zenith']
    df['solar_azimuth'] = solar_position['azimuth']
    df['poa_direct'] = total_irrad['poa_direct']
    df['poa_diffuse'] = total_irrad['poa_diffuse']
    return df


#
# def pvfactors_timeseries(
#         solar_azimuth, solar_zenith, surface_azimuth, surface_tilt,
#         axis_azimuth,
#         timestamps, dni, dhi, gcr, pvrow_height, pvrow_width, albedo,
#         n_pvrows=3,fast_mode_pvrow_index=2,index_observed_pvrow=1,
#         rho_front_pvrow=0.03, rho_back_pvrow=0.05,
#         horizon_band_angle=15.,
#         run_parallel_calculations=True, n_workers_for_parallel_calcs=2):
#     """
#     Calculate front and back surface plane-of-array irradiance on
#     a fixed tilt or single-axis tracker PV array configuration, and using
#     the open-source "pvfactors" package.
#     Please refer to pvfactors online documentation for more details:
#     https://sunpower.github.io/pvfactors/
#
#     Parameters
#     ----------
#     solar_azimuth: numeric
#         Sun's azimuth angles using pvlib's azimuth convention (deg)
#     solar_zenith: numeric
#         Sun's zenith angles (deg)
#     surface_azimuth: numeric
#         Azimuth angle of the front surface of the PV modules, using pvlib's
#         convention (deg)
#     surface_tilt: numeric
#         Tilt angle of the PV modules, going from 0 to 180 (deg)
#     axis_azimuth: float
#         Azimuth angle of the rotation axis of the PV modules, using pvlib's
#         convention (deg). This is supposed to be fixed for all timestamps.
#     timestamps: datetime or DatetimeIndex
#         List of simulation timestamps
#     dni: numeric
#         Direct normal irradiance (W/m2)
#     dhi: numeric
#         Diffuse horizontal irradiance (W/m2)
#     gcr: float
#         Ground coverage ratio of the pv array
#     pvrow_height: float
#         Height of the pv rows, measured at their center (m)
#     pvrow_width: float
#         Width of the pv rows in the considered 2D plane (m)
#     albedo: float
#         Ground albedo
#     n_pvrows: int, default 3
#         Number of PV rows to consider in the PV array
#     fast_mode_pvrow_index: int
#         In fast mode, the user will be able to calculate rapidly (but with
#         additional approximations) the incident irradiance on the back side
#         of one PV row in the PV array, and the index of that PV row needs to
#         be passed as a keyword argument to fast_mode_pvrow_index
#     index_observed_pvrow: int, default 1
#         Index of the PV row whose incident irradiance will be returned. Indices
#         of PV rows go from 0 to n_pvrows-1.
#     rho_front_pvrow: float, default 0.03
#         Front surface reflectivity of PV rows
#     rho_back_pvrow: float, default 0.05
#         Back surface reflectivity of PV rows
#     horizon_band_angle: float, default 15
#         Elevation angle of the sky dome's diffuse horizon band (deg)
#     run_parallel_calculations: bool, default True
#         pvfactors is capable of using multiprocessing. Use this flag to decide
#         to run calculations in parallel (recommended) or not.
#     n_workers_for_parallel_calcs: int, default 2
#         Number of workers to use in the case of parallel calculations. The
#         '-1' value will lead to using a value equal to the number
#         of CPU's on the machine running the model.
#
#     Returns
#     -------
#     front_poa_irradiance: numeric
#         Calculated incident irradiance on the front surface of the PV modules
#         (W/m2)
#     back_poa_irradiance: numeric
#         Calculated incident irradiance on the back surface of the PV modules
#         (W/m2)
#     df_registries: pandas DataFrame
#         DataFrame containing detailed outputs of the simulation; for
#         instance the shapely geometries, the irradiance components incident on
#         all surfaces of the PV array (for all timestamps), etc.
#         In the pvfactors documentation, this is refered to as the "surface
#         registry".
#
#     References
#     ----------
#     .. [1] Anoma, Marc Abou, et al. "View Factor Model and Validation for
#         Bifacial PV and Diffuse Shade on Single-Axis Trackers." 44th IEEE
#         Photovoltaic Specialist Conference. 2017.
#     """
#
#     # Convert pandas Series inputs (and some lists) to numpy arrays
#     if isinstance(solar_azimuth, pd.Series):
#         solar_azimuth = solar_azimuth.values
#     elif isinstance(solar_azimuth, list):
#         solar_azimuth = np.array(solar_azimuth)
#     if isinstance(solar_zenith, pd.Series):
#         solar_zenith = solar_zenith.values
#     if isinstance(surface_azimuth, pd.Series):
#         surface_azimuth = surface_azimuth.values
#     elif isinstance(surface_azimuth, list):
#         surface_azimuth = np.array(surface_azimuth)
#     if isinstance(surface_tilt, pd.Series):
#         surface_tilt = surface_tilt.values
#     if isinstance(dni, pd.Series):
#         dni = dni.values
#     if isinstance(dhi, pd.Series):
#         dhi = dhi.values
#     if isinstance(solar_azimuth, list):
#         solar_azimuth = np.array(solar_azimuth)
#
#     # Import pvfactors functions for timeseries calculations.
#     from pvfactors.run import (run_timeseries_engine,
#                                run_parallel_engine)
#
#     # Build up pv array configuration parameters
#     pvarray_parameters = {
#         'n_pvrows': n_pvrows,
#         'axis_azimuth': axis_azimuth,
#         'pvrow_height': pvrow_height,
#         'pvrow_width': pvrow_width,
#         'gcr': gcr,
#         'rho_front_pvrow': rho_front_pvrow,
#         'rho_back_pvrow': rho_back_pvrow,
#         'horizon_band_angle': horizon_band_angle
#     }
#
#     # Run pvfactors calculations: either in parallel or serially
#     if run_parallel_calculations:
#
#         # report_builder = ReportBuilder(fast_mode_pvrow_index)
#
#         report = run_parallel_engine(
#             ReportBuilder(fast_mode_pvrow_index), pvarray_parameters,
#             timestamps, dni, dhi,
#             solar_zenith, solar_azimuth,
#             surface_tilt, surface_azimuth,
#             albedo, n_processes=n_workers_for_parallel_calcs,
#             fast_mode_pvrow_index=fast_mode_pvrow_index
#         )
#     else:
#         report = run_timeseries_engine(
#             PVFactorsReportBuilder.build, pvarray_parameters,
#             timestamps, dni, dhi,
#             solar_zenith, solar_azimuth,
#             surface_tilt, surface_azimuth,
#             albedo)
#
#     print(report)
#     # Turn report into dataframe
#     df_report = pd.DataFrame(report, index=timestamps)
#
#     return df_report.total_inc_front, df_report.total_inc_back
#
#
# class ReportBuilder(object):
#     """A class is required to build reports when running calculations with
#     multiprocessing because of python constraints"""
#
#     def __init__(self, fast_mode_pvrow_index):
#         """Create report builder object for fast mode simulation.
#
#         Parameters
#         ----------
#         fast_mode_pvrow_index : int
#             Index of PV row whose back side irradiance we want to report
#         """
#         self.fast_mode_pvrow_index = fast_mode_pvrow_index
#
#     def build(self, report, pvarray):
#         # Initialize the report as a dictionary
#         if report is None:
#             report = {'total_inc_back': []}
#         # Add elements to the report
#         if pvarray is not None:
#             pvrow = pvarray.pvrows[self.fast_mode_pvrow_index]
#             report['total_inc_back'].append(
#                 pvrow.back.get_param_weighted('qinc'))
#         else:
#             # No calculation was performed, because sun was down
#             report['total_inc_back'].append(np.nan)
#
#         return report
#
#     @staticmethod
#     def merge(reports):
#         """Works for dictionary reports"""
#         report = reports[0]
#         # Merge other reports
#         keys_report = list(reports[0].keys())
#         for other_report in reports[1:]:
#             for key in keys_report:
#                 report[key] += other_report[key]
#         return report


#
# class PVFactorsReportBuilder(object):
#     """In pvfactors, a class is required to build reports when running
#     calculations with multiprocessing because of python constraints"""
#
#     def __init__(self, fast_mode_pvrow_index):
#         """Create report builder object for fast mode simulation.
#
#         Parameters
#         ----------
#         fast_mode_pvrow_index : int
#             Index of PV row whose back side irradiance we want to report
#         """
#         self.fast_mode_pvrow_index = fast_mode_pvrow_index
#
#     # @staticmethod
#     def build(self,report, pvarray):
#         """Reports will have total incident irradiance on front and
#         back surface of center pvrow (index=1)"""
#         # Initialize the report as a dictionary
#         if report is None:
#             list_keys = ['total_inc_back', 'total_inc_front']
#             report = {key: [] for key in list_keys}
#         # Add elements to the report
#         if pvarray is not None:
#             # pvrow = pvarray.pvrows[1]  # use center pvrow
#             pvrow = pvarray.pvrows[self.fast_mode_pvrow_index]
#             print(pvrow.front)
#             report['total_inc_back'].append(
#                 pvrow.back.get_param_weighted('qinc'))
#             report['total_inc_front'].append(
#                 pvrow.front.get_param_weighted('qinc'))
#         else:
#             # No calculation is performed when the sun is down
#             report['total_inc_back'].append(np.nan)
#             report['total_inc_front'].append(np.nan)
#
#         return report
#
#     @staticmethod
#     def merge(reports):
#         """Works for dictionary reports"""
#         report = reports[0]
#         # Merge only if more than 1 report
#         if len(reports) > 1:
#             keys_report = list(reports[0].keys())
#             for other_report in reports[1:]:
#                 if other_report is not None:
#                     for key in keys_report:
#                         report[key] += other_report[key]
#         return report
#

def add_default_module_params(module_parameters):
    """

    Adds default fields to the module_parameters dictionary.

    Parameters
    ----------
    module_parameters : dict

    Examples
    --------

    >> module = add_default_module_params(module)

    Returns
    -------
    module_parameters : dict
        Same as input, except default values are added for the following fields:

        'Mbvoc' : 0

        'FD' : 1

        'iv_model' : 'sapm'

        'aoi_model' : 'no_loss'



    """

    if not 'Mbvoc' in module_parameters:
        module_parameters['Mbvoc'] = 0

    if not 'FD' in module_parameters:
        module_parameters['FD'] = 1

    if not 'iv_model' in module_parameters:
        module_parameters['iv_model'] = 'sapm'

    if not 'aoi_model' in module_parameters:
        module_parameters['aoi_model'] = 'no_loss'

    return module_parameters


def make_voc_summary(df, info, module_parameters,
                     string_design_voltage=1500,
                     safety_factor=0.023,
                     ashrae='local_load'):
    """

    Calculate maximum Voc expected using four relevant standards. See
    documentation for a description of the standards.

    Parameters
    ----------
    df : dataframe
        Dataframe containing fields: 'v_oc', 'ghi', 'temp_air'

    info : dict

        Dictionary containing fields 'lat' and 'lon'. These are used to
        calculate the ASHRAE standards.

    module_parameters : dict
        Dictionary containing module parameters. The module paramaters are
        used in a direct call to the function calculate_voc.

    string_design_voltage : float
        Maximum allowable string voltage for the design, in V. Typically 600
        V, 1200 V or 1500 V

    safety_factor : float
        safety factor for calculating string length as a fraction of max Voc.
        An example value wuold be 0.023, corresponding to a safety factor of
        2.3%. Safety factors are only used for 690.7(A)(1) standards.

    Returns
    -------

    voc_summary : dataframe

        Dataframe containing fields:

        'max_module_voltage' - the maximum module voltage (not including
        safety factor).

        'string_design_voltage' - Maximum allowable string voltage for the
        design, in V. Typically 600 V, 1200 V or 1500 V

        'safety_factor' - safety factor for calculating string length as a
        fraction of max Voc. An example value wuold be 0.023, corresponding
        to a safety factor of 2.3%. Safety factors are only used for 690.7(A)(1)
        standards.

        'string_length' - Longest acceptable string length.

        'Cell Temperature' - Temperature


    """

    voc_summary = pd.DataFrame(
        columns=['Conditions', 'max_module_voltage', 'string_design_voltage',
                 'safety_factor',
                 'string_length',
                 'Cell Temperature', 'POA Irradiance', 'long_note'],
        index=['690.7(A)(3)-P99.5',
               '690.7(A)(3)-P100',
               '690.7(A)(1)-DAY',
               '690.7(A)(1)-NSRDB',
               '690.7(A)(1)-ASHRAE',
               '690.7(A)(2)-ASHRAE'])

    mean_yearly_min_temp = calculate_mean_yearly_min_temp(df.index,
                                                          df['temp_air'])

    if type(ashrae) == type(pd.DataFrame()):
        ashrae_loc = vocmax.ashrae_get_design_conditions_at_loc(
            info['Latitude'], info['Longitude'], ashrae)
        lowest_expected_temperature_ashrae = ashrae_loc[
            'Extreme_Annual_Mean_Min_DB']
    else:
        ashrae_available = ashrae_is_design_conditions_available()

        if ashrae_available:
            ashrae = ashrae_get_design_conditions()
            ashrae_loc = vocmax.ashrae_get_design_conditions_at_loc(
                info['Latitude'], info['Longitude'], ashrae)
            lowest_expected_temperature_ashrae = ashrae_loc[
                'Extreme_Annual_Mean_Min_DB']
        else:
            lowest_expected_temperature_ashrae = np.nan

    # mean_yearly_min_temp_ashrae =
    mean_yearly_min_day_temp = calculate_mean_yearly_min_temp(
        df.index[df['ghi'] > 150],
        df['temp_air'][df['ghi'] > 150])

    voc_summary['safety_factor'] = 0
    for f in ['690.7(A)(3)-P99.5', '690.7(A)(3)-P100']:
        voc_summary.loc[f, 'safety_factor'] = safety_factor

    # Calculate some standard voc values.
    voc_values = {
        '690.7(A)(3)-P99.5': np.percentile(np.array(df['v_oc']), 99.5),
        '690.7(A)(3)-P100': df['v_oc'].max(),
        '690.7(A)(1)-DAY': calculate_voc(1000, mean_yearly_min_day_temp,
                                         module_parameters),
        '690.7(A)(1)-ASHRAE': calculate_voc(1000,
                                            lowest_expected_temperature_ashrae,
                                            module_parameters),
        '690.7(A)(1)-NSRDB': calculate_voc(1000, mean_yearly_min_temp,
                                           module_parameters),
        '690.7(A)(2)-ASHRAE': module_parameters['Voco'] * nec_correction_factor(
            lowest_expected_temperature_ashrae),

    }
    conditions = {
        '690.7(A)(3)-P99.5': 'P99.5 Voc',
        '690.7(A)(3)-P100': 'Historical Maximum Voc',
        '690.7(A)(1)-NSRDB': 'Voc at 1 sun and mean yearly min ambient temperature from NSRDB',
        '690.7(A)(1)-ASHRAE': 'Voc at 1 sun and mean yearly min ambient temperature from ASHRAE',
        '690.7(A)(2)-ASHRAE': 'NEC 690.7a2 Voc, corrected by correction factor',
        '690.7(A)(1)-DAY': 'Voc at 1 sun and mean yearly minimum daytime (GHI>150 W/m2) temperature',
        # 'Norm_P99.5': 'P99.5 Voc assuming module normal to sun',
    }

    s_p99p5 = get_temp_irradiance_for_voc_percentile(df, percentile=99.5)
    s_p100 = get_temp_irradiance_for_voc_percentile(df, percentile=100,
                                                    cushion=0.0001)
    cell_temp = {
        '690.7(A)(3)-P99.5': s_p99p5['temp_cell'],
        '690.7(A)(3)-P100': s_p100['temp_cell'],
        '690.7(A)(1)-DAY': mean_yearly_min_day_temp,
        '690.7(A)(1)-NSRDB': mean_yearly_min_temp,
        '690.7(A)(1)-ASHRAE': lowest_expected_temperature_ashrae,
        '690.7(A)(2)-ASHRAE': lowest_expected_temperature_ashrae,

    }
    poa_irradiance = {
        '690.7(A)(3)-P99.5': s_p99p5['effective_irradiance'],
        '690.7(A)(3)-P100': s_p100['effective_irradiance'],
        '690.7(A)(1)-DAY': 1000,
        '690.7(A)(1)-NSRDB': 1000,
        '690.7(A)(1)-ASHRAE': 1000,
        '690.7(A)(2)-ASHRAE': 1000,
    }

    voc_summary['max_module_voltage'] = voc_summary.index.map(voc_values)
    voc_summary['Conditions'] = voc_summary.index.map(conditions)
    voc_summary['string_design_voltage'] = string_design_voltage
    voc_summary['POA Irradiance'] = voc_summary.index.map(poa_irradiance)
    voc_summary['Cell Temperature'] = voc_summary.index.map(cell_temp)

    voc_summary['string_length'] = voc_summary['max_module_voltage'].map(
        lambda x: voc_to_string_length(x, string_design_voltage, safety_factor))

    max_module_voltage_with_safety_factor = voc_summary[
                                                'max_module_voltage'] * (
                                                        1 + voc_summary[
                                                    'safety_factor'])

    mean_yearly_min_temp = calculate_mean_yearly_min_temp(df.index,
                                                          df['temp_air'])
    long_note = {
        '690.7(A)(3)-P99.5': "99.5 Percentile Voc<br>" + \
                             "690.7(A)(3)-P99.5: {:.3f} V<br>".format(
                                 voc_values['690.7(A)(3)-P99.5']) + \
                             "690.7(A)(3)-P99.5 + {:1.1%} SF: {:.3f} V<br>".format(
                                 voc_summary['safety_factor'][
                                     '690.7(A)(3)-P99.5'],
                                 max_module_voltage_with_safety_factor[
                                     '690.7(A)(3)-P99.5']) + \
                             "Maximum String Length: {:.0f}<br>".format(
                                 voc_summary['string_length'][
                                     '690.7(A)(3)-P99.5']) + \
                             "Recommended 690.7(A)(3) value for string length.",

        '690.7(A)(3)-P100': 'Historical maximum Voc from {:.0f}-{:.0f}<br>'.format(
            df['year'][0], df['year'][-1]) + \
                            '690.7(A)(3)-P100: {:.3f}<br>'.format(
                                voc_values['690.7(A)(3)-P100']) + \
                            "690.7(A)(3)-P100 + {:1.1%} SF: {:.3f} V<br>".format(
                                voc_summary['safety_factor'][
                                    '690.7(A)(3)-P100'],
                                max_module_voltage_with_safety_factor[
                                    '690.7(A)(3)-P100']) + \
                            'Maximum String Length: {:.0f}<br>'.format(
                                voc_summary['string_length'][
                                    '690.7(A)(3)-P100']) + \
                            'Conservative 690.7(A)(3) value for string length.',

        '690.7(A)(1)-DAY': 'Traditional daytime Voc, using 1 sun irradiance and<br>' + \
                           'mean yearly minimum daytime (GHI>150 W/m^2) dry bulb temperature of {:.1f} C.<br>'.format(
                               mean_yearly_min_day_temp) + \
                           'Day Voc: {:.3f} V<br>'.format(
                               voc_values['690.7(A)(1)-DAY']) + \
                           'Maximum String Length:{:.0f}<br>'.format(
                               voc_summary['string_length'][
                                   '690.7(A)(1)-DAY']) + \
                           'Recommended 690.7(A)(1) Value',

        '690.7(A)(1)-NSRDB': 'Traditional 690.7(A)(1) value, using 1 sun irradiance and<br>' + \
                             'mean yearly minimum dry bulb temperature of {:.1f} C.<br>'.format(
                                 mean_yearly_min_temp) + \
                             '690.7(A)(1)-NSRDB: {:.3f}<br>'.format(
                                 voc_values['690.7(A)(1)-NSRDB']) + \
                             'Maximum String Length: {:.0f}'.format(
                                 voc_summary['string_length'][
                                     '690.7(A)(1)-NSRDB']),
        '690.7(A)(1)-ASHRAE': 'Traditional 690.7(A)(1) value<br>' + \
                              'using 1 sun irradiance and<br>' + \
                              'mean yearly minimum dry bulb temperature of {:.1f} C.<br>'.format(
                                  lowest_expected_temperature_ashrae) + \
                              'Trad-ASHRAE-690.7a1 Voc: {:.3f}<br>'.format(
                                  voc_values['690.7(A)(1)-ASHRAE']) + \
                              'Maximum String Length: {:.0f}'.format(
                                  voc_summary['string_length'][
                                      '690.7(A)(1)-ASHRAE']),
        '690.7(A)(2)-ASHRAE': 'Traditional 690.7(A)(2) value<br>' + \
                              'using NEC derating table and<br>' + \
                              'mean yearly minimum dry bulb temperature of {:.1f} C.<br>'.format(
                                  lowest_expected_temperature_ashrae) + \
                              'Trad-ASHRAE-690.7(A)(2) Voc: {:.3f}<br>'.format(
                                  voc_values['690.7(A)(2)-ASHRAE']) + \
                              'Maximum String Length: {:.0f}'.format(
                                  voc_summary['string_length'][
                                      '690.7(A)(2)-ASHRAE']),
        # 'Norm_P99.5': "Normal Voc, 99.5 percentile Voc value<br>".format(voc_values['Norm_P99.5']) +\
        #               "assuming array always oriented normal to sun.<br>" +\
        #               "Norm_P99.5 Voc: {:.3f}<br>".format(voc_values['Norm_P99.5']) +\
        #               "Maximum String Length: {:.0f}".format(voc_summary['string_length']['Norm_P99.5'])
    }
    short_note = {
        '690.7(A)(3)-P99.5': "Recommended 690.7(A)(3) value for string length.",

        '690.7(A)(3)-P100': 'Conservative 690.7(A)(3) value for string length.',

        '690.7(A)(1)-DAY': 'Traditional design using daytime temp (GHI>150 W/m^2)',

        '690.7(A)(1)-ASHRAE': 'Traditional design using ASHRAE and temperature coefficient',
        '690.7(A)(1)-NSRDB': 'Traditional design using NSRDB and temperature coefficient',
        '690.7(A)(2)-ASHRAE': 'Traditional design using ASHRAE and standard derating.',

        # 'Norm_P99.5': ""
    }

    voc_summary['long_note'] = voc_summary.index.map(long_note)
    voc_summary['short_note'] = voc_summary.index.map(short_note)

    return voc_summary


def get_s3_csv(filename):
    """


    """
    import boto3

    # filename = '2017DesignConditions_s.xlsx.csv'

    bucket = 'pvtools-nsrdb-pickle'

    # connect to AWS S3
    s3 = boto3.resource('s3')

    obj = s3.Object(bucket, filename)

    df = pd.read_csv(obj.get()['Body'])

    return df


def scale_to_hours_per_year(y, info):
    return y / info['timedelta_in_years'] * info['interval_in_hours']


def make_voc_histogram(df, info, number_bins=400):
    # Voc histogram
    voc_hist_y_raw, voc_hist_x_raw = np.histogram(df['v_oc'],
                                                  bins=np.linspace(
                                                      df['v_oc'].max() * 0.6,
                                                      df['v_oc'].max() + 1,
                                                      number_bins))

    voc_hist_y = scale_to_hours_per_year(voc_hist_y_raw, info)[1:]
    voc_hist_x = voc_hist_x_raw[1:-1]

    return voc_hist_x, voc_hist_y


def make_simulation_summary(df, info, module_parameters, racking_parameters,
                            thermal_model, string_design_voltage, safety_factor,
                            ashrae='local_load'):
    """

    Makes a text summary of the simulation.

    Parameters
    ----------
    info
    module_parameters
    racking_parameters
    max_string_voltage

    Returns
    -------

    """

    voc_summary = make_voc_summary(df, info, module_parameters,
                                   string_design_voltage=string_design_voltage,
                                   safety_factor=safety_factor,
                                   ashrae=ashrae)

    if type(thermal_model) == type(''):
        thermal_model = {'Model parameters': thermal_model}

    if 'Location ID' in info:
        info['Location_ID'] = info['Location ID']
    if 'Time Zone' in info:
        info['local_time_zone'] = info['Time Zone']

    # extra_parameters = calculate_extra_module_parameters(module_parameters)
    voc_hist_x, voc_hist_y = make_voc_histogram(df, info, number_bins=200)

    pd.DataFrame({'Voc': voc_hist_x, 'hours per year': voc_hist_y}).to_csv(
        index=False)

    summary = \
        'Simulation Run Date,' + str(datetime.datetime.now()) + '\n\n' + \
        'Weather data,\n' + \
        pd.Series(info)[
            ['Source', 'Latitude', 'Longitude', 'Location_ID',
             'local_time_zone',
             'Elevation', 'Version', 'interval_in_hours',
             'timedelta_in_years']].to_csv(header=False) + '\n' + \
        'Module Parameters\n' + \
        pd.Series(module_parameters).to_csv(header=False) + '\n' + \
        'Racking Parameters\n' + \
        pd.Series(racking_parameters).to_csv(header=False) + '\n' + \
        'Thermal model\n' + \
        'model type, Sandia\n' + \
        pd.Series(thermal_model).to_csv(header=False) + '\n' + \
        'String Design Voltage,' + str(string_design_voltage) + '\n' + \
        'vocmaxlib Version,' + __version__ + '\n' + \
        '\nKey Voc Values\n' + \
        voc_summary.to_csv() + \
        '\nVoc Histogram\n' + \
        pd.DataFrame(
            {'Voc': voc_hist_x,
             'hours per year': voc_hist_y}
        ).to_csv(index=False)

    return summary


def calculate_normal_voc(poa_direct, poa_diffuse, temp_cell, module_parameters,
                         spectral_loss=1, aoi_loss=1, FD=1):
    """

    Parameters
    ----------
    poa_direct
    poa_diffuse
    temp_cell
    module_parameters
    spectral_loss
    aoi_loss
    FD

    Returns
    -------

    """
    effective_irradiance = calculate_effective_irradiance(
        poa_direct,
        poa_diffuse,
        spectral_loss=spectral_loss,
        aoi_loss=aoi_loss,
        FD=FD
    )
    v_oc = calculate_voc(effective_irradiance, temp_cell,
                         module_parameters)
    return v_oc


# def calculate_effective_irradiance_bifacial(poa_direct_front,
#                                             poa_diffuse_front,
#                                             poa_direct_back,
#                                             spectral_loss_front=1,
#                                             spectral_loss_back=1,
#                                             aoi_loss_front=1,
#                                             FD_front=1):
#     """
#
#     Parameters
#     ----------
#     poa_direct
#     poa_diffuse
#     spectral_loss
#     aoi_loss
#     FD
#
#     Returns
#     -------
#         effective_irradiance in W/m^2
#
#     """
#     # See pvlib.pvsystem.sapm_effective_irradiance for source of this line:
#     effective_irradiance = spectral_loss_front * (
#             poa_direct_front * aoi_loss_front + FD_front * poa_diffuse_front) + \
#         spectral_loss_back*poa_back
#
#     return effective_irradiance


def calculate_effective_irradiance(poa_direct, poa_diffuse, spectral_loss=1,
                                   aoi_loss=1, FD=1):
    """

    Parameters
    ----------
    poa_direct
    poa_diffuse
    spectral_loss
    aoi_loss
    FD

    Returns
    -------
        effective_irradiance in W/m^2

    """
    # See pvlib.pvsystem.sapm_effective_irradiance for source of this line:
    effective_irradiance = spectral_loss * (
            poa_direct * aoi_loss + FD * poa_diffuse)
    return effective_irradiance


def calculate_voc(effective_irradiance, temp_cell, module,
                  reference_temperature=25,
                  reference_irradiance=1000):
    """

    Standard reference conditions are 1000 W/m2 and 25 C.

    Parameters
    ----------
    effective_irradiance
        Irradiance in W/m^2
    temperature
    module_parameters
        Dict or Series containing the fields:

        'alpha_sc': The short-circuit current temperature coefficient of the
        module in units of A/C.

        'a_ref': The product of the usual diode ideality factor (n,
        unitless), number of cells in series (Ns), and cell thermal voltage
        at reference conditions, in units of V

        'I_L_ref': The light-generated current (or photocurrent) at reference
        conditions, in amperes.

        'I_o_ref': The dark or diode reverse saturation current at reference
        conditions, in amperes.

        'R_sh_ref': The shunt resistance at reference conditions, in ohms.

        'R_s': The series resistance at reference conditions, in ohms.

        'Adjust': The adjustment to the temperature coefficient for short
        circuit current, in percent.


        model : str
        Model to use, can be 'cec' or 'desoto'
        XX



    Returns
    -------

    References
    ----------

    [1] A. Dobos, “An Improved Coefficient Calculator for the California
    Energy Commission 6 Parameter Photovoltaic Module Model”, Journal of
    Solar Energy Engineering, vol 134, 2012.

    """

    if (not 'iv_model' in module) or module['iv_model'] == 'sapm':

        v_oc = sapm_voc(effective_irradiance, temp_cell, module,
                        reference_temperature=reference_temperature,
                        reference_irradiance=reference_irradiance)



    elif module['iv_model'] in ['cec', 'desoto']:

        photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = \
            calcparams_singlediode(effective_irradiance, temp_cell, module)

        # out = pvlib.pvsystem.singlediode(photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth,
        #                            method='newton')

        v_oc = pvlib.singlediode.bishop88_v_from_i(0,
                                                   photocurrent,
                                                   saturation_current,
                                                   resistance_series,
                                                   resistance_shunt,
                                                   nNsVth,
                                                   method='newton')

    else:
        raise Exception('iv_model not recognized')

    return v_oc


def singlediode_voc(effective_irradiance, temp_cell, module_parameters):
    """
    Calculate voc using the singlediode model.

    Parameters
    ----------
    effective_irradiance
    temp_cell
    module_parameters

    Returns
    -------

    """

    photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = \
        calcparams_singlediode(effective_irradiance, temp_cell,
                               module_parameters)

    # out = pvlib.pvsystem.singlediode(photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth,
    #                            method='newton')

    v_oc = pvlib.singlediode.bishop88_v_from_i(0,
                                               photocurrent,
                                               saturation_current,
                                               resistance_series,
                                               resistance_shunt,
                                               nNsVth,
                                               method='newton')
    return v_oc


def sapm_voc(effective_irradiance, temp_cell, module, reference_temperature=25,
             reference_irradiance=1000):
    """

    This function differs from the PVLIB version in that the effective
    irradiance is in W/m2.

    Parameters
    ----------
    effective_irradiance : numeric
        Effective irradiance in W/m^2

    temp_cell : numeric

    module : dict
        parameters are:

        'Voco'

        'cells_in_series'

        'Bvoco'

        'Mbvoc'

    reference_temperature : float

    reference_irradiance : float

    Returns
    -------

    """

    T0 = reference_temperature
    q = 1.60218e-19  # Elementary charge in units of coulombs
    kb = 1.38066e-23  # Boltzmann's constant in units of J/K

    # avoid problem with integer input
    Ee = np.array(effective_irradiance, dtype='float64')

    # set up masking for 0, positive, and nan inputs
    Ee_gt_0 = np.full_like(Ee, False, dtype='bool')
    Ee_eq_0 = np.full_like(Ee, False, dtype='bool')
    notnan = ~np.isnan(Ee)
    np.greater(Ee, 0, where=notnan, out=Ee_gt_0)
    np.equal(Ee, 0, where=notnan, out=Ee_eq_0)

    # Bvmpo = module['Bvmpo'] + module['Mbvmp'] * (1 - Ee)
    if 'Mbvoc' in module:
        Bvoco = module['Bvoco'] + module['Mbvoc'] * (1 - Ee)
    else:
        Bvoco = module['Bvoco']
    delta = module['n_diode'] * kb * (temp_cell + 273.15) / q

    # avoid repeated computation
    logEe = np.full_like(Ee, np.nan)
    np.log(Ee / reference_irradiance, where=Ee_gt_0, out=logEe)
    logEe = np.where(Ee_eq_0, -np.inf, logEe)

    # avoid repeated __getitem__
    cells_in_series = module['cells_in_series']

    v_oc = np.maximum(0, (
            module['Voco'] + cells_in_series * delta * logEe +
            Bvoco * (temp_cell - T0)))

    return v_oc


def sapm_temperature_to_get_voc(effective_irradiance,
                                Voc,
                                Voco,
                                Bvoco,
                                diode_factor,
                                cells_in_series,
                                Mbvoc=0,
                                reference_temperature=25,
                                reference_irradiance=1000
                                ):
    """

    Calculate the cell temperature to achieve a certain Voc at a value of
    effective irradiance.

    Parameters
    ----------
    effective_irradiance
    Voc
    Voco
    Bvoco
    diode_factor
    cells_in_series
    Mbvoc
    reference_temperature
    reference_irradiance

    Returns
    -------

    """

    T0 = reference_temperature
    q = 1.60218e-19  # Elementary charge in units of coulombs
    kb = 1.38066e-23  # Boltzmann's constant in units of J/K

    # avoid problem with integer input
    Ee = np.array(effective_irradiance, dtype='float64')

    # set up masking for 0, positive, and nan inputs
    Ee_gt_0 = np.full_like(Ee, False, dtype='bool')
    Ee_eq_0 = np.full_like(Ee, False, dtype='bool')
    notnan = ~np.isnan(Ee)
    np.greater(Ee, 0, where=notnan, out=Ee_gt_0)
    np.equal(Ee, 0, where=notnan, out=Ee_eq_0)

    # avoid repeated computation
    logEe = np.full_like(Ee, np.nan)
    np.log(Ee / reference_irradiance, where=Ee_gt_0, out=logEe)
    logEe = np.where(Ee_eq_0, -np.inf, logEe)

    Bvoco = Bvoco + Mbvoc * (1 - Ee)
    delta_ref = diode_factor * kb * (reference_temperature + 273.15) / q
    delta_prime = diode_factor * kb / q

    temperature_cell = reference_temperature + (Voc - Voco - cells_in_series * delta_ref * logEe) / (
                cells_in_series * delta_prime * logEe + Bvoco)
    return temperature_cell


def sapm_mpp(effective_irradiance, temperature, module_parameters):
    photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = \
        calcparams_singlediode(effective_irradiance, temperature,
                               module_parameters)

    i_mp, v_mp, p_mp = pvlib.singlediode.bishop88_mpp(photocurrent,
                                                      saturation_current,
                                                      resistance_series,
                                                      resistance_shunt, nNsVth,
                                                      method='newton')

    return i_mp, v_mp, p_mp


def calcparams_singlediode(effective_irradiance, temperature,
                           module_parameters):
    # Default to desoto model.
    if not 'iv_model' in module_parameters.keys():
        module_parameters['iv_model'] = 'desoto'

    if module_parameters['iv_model'] == 'sapm':
        module_parameters['iv_model'] = 'desoto'

    if module_parameters['iv_model'] == 'desoto':
        photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = \
            pvlib.pvsystem.calcparams_desoto(effective_irradiance,
                                             temperature,
                                             module_parameters['alpha_sc'],
                                             module_parameters['a_ref'],
                                             module_parameters['I_L_ref'],
                                             module_parameters['I_o_ref'],
                                             module_parameters['R_sh_ref'],
                                             module_parameters['R_s']
                                             )
    elif module_parameters['iv_model'] == 'cec':
        photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = \
            pvlib.pvsystem.calcparams_cec(effective_irradiance,
                                          temperature,
                                          module_parameters['alpha_sc'],
                                          module_parameters['a_ref'],
                                          module_parameters['I_L_ref'],
                                          module_parameters['I_o_ref'],
                                          module_parameters['R_sh_ref'],
                                          module_parameters['R_s'],
                                          module_parameters['Adjust'],
                                          )

    else:
        raise Exception(
            "module_parameters['iv_model'] must be 'cec' or 'desoto'")

    return photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth


def calculate_iv_curve(effective_irradiance, temperature, module_parameters,
                       ivcurve_pnts=200):
    """

    :param effective_irradiance:
    :param temperature:
    :param module_parameters:
    :param ivcurve_pnts:
    :return:
    """
    photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = \
        calcparams_singlediode(effective_irradiance, temperature,
                               module_parameters)

    iv_curve = pvlib.pvsystem.singlediode(photocurrent, saturation_current,
                                          resistance_series, resistance_shunt,
                                          nNsVth,
                                          ivcurve_pnts=ivcurve_pnts,
                                          method='lambertw')

    return iv_curve


def calculate_sapm_module_parameters(module_parameters,
                                     reference_irradiance=1000,
                                     reference_temperature=25):
    """

    Calculate standard parameters of modules from the single diode model.

    module_parameters: dict

    Returns

    Dict of parameters including:

         'Voco' - open circuit voltage at STC.

         'Bvoco' - temperature coefficient of Voc near STC, in V/C

         Isco - short circuit current at STC

         alpha_sc - temperature coefficient of Isc near STC, in A/C

         Vmpo - voltage at maximum power point at STC, in V

         Pmpo - power at maximum power point at STC, in W

         Impo - current at maximum power point at STC, in A

         Bpmpo - temperature coefficient of maximum power near STC, in W/C


    """
    param = {}
    param['cells_in_series'] = module_parameters['N_s']

    kB = 1.381e-23
    q = 1.602e-19
    Vthref = kB * (273.15 + 25) / q

    param['n_diode'] = module_parameters['a_ref'] / (
            module_parameters['N_s'] * Vthref)

    # Calculate Voc vs. temperature for finding coefficients
    temp_cell_smooth = np.linspace(reference_temperature - 5,
                                   reference_temperature + 5, 5)

    photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = \
        calcparams_singlediode(effective_irradiance=reference_irradiance,
                               temperature=temp_cell_smooth,
                               module_parameters=module_parameters)
    iv_points = pvlib.pvsystem.singlediode(photocurrent,
                                           saturation_current,
                                           resistance_series, resistance_shunt,
                                           nNsVth)

    photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = \
        calcparams_singlediode(
            effective_irradiance=reference_irradiance,
            temperature=reference_temperature,
            module_parameters=module_parameters)

    iv_points_0 = pvlib.pvsystem.singlediode(photocurrent,
                                             saturation_current,
                                             resistance_series,
                                             resistance_shunt, nNsVth)

    param['Voco'] = iv_points_0['v_oc']
    # param['Voco'] = module_parameters['V_oc_ref']
    param['Isco'] = iv_points_0['i_sc']
    # param['Isco'] = module_parameters['I_sc_ref']
    param['Impo'] = iv_points_0['i_mp']
    param['Vmpo'] = iv_points_0['v_mp']
    param['Pmpo'] = iv_points_0['p_mp']
    # param['Ixo'] = iv_points_0['i_x']
    # param['Ixxo'] = iv_points_0['i_xx']

    voc_fit_coeff = np.polyfit(temp_cell_smooth, iv_points['v_oc'], 1)
    param['Bvoco'] = voc_fit_coeff[0]

    pmp_fit_coeff = np.polyfit(temp_cell_smooth, iv_points['p_mp'], 1)
    param['Bpmpo'] = pmp_fit_coeff[0]

    isc_fit_coeff = np.polyfit(temp_cell_smooth, iv_points['i_sc'], 1)
    param['alpha_sc'] = isc_fit_coeff[0]

    param['Mbvoc'] = 0
    param['FD'] = 1

    param['iv_model'] = 'sapm'

    # description = {
    #     'Voco':'Open circuit voltage at STC (V)',
    #     'Isco':'Short circuit current at STC (A)',
    #     'Impo':'Max power current at STC (A)',
    #     'Vmpo':'Max power voltage at STC (V)',
    #     'Pmpo':'Max power power at STC (W)',
    #     'Bvoco':'Temperature coeff. of open circuit voltage near STC (V/C)',
    #     'Bpmpo':'Temperature coeff. of max power near STC (W/C)',
    #     'Bisco':'Tempearture coeff. of short circuit current near STC (A/C)',
    #     'cells_in_series': 'Number of cells in series',
    #     'n_diode': 'diode ideality factor',
    #
    # }
    #
    # sapm_module = pd.DataFrame(
    #             index= list(param.keys()),
    #              columns=['Parameter','Value','Description'])
    #
    # sapm_module['Parameter'] = sapm_module.index
    # sapm_module['Value'] = sapm_module.index.map(param)
    # sapm_module['Description'] = sapm_module.index.map(description)
    #

    return param


def cec_to_sapm(module, reference_irradiance=1000, reference_temperature=25):
    """

    Parameters
    ----------
    module : dict or series
        CEC module parameters.

        'alpha_sc': temperature coefficient of short-circuit current near
        reference conditions, in A/C.


    reference_irradiance
    reference_temperature

    Returns
    -------
    sapm : dict or series

        'alpha_sc': temperature coefficient of short-circuit current near
        reference conditions, in A/C


    """
    # Calculate sapm parameters.
    sapm = calculate_sapm_module_parameters(module,
                                            reference_irradiance=reference_irradiance,
                                            reference_temperature=reference_temperature)

    # Replace certain parameters with those explicitly specified.
    if np.abs((sapm['Bvoco'] - module['beta_oc']) / sapm['Bvoco']) > 0.25:
        warnings.warn(
            'Inconsistency found in Bvoco, suggest to check datasheet. ')
    sapm['Bvoco'] = module['beta_oc']

    sapm['Voco'] = module['V_oc_ref']
    sapm['Isco'] = module['I_sc_ref']
    sapm['Impo'] = module['I_mp_ref']
    sapm['Vmpo'] = module['V_mp_ref']

    # Note that
    alpha_sc_Amp_per_C = module['alpha_sc']
    if np.abs(
            (sapm['alpha_sc'] - alpha_sc_Amp_per_C) / sapm['alpha_sc']) > 0.25:
        warnings.warn(
            "From Desoto model, find alpha_sc = {:1.3f} A/C, but value in CEC database is {:1.3f} A/C".format(
                sapm['alpha_sc'], alpha_sc_Amp_per_C))
    sapm['alpha_sc'] = alpha_sc_Amp_per_C

    sapm['Pmpo'] = module['I_mp_ref'] * module['V_mp_ref']

    sapm['efficiency'] = module['I_mp_ref'] * \
                         module['V_mp_ref'] / \
                         module['A_c'] / 1000
    return sapm


#     raise Warning('Inconsistency found in Bvoco, suggest to check datasheet. ')

#
#
# def calculate_sapm_module_parameters_df(module_parameters,reference_irradiance=1000,
#                                           reference_temperature=25):
#     """
#
#     Calculate standard parameters of modules from the single diode model.
#
#     module_parameters: dict
#
#     Returns
#
#     Dict of parameters including:
#
#          'Voco' - open circuit voltage at STC.
#
#          'Bvoco' - temperature coefficient of Voc near STC, in V/C
#
#          Isco - short circuit current at STC
#
#          Bisco - temperature coefficient of Isc near STC, in A/C
#
#          Vmpo - voltage at maximum power point at STC, in V
#
#          Pmpo - power at maximum power point at STC, in W
#
#          Impo - current at maximum power point at STC, in A
#
#          Bpmpo - temperature coefficient of maximum power near STC, in W/C
#
#
#     """
#
#     param = calculate_sapm_module_parameters(module_parameters,
#                                         reference_irradiance=reference_irradiance,
#                                           reference_temperature=reference_temperature)
#
#     description = {
#         'Voco':'Open circuit voltage at STC (V)',
#         'Isco':'Short circuit current at STC (A)',
#         'Impo':'Max power current at STC (A)',
#         'Vmpo':'Max power voltage at STC (V)',
#         'Pmpo':'Max power power at STC (W)',
#         'Bvoco':'Temperature coeff. of open circuit voltage near STC (V/C)',
#         'Bpmpo':'Temperature coeff. of max power near STC (W/C)',
#         'Bisco':'Tempearture coeff. of short circuit current near STC (A/C)'
#     }
#
#     extra_parameters = pd.DataFrame(
#                 index= list(param.keys()),
#                  columns=['Parameter','Value','Description'])
#
#     extra_parameters['Parameter'] = extra_parameters.index
#     extra_parameters['Value'] = extra_parameters.index.map(param)
#     extra_parameters['Description'] = extra_parameters.index.map(description)
#
#
#     return extra_parameters

def calculate_mean_yearly_min_temp(datetimevec, temperature):
    """

    Calculate the mean of the yearly minimum temperatures.

    Parameters
    ----------
    datetimevec
        datetime series giving times corresponding to the temperatures
    temperature
        series of temperatures
    Returns
    -------
        mean of yearly minimum temperatures.
    """
    years = list(set(datetimevec.year))
    yearly_min_temp = []

    for j in years:
        yearly_min_temp.append(
            temperature[datetimevec.year == j].min()
        )

    return np.mean(yearly_min_temp)


def get_temp_irradiance_for_voc_percentile(df, percentile=99.5, cushion=0.0025):
    """

    Find the lowest temperature and associated irradiance that produces the
    percentile value of Voc.

    Parameters
    ----------
    df : dataframe
        Dataframe containing 'v_oc'

    percentile

    cushion : numeric


    Returns
    -------
    Series
        Lowest
    """

    Pvoc = np.nanpercentile(np.array(df['v_oc']), percentile,
                            interpolation='nearest')
    df_close = df[df['v_oc'] > Pvoc * (1 - cushion)]
    df_close = df_close[df_close['v_oc'] < Pvoc * (1 + cushion)]

    if len(df_close['temp_air']) > 0:
        i_near = df_close['temp_air'].idxmin()
    else:
        i_near = abs(df['v_oc'] - Pvoc).idxmin()

    return df.iloc[df.index.get_loc(i_near)]


def voc_to_string_length(voc, max_string_voltage, safety_factor):
    """

    Returns the maximum number N modules with open circuit voltage voc that
    do not exceed max_string_voltage when connected in series.

    Parameters
    ----------
    voc : float
        Open circuit voltage
    max_string_voltage : float
        Maximum string voltage

    safety_factor : float
        safety factor for string length.

    Returns
    -------
    N : float
        Maximum string length

    """

    if voc == 0:
        string_length = np.nan
    else:
        string_length = np.round(
            np.floor(max_string_voltage * (1 - safety_factor) / voc))

    return string_length


def simulate_system_sandia(weather, info, module_parameters=None,
                           system_parameters=None):
    """
    Use the PVLIB Sandia model to calculate max voc.


    :param weather:
    :param info:
    :param module_parameters:
    :param system_parameters:
    :return:
    """

    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    inverter_parameters = cec_inverters[
        'Power_Electronics__FS1700CU15__690V__690V__CEC_2018_']

    # Set location
    location = pvlib.location.Location(latitude=info['Latitude'][0],
                                       longitude=info['Longitude'][0])

    # Weather must have field dni, dhi, ghi, temp_air, and wind_speed.

    # Make pvsystem

    if system_parameters['mount_type'].lower() == 'fixed_tilt':
        system = pvlib.pvsystem.PVSystem(
            module_parameters=module_parameters,
            inverter_parameters=inverter_parameters,
            surface_tilt=system_parameters['surface_tilt'],
            surface_azimuth=system_parameters['surface_azimuth'],
        )
    elif system_parameters['mount_type'].lower() == 'single_axis_tracker':
        system = pvlib.tracking.SingleAxisTracker(
            module_parameters=module_parameters,
            inverter_parameters=inverter_parameters,
            axis_tilt=system_parameters['axis_tilt'],
            axis_azimuth=system_parameters['axis_azimuth'],
            max_angle=system_parameters['max_angle'],
            backtrack=system_parameters['backtrack'],
            gcr=system_parameters['ground_coverage_ratio']
        )

    # print(system_parameters['surface_tilt'])

    mc = pvlib.modelchain.ModelChain(system, location)
    mc.system.racking_model = system_parameters['racking_model']

    # mc.complete_irradiance(times=weather.index, weather=weather)
    mc.run_model(times=weather.index, weather=weather)

    df = weather
    df['v_oc'] = mc.dc.v_oc
    df['temp_cell'] = mc.temps['temp_cell']

    return (df, mc)


def import_nsrdb_csv(filename):
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
    year = str(df['Year'][0])

    if np.diff(df[0:2].Minute) == 30:
        interval = '30'
        info['interval_in_hours'] = 0.5
        df = df.set_index(
            pd.date_range('1/1/{yr}'.format(yr=year), freq=interval + 'Min',
                          periods=60 * 24 * 365 / int(interval)))
    elif df['Minute'][1] - df['Minute'][0] == 0:
        interval = '60'
        info['interval_in_hours'] = 1
        df = df.set_index(
            pd.date_range('1/1/{yr}'.format(yr=year), freq=interval + 'Min',
                          periods=60 * 24 * 365 / int(interval)))
    else:
        print('Interval not understood!')

    df.index = df.index.tz_localize(
        pytz.FixedOffset(float(info['Time Zone'] * 60)))

    return (df, info)


# df, info = import_csv('nsrdb_1degree_uv/104_30.97_-83.22_tmy.csv')

def import_nsrdb_sequence(folder):
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

    if len(files) == 0:
        raise ValueError('No input files found in directory')
    files.sort()
    df = pd.DataFrame()
    for f in files:
        print(f)
        (df_temp, info) = import_nsrdb_csv(f)

        df = df.append(df_temp)

    info['timedelta_in_years'] = (df.index[-1] - df.index[0]).days / 365

    return (df, info)
