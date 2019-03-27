import numpy as np
import pvlib
import nsrdbtools
import socket
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import pandas as pd

# Parameters entering into Voc calculation:
# surface_azimuth
# surface_tilt
# Bvoco
# weather
# Voco
# Cells_in_Series
# N (Diode ideality factor)
# Latitude
# Longitude
# Air mass coefficients.

# modules = pvlib.pvsystem.retrieve_sam('cecmod')

def simulate_system(weather,info, module_paramet0ers=None,system_parameters=None):

    # sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    # module_parameters = sandia_modules['SunPower_SPR_315E_WHT__2007__E__']
    # module_parameters = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    # module_parameters = sandia_modules[module_name]

    # Inverter doesn't matter for Voc but is needed to run the model.
    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    inverter_parameters = cec_inverters['Power_Electronics__FS1700CU15__690V__690V__CEC_2018_']

    # Get the weather data.
    # df, info = nsrdbtools.import_sequence('nsrdb_Fairfield')
    # df, info = nsrdbtools.import_sequence('nsrdb_single_location/New_York')
    # weather, info = nsrdbtools.import_csv('nsrdb_Fairfield/128364_38.29_-122.14_1998.csv')

    # Set location
    location = pvlib.location.Location(latitude=info['Latitude'][0],
                                       longitude=info['Longitude'][0])

    # Rename the weather data for the location of interest.
    # weather = pd.DataFrame.from_dict({
    #     'dni':df['DNI'],
    #     'dhi':df['DHI'],
    #     'ghi':df['GHI'],
    #     'temp_air':df['Temperature'],
    #     'wind_speed':df['Wind Speed']})

    # Panel tilt.
    # surface_tilt = system_parameters['surface_tilt']
    # surface_azimuth=180
    system = pvlib.pvsystem.PVSystem(module_parameters=module_parameters,
                                     inverter_parameters=inverter_parameters,
                                     surface_tilt=system_parameters['surface_tilt'],
                                     surface_azimuth=system_parameters['surface_azimuth'],
                                     )

    # print(system_parameters['surface_tilt'])

    mc = pvlib.modelchain.ModelChain(system, location)
    mc.system.racking_model = system_parameters['racking_model']

    # mc.complete_irradiance(times=weather.index, weather=weather)
    mc.run_model(times=weather.index, weather=weather)


    return mc

def calculate_max_voc(weather,info, module_parameters=None,system_parameters=None):


    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    inverter_parameters = cec_inverters['Power_Electronics__FS1700CU15__690V__690V__CEC_2018_']

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



#
# print(system)1
# # Calculate some standard values for max voc.
# voc_list = {'1 Sun, min air temp': system.sapm(1,weather['temp_air'].min())['v_oc'],
# '1 sun, min daytime air temp': system.sapm(1,weather['temp_air'][weather['ghi']>200].min())['v_oc'],
# '1 sun, min daytime cell temp': system.sapm(1,mc.temps['temp_cell'][weather['ghi']>200].min())['v_oc'],
# 'absolute max Voc': mc.dc.v_oc.max()
#  }
#
#

sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')


# Get filedata for
if socket.gethostname()[0:6]=='guests':
    filedata = nsrdbtools.inspect_pickle_database('/Users/toddkarin/Documents/NSRDB_pickle/')
    # filedata = nsrdbtools.inspect_pickle_database('NSRDB_pickle')

else:
    filedata = nsrdbtools.inspect_pickle_database('/var/www/FlaskApp/NSRDB_pickle/')

def get_sandia_module_dropdown_list():
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')

    sandia_module_dropdown_list = []
    for m in list(sandia_modules.keys()):
        sandia_module_dropdown_list.append(
            {'label': m.replace('_', ' '), 'value': m})

    return sandia_module_dropdown_list


def mean_yearly_min_temp(datetimevec, temperature):
    years = list(set(datetimevec.year))
    yearly_min_temp = []

    for j in years:
        yearly_min_temp.append(
            temperature[datetimevec.year == j].min()
        )


    return np.mean(yearly_min_temp)