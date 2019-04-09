"""
Example max voc calculation using PVLIB. Uses singlediode model.

"""

import numpy as np
import pvlib
import nsrdbtools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import vocmaxlib
# import time

# ------------------------------------------------------------------------------
# Choose Module Parameters
# ------------------------------------------------------------------------------
# If the module is in the CEC database, then can retieve parameters.
cec_modules = vocmaxlib.cec_modules
module_parameters = cec_modules['1Soltech_1STH_215_P'].to_dict()
module_parameters['FD'] = 1

# Or can build a dictionary of paramaters manually, see:
# https://pvlib-python.readthedocs.io/en/latest/generated/pvlib.pvsystem.calcparams_cec.html

# aoi model determines how much radiation is lost from reflection. aoi_model can
# be 'ashrae' or 'no_loss'

# module_parameters['aoi_model'] = 'ashrae'
# module_parameters['ashrae_iam_param'] = 0.05
module_parameters['aoi_model'] = 'no_loss'

print('\n** Module parameters **')
print(pd.Series(module_parameters))

# ------------------------------------------------------------------------------
# Choose Racking Method
# ------------------------------------------------------------------------------

# Example racking parameters for single axis tracking.
racking_parameters = {
    'racking_type': 'single_axis',
    'axis_tilt': 0,
    'axis_azimuth': 0,
    'max_angle': 90,
    'backtrack': True,
    'gcr': 2.0 / 7.0

}

# Example racking parameters for fixed tilt (only use one racking_parameters,
# comment the other one out!)
racking_parameters = {
    'racking_type': 'fixed_tilt',
    'surface_tilt': 30,
    'surface_azimuth': 180
}

# Sandia thermal model can be a string for using default coefficients or the
# parameters can be set manually.
thermal_model = 'open_rack_cell_glassback'
# thermal_model = {'a':1,'b':1,'DT':3}

print('\n** Racking parameters **')
print(pd.Series(racking_parameters))

# ------------------------------------------------------------------------------
# Max string length
# ------------------------------------------------------------------------------

# Max allowable string voltage, for determining string length.
max_string_voltage = 1500

# ------------------------------------------------------------------------------
# Import weather data
# ------------------------------------------------------------------------------

# Get the weather data. Weather is loaded from NSRDB files placed in a
# directory.
print("\nImporting weather data...")
weather_data_directory = 'NSRDB_sample'
weather, info = nsrdbtools.import_sequence(weather_data_directory)

# Make sure that the weather data has the correct fields for pvlib.
weather = weather.rename(columns={'DNI':'dni','DHI':'dhi','GHI':'ghi',
                     'Temperature':'temp_air',
                     'Wind Speed':'wind_speed'})

# ------------------------------------------------------------------------------
# Simulate system
# ------------------------------------------------------------------------------

# Run the calculation.
print('Running Simulation...')
df = vocmaxlib.simulate_system(weather,
                               info,
                               module_parameters,
                               racking_parameters,
                               thermal_model)

voc_summary = vocmaxlib.make_voc_summary(df, module_parameters,
                                   max_string_voltage=max_string_voltage)

print('Simulation complete.')

# Make a csv file for saving simulation parameters
summary_text = vocmaxlib.make_simulation_summary(df, info,
                                                 module_parameters,
                                                 racking_parameters,
                                                 thermal_model,
                                                 max_string_voltage)

# Save file as summary_file
summary_file = 'out.csv'
with open(summary_file,'w') as f:
    f.write(summary_text)

print('\n** Voc Results **')
print(voc_summary.to_string())


# ------------------------------------------------------------------------------
# Plot results
# ------------------------------------------------------------------------------

pd.plotting.register_matplotlib_converters(explicit=True)
fig_width = 6
fig_height = 4

max_pos = np.argmax(np.array(df['v_oc']))
plot_width = 300


# Plot Voc vs. time
plt.figure(0,figsize=(fig_width,fig_height))
plt.clf()
plt.plot(df.index[max_pos-plot_width:max_pos+plot_width],
         df['v_oc'][max_pos-plot_width:max_pos+plot_width])
ylims = np.array(plt.ylim())
plt.plot([ df.index[max_pos],df.index[max_pos]] , ylims)

plt.show()

# Plot Voc histogram
plt.figure(1,figsize=(fig_width,fig_height))
plt.clf()
y,c = np.histogram(df['v_oc'],
                 bins=np.linspace(df['v_oc'].max()*0,df['v_oc'].max()*1.2,600))

y_scale = y/info['timedelta_in_years']*info['interval_in_hours']
plt.plot(c[2:],y_scale[1:])
plt.xlabel('Voc (Volts)')
plt.ylabel('hrs/year')


for j in voc_summary.index:
    plt.plot(voc_summary['v_oc'][j] * np.array([1,1]), [0,10],
             label=voc_summary['Conditions'][j])
    plt.text(voc_summary['v_oc'][j],12, j,
             rotation=90,
             verticalalignment='bottom',
             horizontalalignment='center')


plt.show()

