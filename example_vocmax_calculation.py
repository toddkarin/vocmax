"""

This script shows an example calculation for calculating the maximum
string length allowed in a particular location.

The method proceeds in the following steps

- Choose module parameters
- Choose racking method
- Set maximum allowable string voltage.
- Import weather data
- Run the calculation
- Plot.

"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import vocmax

# ------------------------------------------------------------------------------
# Choose Module Parameters
# ------------------------------------------------------------------------------

# Option 1. If the module is in the CEC database, then can retreive parameters.
"""
cec_modules = vocmax.cec_modules
cec_parameters = cec_modules['Jinko_Solar_JKM175M_72'].to_dict()
sapm_parameters = vocmax.calculate_sapm_module_parameters(cec_parameters)
# Calculate extra module parameters for your information:
module = {**sapm_parameters, **cec_parameters}
"""

# Option 2. Or can build a dictionary of parameters manually. Note that in order
# to calculate MPP, it is necessary to include the CEC parameters: alpha_sc,
# a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s, and Adjust.
module = {
    # Number of cells in series in each module.
    'cells_in_series': 60,
    # Open circuit voltage at reference conditions, in Volts.
    'Voco': 37.2,
    # Temperature coefficient of Voc, in Volt/C
    'Bvoco': -0.127,
    # Short circuit current, in Amp
    'Isco': 8.09,
    # Short circuit current temperature coefficient, in Amp/C
    'alpha_sc': 0.0036,
    # Diode Ideality Factor, unitless
    'n_diode': 1.2,
    # Fracion of diffuse irradiance used by the module.
    'FD': 1,
    # Whether the module is bifacial
    'is_bifacial': True,
    # Ratio of backside to frontside efficiency for bifacial modules. Only used if 'is_bifacial'==True
    'bifaciality_factor': 0.7,
    }


is_cec_module = 'a_ref' in module
print('\n** Module parameters **')
print(pd.Series(module))

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
    'gcr': 2.0 / 7.0,
    'bifacial_model': 'proportional',
    'backside_irradiance_fraction': 0.2,
    'albedo': 0.25

}

# Example racking parameters for fixed tilt (only use one racking_parameters,
# comment the other one out!)
"""
racking_parameters = {
    'racking_type': 'fixed_tilt',
    'surface_tilt': 30,
    'surface_azimuth': 180
}
"""

# Sandia thermal model can be a string for using default coefficients or the
# parameters can be set manually.
thermal_model = 'open_rack_cell_glassback'
# Or can set thermal model coefficients manually:
"""
thermal_model = {'a':1,'b':1,'DT':3}
"""

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

# Get the weather data.
print("\nImporting weather data...")

# Define the lat, lon of the location (this location is preloaded and does not
# require an API key)
lat, lon = 37.876, -122.247
# Get an NSRDB api key for any point but the preloaded one (this api key will
# not work, you need to get your own.)
api_key = 'BP2hICfC0ZQ2PT6h4xaU3vc4GAadf39fasdsPbZN'
# Get weather data (takes a few minutes, result is cached for quick second calls).
weather, info = vocmax.get_weather_data(lat,lon,api_key=api_key)

# Option 2: Get weather data from a series of NSRDB csv files.
"""
weather_data_directory = 'vocmax/NSRDB_sample'
weather, info = vocmax.import_nsrdb_sequence(weather_data_directory)
"""

# Make sure that the weather data has the correct fields for pvlib.
weather = weather.rename(columns={'DNI':'dni','DHI':'dhi','GHI':'ghi',
                     'Temperature':'temp_air',
                     'Wind Speed':'wind_speed'})


# ------------------------------------------------------------------------------
# Simulate system
# ------------------------------------------------------------------------------

# Run the calculation.
print('Running Simulation...')
df = vocmax.simulate_system(weather,
                               info,
                               module,
                               racking_parameters,
                               thermal_model)

# Calculate max power voltage, only possible if using CEC database for module parameters.
if is_cec_module:
    _, df['v_mp'], _ = vocmax.sapm_mpp(df['effective_irradiance'],
                          df['temp_cell'],
                          module)

# ------------------------------------------------------------------------------
# Calculate String Size
# ------------------------------------------------------------------------------

# Look up weather data uncertainty safety factor at the point of interest.
temperature_error = vocmax.get_nsrdb_temperature_error(
    info['Latitude'],info['Longitude'])

# Calculate weather data safety factor using module Voc temperature coefficient
weather_data_safety_factor = temperature_error*np.abs(
    module['Bvoco'])/module['Voco']

# Add up different contributions to obtain an overall safety factor
safety_factor = weather_data_safety_factor + 0.01

# Calculate string length.
voc_summary = vocmax.make_voc_summary(df, module,
                                   max_string_voltage=max_string_voltage,
                                safety_factor=safety_factor)

print('Simulation complete.')

# Make a csv file for saving simulation parameters
summary_text = vocmax.make_simulation_summary(df, info,
                                                 module,
                                                 racking_parameters,
                                                 thermal_model,
                                                 max_string_voltage,
                                                 safety_factor)

# Save the summary csv to file.
summary_file = 'out.csv'
with open(summary_file,'w') as f:
    f.write(summary_text)

print('\n** Voc Results **')
print(voc_summary.to_string())

# Calculate some IV curves if we are using CEC database.
if is_cec_module:
    irradiance_list = [200,400,600,800,1000]
    iv_curve = []
    for e in irradiance_list:
        ret = vocmax.calculate_iv_curve(e, 25, cec_parameters)
        ret['effective_irradiance'] = e
        iv_curve.append(ret)


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
plt.ylabel('Voc (V)')
plt.show()

# Plot Irradiance vs. time
plt.figure(11,figsize=(fig_width,fig_height))
plt.clf()
plt.plot(df.index[max_pos-plot_width:max_pos+plot_width],
         df['effective_irradiance'][max_pos-plot_width:max_pos+plot_width])
ylims = np.array(plt.ylim())
plt.plot([ df.index[max_pos],df.index[max_pos]] , ylims)
plt.ylabel('POA Irradiance')
plt.show()

# Plot Voc histogram
plt.figure(1,figsize=(fig_width,fig_height))
plt.clf()
voc_hist_x, voc_hist_y = vocmax.make_voc_histogram(df,info)

plt.plot(voc_hist_x, voc_hist_y)
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


# Plot IV curve
if is_cec_module:
    plt.figure(3)
    plt.clf()
    for j in range(len(iv_curve)):
        plt.plot(iv_curve[j]['v'], iv_curve[j]['i'])

    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.grid()

if is_cec_module:
    # Oerating voltage vs. time.
    plt.figure(4,figsize=(7.5,3.5))
    plt.clf()

    voltage_bins = np.linspace(60,110,400)
    dV = voltage_bins[1] - voltage_bins[0]
    voc_hist_y_raw, voc_hist_x_raw = np.histogram(df['v_oc']/module['Voco']*100,
                                                  bins=voltage_bins)

    voc_hist_y = vocmax.scale_to_hours_per_year(voc_hist_y_raw, info)[1:]
    voc_hist_x = voc_hist_x_raw[1:-1]

    vmp_hist_y_raw, vmp_hist_x_raw = np.histogram(df['v_mp']/module['Voco']*100,
                                                  bins=voltage_bins)

    vmp_hist_y = vocmax.scale_to_hours_per_year(vmp_hist_y_raw, info)[1:]
    vmp_hist_x = vmp_hist_x_raw[1:-1]


    # plt.plot(voc_hist_x, voc_hist_y)
    plt.plot(voltage_bins,0*voltage_bins,
             color=[0.5,0.5,0.5])
    plt.plot(vmp_hist_x, (0.99*vmp_hist_y + 0.01*voc_hist_y))
    # plt.yscale('log')

    plt.xlabel('Operating voltage/Voco (%)',fontsize=9)
    plt.ylabel('hours/year',fontsize=9)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)


    voc_summary.loc['P99.5 + safety factor',:] = voc_summary.loc['P99.5',:]
    voc_summary.loc['P99.5 + safety factor','v_oc'] = voc_summary.loc['P99.5 + safety factor','v_oc']*1.03

    voc_summary.loc['P99.5','color'] = 'C1'
    voc_summary.loc['P99.5 + safety factor','color'] = 'C1'
    voc_summary.loc['Trad','color'] = 'C3'
    voc_summary.loc['Hist','color'] = 'C2'
    voc_summary.loc['Day','color'] = 'C4'


    n = 1
    for j in voc_summary.index:
        if j in ['P99.5', 'P99.5 + safety factor']:
            line_y = [1,10]
            text_y = 13
        else:
            line_y = [1, 5]
            text_y = 8
        plt.plot(voc_summary['v_oc'][j]/module['V_oc_ref']*100 * np.array([1,1]), line_y,
                 label=voc_summary['Conditions'][j],
                 color= voc_summary.loc[j,'color'])
        plt.text(voc_summary['v_oc'][j]/module['V_oc_ref']*100,text_y, j,
                 rotation=90,
                 verticalalignment='bottom',
                 horizontalalignment='center',
                 color=voc_summary.loc[j,'color'],
                 fontsize=9)
        n=n+1

    plt.show()


# Scatter plot of Temperature/Irradiance where Voc is highest.
plt.figure(5)
plt.clf()
cax = df['v_oc']>np.percentile(df['v_oc'],99.9)
plt.plot(df.loc[:,'effective_irradiance'], df.loc[:,'temp_cell'],'.',
         label='all data')
plt.plot(df.loc[cax,'effective_irradiance'], df.loc[cax,'temp_cell'],'.',
         label='Voc>P99.9')
plt.xlabel('POA Irradiance (W/m^2)')
plt.ylabel('Cell Temperature (C)')
plt.legend()
# plt.xlim([0,1000])
plt.show()
