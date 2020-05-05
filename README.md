# vocmax
Calculate the maximum sting size for a photovoltaic installation. The method is consistent with the NEC 2017 690.7 standard.  

# Summary
One key design decision for photovoltaic (PV) power plants is to select the string size, the number of PV modules connected in series. Longer strings tend to lower total system costs, but the string size must still meet relevant electrical standards to ensure that the maximum system voltage remains less than the design voltage. Conventional methods calculate string size using the temperature coefficient of open-circuit voltage (Voc) assuming that the coldest-expected temperature occurs simultaneously with a full-sun irradiance of 1000 W/m^2. Here, we demonstrate that this traditional method is unnecessarily conservative, resulting in a string size that is ~10% shorter than necessary to maintain system voltage within limits. Instead, we suggest to calculate string size by modeling Voc over time using historical weather data, a method in compliance with the 2017 National Electric Code. We demonstrate that this site-specific modeling procedure is in close agreement with data from field measurements. Furthermore, we perform a comprehensive sensitivity and uncertainty analysis to identify an appropriate safety factor for this method. By using site-specific modeling instead of conventional methods, the levelized cost of electricity is reduced by up to ~1.2%, an impressive improvement attainable just by reorganizing strings. The method is provided as an easy-to-use [web tool](https://pvtools.lbl.gov/string-length-calculator) and an open-source Python package (vocmax) for the PV community. 

# Files
- **example_vocmax_calculation.py** - Script for calculating maximum string length. Start here!
- **vocmax/main.py** - vocmax main functions.
- **vocmax/NSRDB_sample/** - an example set of NSRDB data files for running sample calculation. You may want to download your own for the location of interest.
- **vocmax05_compress_database.py** - Script used to compress NSRDB csv files into a python pickle.


# Install

The vocmax library can be installed with pip:
```bash
pip install vocmax
```

This package depends on the following packages:
- pvlib
- pandas
- numpy
- pvfactors (for bifacial modeling)
- matplotlib

# Examples


## Full Example String Size calculation

The following code runs a standard string size calculation. This file is saved in the repository as 'example_vocmax_calculation.py'.


```python
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
import time

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
module['aoi_model'] = 'ashrae'
module['ashrae_iam_param'] = 0.05
module['is_bifacial'] = False
module['efficiency'] = 0.18
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
    # Module efficiency, unitless
    'efficiency': 0.15,
    # Diode Ideality Factor, unitless
    'n_diode': 1.2,
    # Fracion of diffuse irradiance used by the module.
    'FD': 1,
    # Whether the module is bifacial
    'is_bifacial': True,
    # Ratio of backside to frontside efficiency for bifacial modules. Only used if 'is_bifacial'==True
    'bifaciality_factor': 0.7,
    # AOI loss model
    'aoi_model':'ashrae',
    # AOI loss model parameter.
    'ashrae_iam_param': 0.05
    }


is_cec_module = 'a_ref' in module
print('\n** Module parameters **')
print(pd.Series(module))

# ------------------------------------------------------------------------------
# Choose Racking Method
# ------------------------------------------------------------------------------

# Racking parameters for single axis tracking (fixed tilt parameters are below).
racking_parameters = {
    # Racking type, can be 'single_axis' or 'fixed_tilt'
    'racking_type': 'single_axis',
    # The tilt of the axis of rotation with respect to horizontal, in degrees
    'axis_tilt': 0,
    # Compass direction along which the axis of rotation lies. Measured in
    # degrees East of North
    'axis_azimuth': 0,
    # Maximum rotation angle of the one-axis tracker from its horizontal
    # position, in degrees.
    'max_angle': 90,
    # Controls whether the tracker has the capability to “backtrack” to avoid
    # row-to-row shading. False denotes no backtrack capability. True denotes
    # backtrack capability.
    'backtrack': True,
    # A value denoting the ground coverage ratio of a tracker system which
    # utilizes backtracking; i.e. the ratio between the PV array surface area
    # to total ground area.
    'gcr': 2.0 / 7.0,
    # Bifacial model can be 'proportional' or 'pvfactors'
    'bifacial_model': 'proportional',
    # Proportionality factor determining the backside irradiance as a fraction
    # of the frontside irradiance. Only used if 'bifacial_model' is
    # 'proportional'.
    'backside_irradiance_fraction': 0.2,
    # Ground albedo
    'albedo': 0.25
}

# Example racking parameters for fixed tilt (only use one racking_parameters,
# comment the other one out!)
"""
racking_parameters = {
    'racking_type': 'fixed_tilt',
    # Tilt of modules from horizontal.
    'surface_tilt': 30,
    # 180 degrees orients the modules towards the South.
    'surface_azimuth': 180,
    # Ground albedo
    'albedo':0.25
}
"""

# Additionally, here is an example set of racking parameters for full bifacial
# modeling. Make sure 'is_bifacial' is True in the module parameters. Full
# bifacial modeling takes about 10 minutes depending on the exact configuration.
# See documentation for pvfactors for additional description of parameters.
"""
racking_parameters = {
    # Racking type, can be 'single_axis' or 'fixed_tilt'
    'racking_type': 'single_axis',
    # The tilt of the axis of rotation with respect to horizontal, in degrees
    'axis_tilt': 0,
    # Compass direction along which the axis of rotation lies. Measured in
    # degrees East of North
    'axis_azimuth': 0,
    # Maximum rotation angle of the one-axis tracker from its horizontal
    # position, in degrees.
    'max_angle': 90,
    # Controls whether the tracker has the capability to “backtrack” to avoid
    # row-to-row shading. False denotes no backtrack capability. True denotes
    # backtrack capability.
    'backtrack': True,
    # A value denoting the ground coverage ratio of a tracker system which
    # utilizes backtracking; i.e. the ratio between the PV array surface area
    # to total ground area.
    'gcr': 2.0 / 7.0,
    # Ground albedo
    'albedo':0.25,
    # bifacial model can be 'pfvactors' or 'simple'
    'bifacial_model': 'pvfactors',
    # number of pv rows
    'n_pvrows': 3,
    # Index of row to use backside irradiance for
    'index_observed_pvrow': 1,
    # height of pvrows (measured at center / torque tube)
    'pvrow_height': 1,
    # width of pvrows
    'pvrow_width': 1,
    # azimuth angle of rotation axis
    'axis_azimuth': 0.,
    # pv row front surface reflectivity
    'rho_front_pvrow': 0.01,
    # pv row back surface reflectivity
    'rho_back_pvrow': 0.03,
    # Horizon band angle.
    'horizon_band_angle': 15,
    'run_parallel_calculations': True,
    'n_workers_for_parallel_calcs': -1,
}
"""

# Sandia thermal model can be a string for using default coefficients or the
# parameters can be set manually. Parameters are described in [1].
#
# [1] D.L. King, W.E. Boyson, J.A. Kratochvill. Photovoltaic Array Performance
# Model. Sand2004-3535 (2004).

thermal_model = {
    'named_model': 'open_rack_cell_glassback',
    # Temperature of open circuit modules is higher, specify whether to include
    # this effect.
    'open_circuit_rise': True
     }
# Or can set thermal model coefficients manually:
"""
thermal_model = {
    'named_model': 'explicit',
    'a':-3.56,
    'b':-0.075,
    'deltaT':3,
    'open_circuit_rise':True
}
"""

print('\n** Racking parameters **')
print(pd.Series(racking_parameters))

# ------------------------------------------------------------------------------
# Max string length
# ------------------------------------------------------------------------------

# Max allowable string voltage, for determining string length. Typically this
# number is determined by the inverter.
string_design_voltage = 1500

# ------------------------------------------------------------------------------
# Import weather data
# ------------------------------------------------------------------------------

# Get the weather data.
print("\nImporting weather data...")

# Define the lat, lon of the location (this location is preloaded and does not
# require an API key)
lat, lon = 37.876, -122.247
# Get an NSRDB api key for any point but the preloaded one (this api key will
# not work, you need to get your own which will look like it.)
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
t0 = time.time()
df = vocmax.simulate_system(weather,
                               info,
                               module,
                               racking_parameters,
                               thermal_model)
print('Simulation time: {:1.2f}'.format(time.time()-t0))
# Calculate max power voltage, only possible if using CEC database for module parameters.
if is_cec_module:
    _, df['v_mp'], _ = vocmax.sapm_mpp(df['effective_irradiance'],
                          df['temp_cell'],
                          module)

# ------------------------------------------------------------------------------
# Calculate String Size
# ------------------------------------------------------------------------------

# Get ASHRAE design temperature:
ashrae = vocmax.ashrae_get_data_at_loc(lat,lon)

# Look up weather data uncertainty safety factor at the point of interest.
temperature_error = vocmax.get_nsrdb_temperature_error(info['Latitude'],info['Longitude'])

# Calculate weather data safety factor using module Voc temperature coefficient
Beta_Voco_fraction = np.abs(module['Bvoco'])/module['Voco']
weather_data_safety_factor = temperature_error*Beta_Voco_fraction

# Calculate propensity for extreme temperature fluctuations.
extreme_cold_delta_T = vocmax.calculate_mean_yearly_min_temp(df.index,df['temp_air']) - df['temp_air'].min()

# Compute safety factor for extreme cold temperatures
extreme_cold_safety_factor = extreme_cold_delta_T*Beta_Voco_fraction

# Add up different contributions to obtain an overall safety factor
safety_factor = weather_data_safety_factor + 0.016
print('Total Safety Factor: {:1.1%}'.format(safety_factor))

# Calculate string length.
voc_summary = vocmax.make_voc_summary(df, info, module,
                                string_design_voltage=string_design_voltage,
                                safety_factor=safety_factor)

print('Simulation complete.')

# Make a csv file for saving simulation parameters
summary_text = vocmax.make_simulation_summary(df, info,
                                                 module,
                                                 racking_parameters,
                                                 thermal_model,
                                                 string_design_voltage,
                                                 safety_factor)

# Save the summary csv to file.
summary_file = 'out.csv'
with open(summary_file,'w') as f:
    f.write(summary_text)

print('\n** Voc Results **')
print(voc_summary[[ 'max_module_voltage', 'safety_factor','string_length',
                 'Cell Temperature', 'POA Irradiance']].to_string())

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

for key in voc_summary.index:
    plt.plot(voc_summary['max_module_voltage'][key] * np.array([1,1]), [0,10],
             label=key)
plt.show()
plt.legend()


# Plot IV curve
if is_cec_module:
    plt.figure(3)
    plt.clf()
    for j in range(len(iv_curve)):
        plt.plot(iv_curve[j]['v'], iv_curve[j]['i'])

    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.grid()

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




```





## Set module parameters
Module parameters are set using a dictionary. 
```python
sapm_parameters = {'cells_in_series': 96,
                   'n_diode': 1.2,
                   'Voco': 69.7015,
                   'Bvoco': -0.159,
                   'FD': 1}
```

## Get weather data

The following code will download weather data from the [national solar radiation database (NSRDB)](https://nsrdb.nrel.gov/) for the lat/lon coordinate.
```python
import vocmax
# Define the lat, lon of the location and the year (this one is preloaded)
lat, lon = 37.876, -122.247

# You must request an NSRDB api key from the link above
api_key = 'apsdofijasdpafkjweo21u09u1082h8h2d2d' # not a real key -- get your own!

weather, info = vocmax.get_weather_data(lat,lon,api_key=api_key)
```

Another possibility is to download data directly from the NSRDB map viewer.


## Get NSRDB safety factor
The safety factor to use depends on location, here is how to look it up.
```python
import vocmax

# Define the lat, long of the location
lat, lon = 37.876, -91

# Find the max temperature error for the location
temperature_error = vocmax.get_nsrdb_temperature_error(lat,lon)

# Temperature coefficient of Voc divided by Voco in 1/C.
temperature_coefficient_of_voc = 0.0035

# Find the safety factor
safety_factor = temperature_error*temperature_coefficient_of_voc

print('Safety Factor for NSRDB weather data: {:.2%}'.format(safety_factor))
```

## Load ASHRAE data

Due to copyright, the ASHRAE design conditions filemust be purchased separately, directly from ASHRAE. 
The weather data viewer DVD, version 6.0 is available at: https://www.techstreet.com/ashrae/standards/weather-data-viewer-dvd-version-6-0?ashrae_auth_token=12ce7b1d-2e2e-472b-b689-8065208f2e36&product_id=1949790

Within this DVD is a file titled "2017DesignConditions_s.xlsx" One way to load this file is to place it in the current directory. 

An exmaple of loading the ASHRAE dataset is

```python
import vocmax
ashrae = vocmax.ashrae_get_design_conditions()
```


# Todo

- Change the voc summary to be Vmax rather than Voc.

# Copyright notice

String Length Calculator Copyright (c) 2020, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject to receipt of 
any required approvals from the U.S. Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.
