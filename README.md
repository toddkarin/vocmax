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
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import vocmax

# ------------------------------------------------------------------------------
# Choose Module Parameters
# ------------------------------------------------------------------------------

# Option 1. If the module is in the CEC database, then can retieve parameters.
cec_modules = vocmax.cec_modules
cec_parameters = cec_modules['Jinko_Solar_JKM175M_72'].to_dict()
sapm_parameters = vocmax.calculate_sapm_module_parameters(cec_parameters)
# Calculate extra module parameters for your information:
module = {**sapm_parameters, **cec_parameters}

# Option 2. Or can build a dictionary of parameters manually. Note that in order
# to calculate MPP, it is necessary to include the CEC parameters: alpha_sc,
# a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s, and Adjust.
"""
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
"""

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

# Calculate max power voltage
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

# Calculate some IV curves.
irradiance_list = [200,400,600,800,1000]
iv_curve = []
for e in irradiance_list:
    ret = vocmax.calculate_iv_curve(e, 25, cec_parameters)
    ret['effective_irradiance'] = e
    iv_curve.append(ret)
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


## Get safety factor
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

print('Safety Factor for weather data: {:.2%}'.format(safety_factor))
```


# Todo

- Change the voc summary to be Vmax rather than Voc.