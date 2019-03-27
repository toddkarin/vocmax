"""
Example max voc calculation using PVLIB modelchain method.

"""

import numpy as np
import pvlib
import nsrdbtools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import vocmaxlib

# modules = pvlib.pvsystem.retrieve_sam('cecmod')

sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
# module_parameters = sandia_modules['SunPower_SPR_315E_WHT__2007__E__']
module_parameters = sandia_modules['Canadian_Solar_CS5P_220M___2009_']

# Inverter doesn't matter for Voc but is needed to run modelchain.
cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
inverter_parameters = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']

# Get the weather data.
df, info = nsrdbtools.import_sequence('NSRDB_sample')

# Set location
location = pvlib.location.Location(latitude=info['Latitude'][0],
                                   longitude=info['Longitude'][0])

# Rename the weather data for input to PVLIB.
weather = pd.DataFrame.from_dict({
    'dni':df['DNI'],
    'dhi':df['DHI'],
    'ghi':df['GHI'],
    'temp_air':df['Temperature'],
    'wind_speed':df['Wind Speed']})

# Panel tilt.
surface_tilt = info['Latitude'][0]
surface_azimuth=180
system = pvlib.pvsystem.PVSystem(module_parameters=module_parameters,
                                 inverter_parameters=inverter_parameters,
                                 surface_tilt=surface_tilt,
                                 surface_azimuth=surface_azimuth,
                                 )

# Create the model system.
mc = pvlib.modelchain.ModelChain(system, location)

print('Running...')
# Run the model
mc.run_model(times=weather.index, weather=weather)
print('done.')

mean_yearly_min_ambient_temp = vocmaxlib.mean_yearly_min_temp(weather.index, weather['temp_air'])



# Calculate Standard Voc values.
#
#
# #
# years = list(set(weather.index.year))
# yearly_min_temp = []
# yearly_min_daytime_temp = []
# for j in years:
#     yearly_min_temp.append(
#         weather[weather.index.year == j]['temp_air'].min())
#     yearly_min_daytime_temp.append(
#         weather[weather.index.year == j]['temp_air'][
#             weather[weather.index.year == j]['ghi'] > 150].min()
#     )
# mean_yearly_min_ambient_temp = np.mean(yearly_min_temp)
# mean_yearly_min_daytime_ambient_temp = np.mean(yearly_min_daytime_temp)

# # min_daytime_temp = df['temp_air'][df['ghi']>150].min()
#
# voc_1sun_min_temp = mc.system.sapm(1, mean_yearly_min_ambient_temp)['v_oc']
# voc_1sun_min_daytime_temp = \
# mc.system.sapm(1, mean_yearly_min_daytime_ambient_temp)['v_oc']
#
# voc_dni_cell_temp = \
# # mc.system.sapm((df['dni'] + df['dhi']) / 1000, df['temp_cell'])['v_oc'].max()
# voc_P99p9 = np.percentile(
#     df['v_oc'][np.logical_not(np.isnan(df['v_oc']))],
#     99.9)
# voc_P99 = np.percentile(df['v_oc'][np.logical_not(np.isnan(df['v_oc']))],
#                         99)



# Calculate some standard values for max voc.
voc_list = {
    '1 Sun, min air temp': system.sapm(1,weather['temp_air'].min())['v_oc'],
    '1 sun, min daytime air temp': system.sapm(1,weather['temp_air'][weather['ghi']>200].min())['v_oc'],
    '1 sun, min daytime cell temp': system.sapm(1,mc.temps['temp_cell'][weather['ghi']>200].min())['v_oc'],
    'absolute max Voc': mc.dc.v_oc.max()
 }

print('Plotting...')
fig_width = 4
fig_height = 6

max_pos = np.argmax(np.array(mc.dc.v_oc[:]))
plot_width = 300

plt.figure(0,figsize=(fig_width,fig_height))
plt.clf()

plt.plot(mc.dc.v_oc.index[max_pos-plot_width:max_pos+plot_width], mc.dc.v_oc[max_pos-plot_width:max_pos+plot_width])
ylims = np.array(plt.ylim())
plt.plot([ mc.dc.v_oc.index[max_pos],mc.dc.v_oc.index[max_pos]] , ylims)

plt.show()


# Make figures
plt.figure(1,figsize=(fig_width,fig_height))
plt.clf()
y,c = np.histogram(mc.dc.v_oc,
                   bins=np.linspace(mc.dc.v_oc.max()*0.75,mc.dc.v_oc.max()+1,600))

y_scale = y/info.timedelta_in_years[0]*info.interval_in_hours[0]
plt.plot(c[2:],y_scale[1:])
plt.xlabel('Voc (Volts)')
plt.ylabel('hrs/year')

p_values = np.array([99.5,98])
v_oc_percentile = np.percentile(mc.dc.v_oc,p_values)

n=0
for v in voc_list:
    plt.plot(voc_list[v] * np.array([1, 1]), [0, 10],
             label=v)
    plt.text(voc_list[v],12,v + ', {:.2f} V, N={:.1f}'.format(voc_list[v],
                                                  1500 /voc_list[v]),
             rotation=90,
             verticalalignment='bottom')
    n=n+1


for j in range(len(v_oc_percentile)):
    plt.plot(v_oc_percentile[j]*np.array([1,1]), [0,10])
    plt.text(v_oc_percentile[j],12,
             'P{:.1f} Voc, {:.2f} V, N={:.1f}'.format(p_values[j],
                                                  v_oc_percentile[j],
                                                  1500 / v_oc_percentile[j]),
             rotation=90,
             verticalalignment='bottom')
# plt.legend()
plt.ylim([0,40])
plt.show()
