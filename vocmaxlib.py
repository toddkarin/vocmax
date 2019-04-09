import numpy as np
import pvlib
import nsrdbtools
import socket
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import pandas as pd

# Parameters entering into Voc calculation:

cec_modules = pvlib.pvsystem.retrieve_sam('CeCMod')

vocmaxlib_version = '0.1.0'

def simulate_system(weather, info, module_parameters, racking_parameters, thermal_model):
    """

    Parameters
    ----------
    weather
        Dataframe of weather data, includes fields
            'dni': Direct Normal Irradiance (W/m^2)
            'dhi': Diffuse Horizontal Irradiance (W/m^2)
            'ghi' Global Horizontal Irradiance (W/m^2)
            'temp_air': air temperature (C)
            'wind_speed': 10 m wind speed in (m/s)
    info
        Dictionary containing fields
            'Latitude': latitude in degrees
            'Longitude': longitude in degrees.
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
        circuit current, in percent.    racking_parameters
        dictionary

        'aoi_model': (optional). Model for angle of incidence losses (
        reflection losses), can be 'ashrae' or 'no_loss'. default is 'no_loss'.

        'ashrae_iam_param': (optional). If aoi_model is 'ashrae', specify the
        ashrae coefficient. Typical value is 0.05.


    racking_parameters

    thermal_model: string or dict
        If string, can be
            ‘open_rack_cell_glassback’ (default)
            ‘roof_mount_cell_glassback’
            ‘open_rack_cell_polymerback’
            ‘insulated_back_polymerback’
            ‘open_rack_polymer_thinfilm_steel’
            ‘22x_concentrator_tracker’

        If dict, supply the following parameters:

            a: float

            SAPM module parameter for establishing the upper limit for module
            temperature at low wind speeds and high solar irradiance.

            b :float

            SAPM module parameter for establishing the rate at which the
            module temperature drops as wind speed increases (see SAPM eqn.
            11).

            deltaT :float

            SAPM module parameter giving the temperature difference between
            the cell and module back surface at the reference irradiance, E0.

    Returns
    -------

    """

    # Rename the weather data for input to PVLIB.
    if np.all([c in weather.columns for c in ['dni','dhi','ghi','temp_air',
                                              'wind_speed','year','month','day','hour','minute']]):
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
                     'Year':'year',
                     'Month':'month',
                     'Day':'day',
                     'Hour':'hour',
                     'Minute':'minute'})

    # Set location
    location = pvlib.location.Location(latitude=info['Latitude'],
                                       longitude=info['Longitude'])

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

    # Todo: Why haydavies?
    total_irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt,
        surface_azimuth,
        solar_position['zenith'],
        solar_position['azimuth'],
        weather['dni'],
        weather['ghi'],
        weather['dhi'],
        model='haydavies',
        dni_extra=dni_extra)

    if racking_parameters['racking_type'] == 'fixed_tilt':
        aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
                                   solar_position['zenith'],
                                   solar_position['azimuth'])
    elif racking_parameters['racking_type'] == 'single_axis':
        aoi = single_axis_vals['aoi']
    else:
        raise Exception('Racking type not understood')
        # aoi = single_axis_vals['aoi']

    airmass = location.get_airmass(solar_position=solar_position)
    temps = pvlib.pvsystem.sapm_celltemp(total_irrad['poa_global'],
                                         weather['wind_speed'],
                                         weather['temp_air'],
                                         thermal_model)

    # Spectral loss is typically very small on order of a few percent, assume no
    # spectral loss for simplicity
    spectral_loss = 1


    if not 'aoi_model' in module_parameters:
        module_parameters['aoi_model'] = 'no_loss'
    if not 'FD' in module_parameters:
        module_parameters['FD'] = 1

    # AOI loss:
    if module_parameters['aoi_model'] == 'no_loss' :
        aoi_loss = 1
    elif module_parameters['aoi_model'] == 'ashrae':
        aoi_loss = pvlib.pvsystem.ashraeiam(aoi,
                                            b=module_parameters[
                                                'ashrae_iam_param'])
    else:
        raise Exception('aoi_model must be ashrae or no_loss')

    effective_irradiance = calculate_effective_irradiance(
        total_irrad['poa_direct'],
        total_irrad['poa_diffuse'],
        aoi_loss=aoi_loss,
        FD=module_parameters['FD']
        )
    v_oc = calculate_voc(effective_irradiance, temps['temp_cell'],
                                   module_parameters)
    df = weather.copy()
    df['aoi'] = aoi
    df['aoi_loss'] = aoi_loss
    df['temp_cell'] = temps['temp_cell']
    df['v_oc'] = v_oc

    return df


def make_voc_summary(df,module_parameters,max_string_voltage=1500):
    """

    Parameters
    ----------
    df
    module_parameters
    max_string_voltage

    Returns
    -------

    """


    voc_summary = pd.DataFrame(
        columns=['Conditions', 'v_oc', 'max_string_voltage', 'string_length','long_note'],
        index=['P99.5', 'Norm_P99.5', 'Hist', 'Trad','Day'])

    mean_yearly_min_temp = calculate_mean_yearly_min_temp(df.index,df['temp_air'])
    mean_yearly_min_day_temp = calculate_mean_yearly_min_temp(df.index[df['ghi']>150],
                                              df['temp_air'][df['ghi']>150])

    # Calculate some standard voc values.
    voc_values = {
        'Hist': df['v_oc'].max(),
        'Trad': calculate_voc(1, mean_yearly_min_temp,
                                        module_parameters),
        'Day': calculate_voc(1, mean_yearly_min_day_temp,
                                        module_parameters),
        'Norm_P99.5':
            np.percentile(
                calculate_normal_voc(df['dni'],
                                               df['dhi'],
                                               df['temp_air'],
                                               module_parameters)
                , 99.5),
        'P99.5': np.percentile(df['v_oc'], 99.5),
    }
    conditions = {
        'P99.5': 'P99.5 v_oc',
        'Hist': 'Historical Maximum v_oc',
        'Trad': 'v_oc at 1 sun and mean yearly min ambient temperature',
        'Day': 'v_oc at 1 sun and mean yearly minimum daytime (GHI>150 W/m2) temperature',
        'Norm_P99.5': 'P99.5 v_oc assuming module normal to sun',
    }


    voc_summary['v_oc'] = voc_summary.index.map(voc_values)
    voc_summary['Conditions'] = voc_summary.index.map(conditions)
    voc_summary['max_string_voltage'] = max_string_voltage
    voc_summary['string_length'] = voc_summary['v_oc'].map(
        lambda x: voc_to_string_length(x, max_string_voltage))

    mean_yearly_min_temp = calculate_mean_yearly_min_temp(df.index, df['temp_air'])
    long_note = {
        'P99.5': "99.5 Percentile Voc<br>" + \
                 "P99.5 V_oc: {:.3f} V<br>".format(voc_values['P99.5']) +\
                 "Maximum String Length: {:.0f}<br>".format(voc_summary['string_length']['P99.5']) +\
                 "Recommended 690.7(A)(3) value for string length.",

        'Hist': 'Historical maximum Voc from {:.0f}-{:.0f}<br>'.format(df['year'][0], df['year'][-1]) +\
                'Hist V_oc: {:.3f}<br>'.format(voc_values['Hist']) + \
                'Maximum String Length: {:.0f}<br>'.format(voc_summary['string_length']['Hist']) + \
                'Conservative value for string length.',

        'Day': 'Traditional daytime Voc, using 1 sun irradiance and<br>' +\
                'mean yearly minimum daytime (GHI>150 W/m^2) dry bulb temperature of {:.1f} C.<br>'.format(mean_yearly_min_day_temp) +\
                'Trad V_oc: {:.3f} V<br>'.format(voc_values['Day']) +\
                'Maximum String Length:{:.0f}<br>'.format(voc_summary['string_length']['Trad']) +\
                'Recommended 690.7(A)(1) Value',

        'Trad': 'Traditional Voc, using 1 sun irradiance and<br>' +\
                'mean yearly minimum dry bulb temperature of {:.1f} C.<br>'.format(mean_yearly_min_temp) +\
                'Trad V_oc: {:.3f}<br>'.format(voc_values['Trad']) +\
                'Maximum String Length: {:.0f}'.format(voc_summary['string_length']['Trad']),

        'Norm_P99.5': "Normal Voc, 99.5 percentile Voc value<br>".format(voc_values['Norm_P99.5']) +\
                      "assuming array always oriented normal to sun.<br>" +\
                      "Norm_P99.5 V_oc: {:.3f}<br>".format(voc_values['Norm_P99.5']) +\
                      "Maximum String Length: {:.0f}".format(voc_summary['string_length']['Norm_P99.5'])
    }

    voc_summary['long_note'] = voc_summary.index.map(long_note)

    return voc_summary


def make_simulation_summary(df, info,module_parameters,racking_parameters,
                            thermal_model,max_string_voltage):
    """

    Parameters
    ----------
    info
    module_parameters
    racking_parameters
    max_string_voltage

    Returns
    -------

    """

    voc_summary = make_voc_summary(df, module_parameters, max_string_voltage=max_string_voltage)

    if type(thermal_model)==type(''):
        thermal_model = {'Model parameters': thermal_model}

    if 'Location ID' in info:
        info['Location_ID'] = info['Location ID']
    if 'Time Zone' in info:
        info['local_time_zone'] = info['Time Zone']

    summary = \
        'Weather data,\n' + \
        pd.Series(info)[
            ['Source', 'Latitude', 'Longitude', 'Location_ID', 'local_time_zone',
             'Elevation', 'Version', 'interval_in_hours',
             'timedelta_in_years']].to_csv(header=False) +  '\n' + \
        'Module Parameters\n' + \
        pd.Series(module_parameters).to_csv(header=False) + '\n' + \
        'Racking Parameters\n' + \
        pd.Series(racking_parameters).to_csv(header=False) +  '\n' + \
        'Thermal model\n' + \
        'model type, Sandia\n' + \
        pd.Series(thermal_model).to_csv(header=False) + '\n' + \
        'Max String Voltage,' + str(max_string_voltage) + '\n' + \
        'vocmaxlib Version,' + vocmaxlib_version + '\n' + \
        '\nKey Voc Values\n' + \
        voc_summary.to_csv()

    return summary


def calculate_normal_voc(poa_direct, poa_diffuse, temp_cell, module_parameters,
                         spectral_loss=1,aoi_loss=1,FD=1):
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


def calculate_effective_irradiance(poa_direct, poa_diffuse, spectral_loss=1,
                                   aoi_loss=1, FD=1,reference_irradiance=1000):
    """

    Parameters
    ----------
    poa_direct
    poa_diffuse
    spectral_loss
    aoi_loss
    FD
    reference_irradiance

    Returns
    -------

    """
    # See pvlib.pvsystem.sapm_effective_irradiance for source of this line:
    effective_irradiance = spectral_loss * (
            poa_direct * aoi_loss + FD * poa_diffuse) / reference_irradiance
    return effective_irradiance


def calculate_voc(effective_irradiance, temperature, module_parameters):
    """

    Reference conditions are 1000 W/m2 and 25 C.

    Parameters
    ----------
    effective_irradiance
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


    Returns
    -------

    References
    ----------

    [1] A. Dobos, “An Improved Coefficient Calculator for the California
    Energy Commission 6 Parameter Photovoltaic Module Model”, Journal of
    Solar Energy Engineering, vol 134, 2012.

    """
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


def voc_to_string_length(voc,max_string_voltage):
    """

    Parameters
    ----------
    voc
    max_string_voltage

    Returns
    -------

    """
    return np.round(np.floor(max_string_voltage/voc))
