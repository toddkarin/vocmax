"""
vocmaxlib

Module for performing maximum string voltage calculations.

toddkarin
"""


import numpy as np
import pvlib
# import nsrdbtools
# import socket
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import pandas as pd

# Parameters entering into Voc calculation:

cec_modules = pvlib.pvsystem.retrieve_sam('CeCMod')

vocmaxlib_version = '0.1.1'

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

def simulate_system(weather, info, module_parameters,
                    racking_parameters, thermal_model):
    """

    Use the PVLIB SAPM model to calculate maximum Voc.

    Parameters
    ----------
    weather : Dataframe
        Weather data
            'dni': Direct Normal Irradiance (W/m^2)
            'dhi': Diffuse Horizontal Irradiance (W/m^2)
            'ghi' Global Horizontal Irradiance (W/m^2)
            'temp_air': air temperature (C)
            'wind_speed': 10 m wind speed in (m/s)

    info : dict
        Dictionary containing location information with fields:
            'Latitude': latitude in degrees
            'Longitude': longitude in degrees.

    module_parameters : dict
        Dict or Series containing the fields diescribing the module

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
            circuit current, in percent. This parameter is only used if model
            is 'cec'.

            'aoi_model': (optional). Model for angle of incidence losses (
            reflection losses), can be 'ashrae' or 'no_loss'. default is
            'no_loss'.

            'ashrae_iam_param': (optional). If aoi_model is 'ashrae', specify
            the ashrae coefficient. Typical value is 0.05.

            'model' : str
                Method to use for calculation, can be 'cec' or 'desoto'

    racking_parameters : dict
        dictionary describing the racking setup. Contains fields

            'racking_type' : str. Can be 'fixed_tilt' for a stationary PV system
            or 'single_axis' for a single axis tracker.

            'surface_tilt' : float. If racking_type is 'fixed_tilt', specify the
            surface tilt in degrees from horizontal.

            'surface_azimuth' : float. If racking type is 'surface_azimuth', specify
            the racking azimuth in degrees. A value of 180 degrees has the
            module face oriented due South.

            'axis_tilt' : float. If racking_type is 'single_axis', specify the the
            tilt of the axis of rotation (i.e, the y-axis defined by
            axis_azimuth) with respect to horizontal, in decimal degrees.
            Standard value is 0.

            'axis_azimuth' : float. If racking_type is 'single_axis', specify a
            value denoting the compass direction along which the axis of
            rotation lies. Measured in decimal degrees East of North.
            Standard value is 0.

            'backtrack' : bool. Controls whether the tracker has the
            capability to ''backtrack'' to avoid row-to-row shading. False
            denotes no backtrack capability. True denotes backtrack capability.

            'gcr' : float. A value denoting the ground coverage ratio of a
            tracker system which utilizes backtracking; i.e. the ratio
            between the PV array surface area to total ground area. A tracker
            system with modules 2 meters wide, centered on the tracking axis,
            with 6 meters between the tracking axes has a gcr of 2/6=0.333.
            If gcr is not provided, a gcr of 2/7 is default. gcr must be <=1

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

    dataframe containing simulation results. Includes the fields present in
    input 'weather' in addtion to:

        'v_oc': open circuit voltage in Volts

        'aoi': angle of incidence in degrees.

        'temp_cell': cell temeprature in C.


    """

    # Rename the weather data for input to PVLIB.
    if np.all([c in weather.columns for c in ['dni','dhi','ghi','temp_air',
                                              'wind_speed','year','month',
                                              'day','hour','minute']]):
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
    v_oc = sapm_voc(effective_irradiance, temps['temp_cell'],
                                   module_parameters)
    df = weather.copy()
    df['aoi'] = aoi
    # df['aoi_loss'] = aoi_loss
    df['temp_cell'] = temps['temp_cell']
    df['effective_irradiance'] = effective_irradiance
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
        'Trad': calculate_voc(1000, mean_yearly_min_temp,
                                        module_parameters),
        'Day': calculate_voc(1000, mean_yearly_min_day_temp,
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
        'P99.5': 'P99.5 Voc',
        'Hist': 'Historical Maximum Voc',
        'Trad': 'Voc at 1 sun and mean yearly min ambient temperature',
        'Day': 'Voc at 1 sun and mean yearly minimum daytime (GHI>150 W/m2) temperature',
        'Norm_P99.5': 'P99.5 Voc assuming module normal to sun',
    }


    voc_summary['v_oc'] = voc_summary.index.map(voc_values)
    voc_summary['Conditions'] = voc_summary.index.map(conditions)
    voc_summary['max_string_voltage'] = max_string_voltage
    voc_summary['string_length'] = voc_summary['v_oc'].map(
        lambda x: voc_to_string_length(x, max_string_voltage))

    mean_yearly_min_temp = calculate_mean_yearly_min_temp(df.index, df['temp_air'])
    long_note = {
        'P99.5': "99.5 Percentile Voc<br>" + \
                 "P99.5 Voc: {:.3f} V<br>".format(voc_values['P99.5']) +\
                 "Maximum String Length: {:.0f}<br>".format(voc_summary['string_length']['P99.5']) +\
                 "Recommended 690.7(A)(3) value for string length.",

        'Hist': 'Historical maximum Voc from {:.0f}-{:.0f}<br>'.format(df['year'][0], df['year'][-1]) +\
                'Hist Voc: {:.3f}<br>'.format(voc_values['Hist']) + \
                'Maximum String Length: {:.0f}<br>'.format(voc_summary['string_length']['Hist']) + \
                'Conservative value for string length.',

        'Day': 'Traditional daytime Voc, using 1 sun irradiance and<br>' +\
                'mean yearly minimum daytime (GHI>150 W/m^2) dry bulb temperature of {:.1f} C.<br>'.format(mean_yearly_min_day_temp) +\
                'Trad Voc: {:.3f} V<br>'.format(voc_values['Day']) +\
                'Maximum String Length:{:.0f}<br>'.format(voc_summary['string_length']['Trad']) +\
                'Recommended 690.7(A)(1) Value',

        'Trad': 'Traditional Voc, using 1 sun irradiance and<br>' +\
                'mean yearly minimum dry bulb temperature of {:.1f} C.<br>'.format(mean_yearly_min_temp) +\
                'Trad Voc: {:.3f}<br>'.format(voc_values['Trad']) +\
                'Maximum String Length: {:.0f}'.format(voc_summary['string_length']['Trad']),

        'Norm_P99.5': "Normal Voc, 99.5 percentile Voc value<br>".format(voc_values['Norm_P99.5']) +\
                      "assuming array always oriented normal to sun.<br>" +\
                      "Norm_P99.5 Voc: {:.3f}<br>".format(voc_values['Norm_P99.5']) +\
                      "Maximum String Length: {:.0f}".format(voc_summary['string_length']['Norm_P99.5'])
    }
    short_note = {
        'P99.5': "Recommended 690.7(A)(3) value for string length.",

        'Hist': 'Conservative 690.7(A)(3) value for string length.',

        'Day':  'mean yearly minimum daytime (GHI>150 W/m^2) dry bulb temperature: {:.1f} C.<br>'.format(mean_yearly_min_day_temp) +\
                'Recommended 690.7(A)(1) Value',

        'Trad': 'mean yearly minimum dry bulb temperature: {:.1f} C.<br>'.format(mean_yearly_min_temp),

        'Norm_P99.5': ""
    }

    voc_summary['long_note'] = voc_summary.index.map(long_note)
    voc_summary['short_note'] = voc_summary.index.map(short_note)

    return voc_summary


def make_simulation_summary(df, info,module_parameters,racking_parameters,
                            thermal_model,max_string_voltage):
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

    voc_summary = make_voc_summary(df, module_parameters, max_string_voltage=max_string_voltage)

    if type(thermal_model)==type(''):
        thermal_model = {'Model parameters': thermal_model}

    if 'Location ID' in info:
        info['Location_ID'] = info['Location ID']
    if 'Time Zone' in info:
        info['local_time_zone'] = info['Time Zone']

    # extra_parameters = calculate_extra_module_parameters(module_parameters)


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

    if module['iv_model'] == 'sapm':

        v_oc = sapm_voc(effective_irradiance,temp_cell,module,
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
        calcparams_singlediode(effective_irradiance, temp_cell, module_parameters)


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

    Parameters
    ----------
    effective_irradiance
        Effective irradiance in W/m^2

    temp_cell

    module

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

    # Bvmpo = module['Bvmpo'] + module['Mbvmp'] * (1 - Ee)
    Bvoco = module['Bvoco'] + module['Mbvoc'] * (1 - Ee)
    delta = module['n_diode'] * kb * (temp_cell + 273.15) / q

    # avoid repeated computation
    logEe = np.full_like(Ee, np.nan)
    np.log(Ee/reference_irradiance, where=Ee_gt_0, out=logEe)
    logEe = np.where(Ee_eq_0, -np.inf, logEe)

    # avoid repeated __getitem__
    cells_in_series = module['cells_in_series']

    v_oc = np.maximum(0, (
            module['Voco'] + cells_in_series * delta * logEe +
            Bvoco * (temp_cell - T0)))

    return v_oc

def calcparams_singlediode(effective_irradiance, temperature, module_parameters):


    # Default to desoto model.
    if not 'iv_model' in module_parameters.keys():
        module_parameters['iv_model'] = 'desoto'

    if module_parameters['iv_model'] =='desoto':
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
        raise Exception("Model type must be 'cec' or 'desoto'")

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
        calcparams_singlediode(effective_irradiance,temperature, module_parameters)

    iv_curve = pvlib.pvsystem.singlediode(photocurrent, saturation_current,
                               resistance_series, resistance_shunt, nNsVth,
                               ivcurve_pnts=ivcurve_pnts, method='lambertw')

    return iv_curve

def calculate_sapm_module_parameters(module_parameters,reference_irradiance=1000,
                                          reference_temperature=25):

    """

    Calculate standard parameters of modules from the single diode model.

    module_parameters: dict

    Returns

    Dict of parameters including:

         'Voco' - open circuit voltage at STC.

         'Bvoco' - temperature coefficient of Voc near STC, in V/C

         Isco - short circuit current at STC

         Bisco - temperature coefficient of Isc near STC, in A/C

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

    param['n_diode'] = module_parameters['a_ref'] / (module_parameters['N_s'] * Vthref)


    # Calculate Voc vs. temperature for finding coefficients
    temp_cell_smooth = np.linspace(reference_temperature-5,
                                   reference_temperature+5, 5)

    photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = \
        calcparams_singlediode(effective_irradiance=reference_irradiance,
                   temperature=temp_cell_smooth,
                   module_parameters=module_parameters)
    iv_points = pvlib.pvsystem.singlediode(photocurrent,
            saturation_current, resistance_series, resistance_shunt, nNsVth)


    photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = \
        calcparams_singlediode(
        effective_irradiance=reference_irradiance,
        temperature=reference_temperature,
        module_parameters=module_parameters)

    iv_points_0 = pvlib.pvsystem.singlediode(photocurrent,
            saturation_current, resistance_series, resistance_shunt, nNsVth)


    param['Voco'] = iv_points_0['v_oc']
    param['Isco'] = iv_points_0['i_sc']
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
    param['Bisco'] = isc_fit_coeff[0]

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


def voc_to_string_length(voc,max_string_voltage):
    """

    Returns the maximum number N modules with open circuit voltage voc that
    do not exceed max_string_voltage when connected in series.

    Parameters
    ----------
    voc : float
        Open circuit voltage
    max_string_voltage : float
        Maximum string voltage

    Returns
    -------
    N : float
        Maximum string length

    """
    return np.round(np.floor(max_string_voltage/voc))



def simulate_system_sandia(weather,info, module_parameters=None,system_parameters=None):
    """
    Use the PVLIB Sandia model to calculate max voc.


    :param weather:
    :param info:
    :param module_parameters:
    :param system_parameters:
    :return:
    """


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