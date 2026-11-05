import numpy as np
import pandas as pd
import pvlib
import satlib
from satlib._trigd import sind, cosd, atand
from satlib.limits import clip_speed
from pvlib import tools

def get_surface_azimuth(solar_azimuth, axis_azimuth):
    # for an HSAT tracker, get surface azimuth from solar azimuth and axis azimuth (all in degrees)
    # the surface azimuth is either +90 or -90 degrees from the axis azimuth, depending on the solar azimuth angle:
        # if solar azimuth is from (axis azimuth - 180) to axis azimuth, surface azimuth = axis azimuth - 90
        # if solar azimuth is from axis azimuth to (axis azimuth + 180), surface azimuth = axis azimuth + 90

    # get the solar azimuth angle with respect to axis aximuth
    relative_angle = (solar_azimuth - axis_azimuth) % 360 

    # flag what side of the axis the solar azimuth is on
    axis_flag = relative_angle > 180

    #if the solar azimuth is >180 degrees past the axis azimuth: surface azimuth = axis azimuth - 90
    #if the solar azimuth is <180 degrees past the axis azimuth: surface azimuth = axis azimuth + 90
    surface_azimuth = np.where(axis_flag, (axis_azimuth - 90) % 360, (axis_azimuth + 90) %360)

    return surface_azimuth, axis_flag

def munoz2024_analytical(solar_zenith, solar_azimuth, dni, dhi, ghi, albedo,axis_azimuth=180):
    r"""
    Calculate the tracker rotation angle using the CENER analytical model.
 
    Implements eq. 12 of [1]_. The optimal rotation angle of a horizontal
    single-axis tracker is found by taking the derivative of the total irradiance
    with respect to the tracker angle and setting it to zero.

    Assumes isotropic diffuse irradiance and Lambertian ground reflection.

    Parameters
    ----------
    solar_zenith : numeric
        Solar zenith angle. [°]
    solar_azimuth : numeric
        Solar azimuth angle. [°]
    dni : numeric
        Direct normal irradiance. [W/m^2]
    dhi : numeric
        Diffuse horizontal irradiance. [W/m^2]
    ghi : numeric
        Global horizontal irradiance. [W/m^2]
    albedo : numeric
        Ground albedo. [unitless]
    axis_azimuth : float, default 180
        Compass direction along which the tracker axis is oriented.
        Clockwise from north. [°]

    Returns
    -------
    tracker_theta : numeric
        Tracker rotation angle as a right-handed rotation around the axis
        defined by ``axis_azimuth``. Negative when the rotation is clockwise
        and positive when counterclockwise, looking along the tracker axis
        from ``axis_azimuth``. ``tracker_theta = 0`` is horizontal. [°]


    References
    ----------
    .. [1] I. Muñoz, A. Guinda, G. Olivares, S. Díaz, A. M. Gracia-Amillo, and
       L. Casajús, "Evaluation of Horizontal Single-Axis Solar Tracker
       Algorithms in Terms of Energy Production and Operational Performance,"
       Solar RRL, vol. 8, no. 1, p. 2300507, 2024.
       :doi:`10.1002/solr.202300507`
    """

    surface_azimuth, axis_flag = get_surface_azimuth(solar_azimuth, axis_azimuth)
    
    numerator = dni * sind(solar_zenith) * cosd(solar_azimuth - surface_azimuth)
    denominator = (dhi - ghi*albedo)/2 + dni * cosd(solar_zenith)
    beta_opt = np.arctan(numerator / denominator)

    #return tracker theta:  negative when clockwise looking along TT from axis azimuth
    #                       positive when counterclockwise looking along TT from axis azimuth  
    beta_opt = np.where(axis_flag, -1*beta_opt, beta_opt)

    return np.degrees(beta_opt)

def kellygibson2009_tracking1(tracker_theta,ghi, ghi_clear,threshold=0.3):
    """
    Implement the "enhanced tracking system number 1" from Kelly & Gibson (2009).
    Adapted for a single-axis tracker as discussed in 3.4.4. Assume standard SAT tracking
    when not in diffuse stow (0 degrees).

    Note: [1]_ does not provide a threshold value, so a default clear-sky threshold of 0.3 is assumed.

    When the ratio of GHI to clear-sky GHI is less than a threshold, set the tracker to horizontal [1]_:

    if ``GHI < GHI_clear * threshold``:
    ``tracker_set = 0``

    if ``GHI >= GHI_clear * threshold``:
    ``tracker_set = tracker_theta_ref`` (reference tracking angle)

    Parameters
    ----------
    tracker_theta : numeric
        Reference tracker rotation angle [degrees]
    ghi : numeric
        Global horizontal irradiance. [W/m^2]
    ghi_clear : numeric
        Clear-sky global horizontal irradiance. [W/m^2]
    threshold : float, default 0.3
        Clear-sky index (``ghi / ghi_clear``) threshold. When clear-sky index
        is below this value, set tracker flat. [unitless]

    Returns
    -------
    tracker_angle_set : numeric
        Tracker rotation angle after applying the algorithm. [degrees]

    References
    ----------
    .. [1] N. A. Kelly and T. L. Gibson, "Improved photovoltaic energy output
       for cloudy conditions with a solar tracking system," Solar Energy,
       vol. 83, no. 11, pp. 2092-2102, Nov. 2009.
       :doi:`10.1016/j.solener.2009.08.009`
    """
    flat_tracking_flag = ghi < ghi_clear*threshold
    tracker_angle_set = np.where(flat_tracking_flag, 0, tracker_theta)

    return tracker_angle_set

def kellygibson2009_tracking2(tracker_theta,ghi, poa_lat,threshold=1.3):
    """
    Implement the "enhanced tracking system number 2" from Kelly & Gibson (2009).
    Adapted for a single-axis tracker as discussed in 3.4.4. Assume standard SAT tracking
    when not in diffuse stow (0 degrees).

    Assume a horizontally mounted and a latitude-tilt, equator-facing reference cell.
    When the horizontal measurement (H, or GHI) exceeds the latitude-tilt measurement (L) 
    multiplied by a threshold, set the tracker to horizontal [1]_:
    
    if ``GHI > L * threshold``:
    ``tracker_set = 0``


    if ``GHI <= L * threshold``:
    ``tracker_set = tracker_theta_ref`` (reference tracking angle)

    Parameters
    ----------
    tracker_theta : numeric
        Reference tracker rotation angle. [degrees]
    ghi : numeric
        Global horizontal irradiance measured by a horizontal 
        reference cell (H). [W/m^2]
    poa_lat : numeric
        Irradiance measured by a latitude-tilt, equator-facing 
        reference cell (L). [W/m^2]
    threshold : float, default 1.3
        Ratio of ``ghi`` to ``poa_lat`` above which the tracker is set
        flat. [unitless]
 
    Returns
    -------
    tracker_angle_set : numeric
        Tracker rotation angle after applying the algorithm. [degrees]

    References
    ----------
    .. [1] N. A. Kelly and T. L. Gibson, "Improved photovoltaic energy output
       for cloudy conditions with a solar tracking system," Solar Energy,
       vol. 83, no. 11, pp. 2092-2102, Nov. 2009.
       :doi:`10.1016/j.solener.2009.08.009`
    """

    # If GHI > G_poa * theshold, rotate to 0 degrees (horizontal)
    # If GHI < G_poa * threshold, follow standard tracking algorithm
    flat_tracking_flag = ghi > poa_lat*threshold

    # take the reference tracking angle if GHI < GPOA_lat*threshold, and 0 if GHI > GPOA_lat*threshold
    tracker_angle_set = np.where(flat_tracking_flag, 0, tracker_theta)

    return tracker_angle_set

def kellygibson2009_tracking3(tracker_theta,dni,threshold=10):
    """
    Implement the "enhanced tracking system number 3" from Kelly & Gibson (2009).
    Adapted for a single-axis tracker as discussed in 3.4.4. Assume standard SAT tracking
    when not in diffuse stow (0 degrees).


    Note: [1]_ lists only "near-zero" DNI, so a default threshold of 10 W/m2 is applied.
    [1]_ calls for combining a GHI and DHI measurement to calculate DNI from the closure equation; 
    here, we assume that DNI is available.

    When DNI is near-zero, set the tracker to horizontal [1]_:
    
    ``DNI < threshold``:
    ``tracker_set = 0``
    
    ``DNI >= threshold``:
    ``tracker_set = tracker_theta_ref`` (reference tracking angle)

    Parameters
    ----------
    tracker_theta : numeric
        Reference tracker rotation angle. [degrees]
    dni : numeric
        Direct normal irradiance. [W/m^2]
    threshold : float, default 10
        DNI below which the tracker is set flat. [W/m^2]

    Returns
    -------
    tracker_angle_set : numeric
        Tracker rotation angle after applying the algorithm. [degrees]

    References
    ----------
    .. [1] N. A. Kelly and T. L. Gibson, “Improved photovoltaic energy output 
       for cloudy conditions with a solar tracking system,” Sol. Energy, 
       vol. 83, no. 11, pp. 2092-2102, Nov. 2009, doi: 10.1016/j.solener.2009.08.009.
    """

    # If DNI < threshold, rotate to 0 degrees (horizontal)
    # If DNI >= threshold, follow standard tracking algorithm
    flat_tracking_flag = dni < threshold

    # take the reference tracking angle if DNI < threshold, and 0 if DNI >= threshold
    tracker_angle_set = np.where(flat_tracking_flag, 0, tracker_theta)

    return tracker_angle_set

def munoz2024_diffuse(timestamp,tracker_theta_ref,solar_zenith,solar_azimuth, dni, dhi, ghi, albedo, axis_azimuth=180):
    r"""
    Calculate HSAT tracker angle using the CENER diffuse model.
    
    See section 2.1.1 of [1]_.

    When GHI > POA, ``tracker_theta = 0``

    When GHI < POA, ``tracker_theta = tracker_theta_ref`` 

    To calculate POA at the reference tracker angle:
    ``G_{poa} = G_{sky\,diffuse} + G_{direct} + G_{ground}``

    Find diffuse irradiance contribution from Perez model.

    Find ground-reflected irradiance assuming Lambertian ground reflection.

    Parameters
    ----------
    timestamp : pandas Datetime
        Timestamps of the input values.
    tracker_theta_ref : numeric
        Reference tracker rotation angle, [°]
    solar_zenith : numeric
        Solar zenith angle. [°]
    solar_azimuth : numeric
        Solar azimuth angle. [°]
    dni : numeric
        Direct normal irradiance. [W/m^2]
    dhi : numeric
        Diffuse horizontal irradiance. [W/m^2]
    ghi : numeric
        Global horizontal irradiance. [W/m^2]
    albedo : numeric
        Ground albedo. [unitless]
    axis_azimuth : float, default 180
        Compass direction along which the tracker axis is oriented.
        Clockwise from north. [°]

    Returns
    -------
    tracker_theta : numeric
        New tracker rotation angle. [°]

    References
    ----------
    .. [1] I. Muñoz, A. Guinda, G. Olivares, S. Díaz, A. M. Gracia-Amillo, and
       L. Casajús, "Evaluation of Horizontal Single-Axis Solar Tracker
       Algorithms in Terms of Energy Production and Operational Performance,"
       Solar RRL, vol. 8, no. 1, p. 2300507, 2024.
       :doi:`10.1002/solr.202300507`
    """

    #Compare G_poa with GHI. If GHI > G_poa, set tracker angle to 0 (flat). If GHI < G_poa, set tracker angle to reference tracking angle.
    
    # get surface orientation
    surface_orientation = pvlib.tracking.calc_surface_orientation(tracker_theta_ref, axis_tilt = 0, axis_azimuth=axis_azimuth)


    #Calculate G_poa:

    # get extraterrestrial irradiance
    dni_extra = pvlib.irradiance.get_extra_radiation(timestamp)
    # calculate G_poa using Perez model
    poa_components = pvlib.irradiance.get_total_irradiance(surface_orientation['surface_tilt'],surface_orientation['surface_azimuth'],
        solar_zenith,solar_azimuth,dni,ghi,dhi,albedo=albedo,dni_extra=dni_extra,model='perez')
    
    poa_total = poa_components['poa_global']

    # get Munoz2024 diffuse tracker angle from reference tracking angle and GHI vs GPOA comparison 
    tracker_theta = munoz2024_diffusetrackerangle(tracker_theta_ref,poa_total,ghi)

    return tracker_theta

def munoz2024_diffusetrackerangle(tracker_theta_ref,POA,GHI):
    r"""
    Calculate tracker angle according to CENER diffuse algorithm.
     
    See section 2.1.1 of [1]_.

    Set the tracker to 0° when GHI > POA.
    Otherwise, set the tracker to the reference tracking angle.
  
    Parameters
    ----------
    tracker_theta_ref : numeric
        Reference tracker rotation angle. [°]
    POA : numeric
        Plane-of-array irradiance at the reference tracker angle. [W/m^2]
    GHI : numeric
        Global horizontal irradiance. [W/m^2]
 
    Returns
    -------
    tracker_theta : numeric
        New tracker rotation angle. [°]
 
    References
    ----------
    .. [1] I. Muñoz, A. Guinda, G. Olivares, S. Díaz, A. M. Gracia-Amillo, and
       L. Casajús, "Evaluation of Horizontal Single-Axis Solar Tracker
       Algorithms in Terms of Energy Production and Operational Performance,"
       Solar RRL, vol. 8, no. 1, p. 2300507, 2024.
       :doi:`10.1002/solr.202300507`
    """

    # Find timestamps where GHI > GPOA
    flat_tracking_flag = GHI > POA

    # take the reference tracking angle if GHI < GPOA, and 0 if GHI > GPOA
    tracker_theta = np.where(flat_tracking_flag, 0, tracker_theta_ref)

    return tracker_theta

def Antonanzas2018_model1(tracker_theta_ref,poa,ghi, timestamps,tracker_speed = 18.3,persistence_min = 5):
    r"""
    Calculate tracker angle according to Antonanzas 2018 model 1. 
    When GHI > POA for more than 5 mins, set the tracker to horizontal.
    See section 4.2 of [1]_.

    The paper does not specify how and when to return to regular tracking. 
    Here, we assume that we return to regular tracking when GHI < POA for more than 5 minutes.
    
    Parameters
    ----------
    tracker_theta_ref : numeric
        Reference tracker rotation angle. [°]
    POA : numeric
        Plane-of-array irradiance at the reference tracker angle. [W/m^2]
    GHI : numeric
        Global horizontal irradiance. [W/m^2]
    timestamps: pandas Datetime
        Timestamps of the tracker angles and irradiance values.
    tracker_speed: float, default 18.3
        Maximum tracker speed in degrees per minute. [°/min]
    persistence_min: float, default 5
        Time in minutes to hold a condition before changing set angle

    Returns
    -------
    tracker_theta : numeric
        New tracker rotation angle. [°]

    References
    ----------
    .. [1] J. Antonanzas, R. Urraca, F. J. Martinez-de-Pison, and F. Antonanzas,
        “Optimal solar tracking strategy to increase irradiance in the plane of 
        array under cloudy conditions: A study across Europe,” Sol. Energy, vol. 
        163, pp. 122-130, Mar. 2018,
        :doi: `10.1016/j.solener.2018.01.080.`
    """

    # Build DatetimeIndex from timestamps 
    idx = pd.DatetimeIndex(timestamps)

    # Reference tracker angle on ts index. assume NaN values are 0 degrees
    ref = pd.Series(np.asarray(tracker_theta_ref, dtype=float), index=idx).fillna(0)

    # True where GHI > POA
    flat_tracking_flag = pd.Series(np.asarray(ghi) > np.asarray(poa), index=idx)

    # Minutes between timesteps; the first row has no predecessor, so assume 1 min
    # Q: how does pvlib handle similar assumptions? verify.
    dt = idx.to_series().diff().dt.total_seconds().div(60).fillna(1.0)

    # Label each tracking block with a unique id
    run_id = (flat_tracking_flag != flat_tracking_flag.shift()).cumsum()

    # Minutes the current condition has lasted (sum of dt for each block)
    run_min = dt.groupby(run_id).cumsum()

    # if GHI > POA for over persistence_min, enter = true
    enter = flat_tracking_flag & (run_min >= persistence_min)

    # if GHI <= POA for over persistence_min, enter = false
    leave = ~flat_tracking_flag & (run_min >= persistence_min)

    # create "state variable: 1.0 = enter flat, 0.0 = leave flat, NaN = no event
    # note: this might not be working?
    state = pd.Series(np.where(enter, 1.0, np.where(leave, 0.0, np.nan)), index=idx)

    # hold tracking mode until a new event occurs
    flat = state.ffill().fillna(0).astype(bool)

    # start moving tracker in the timestep after the condition is met
    flat = flat.shift(1, fill_value=False)

    # Target angle: 0° (horizontal) when flat, otherwise the reference tracking angle
    target = ref.where(~flat, 0.0)

    # apply tracker speed limit with a lag
    tracker_theta = clip_speed(target, tracker_speed, how="lag")
    
    return tracker_theta


def Antonanzas2018_model2(tracker_theta_ref,poa_predicted_hrly,ghi_predicted_hrly,tracker_speed = 18.3,persistence_min = 5):
    r"""
    Calculate tracker angle according to Antonanzas 2018 model 2. 
    Take hourly numerical weather prediction (NWP) of GHI as a 24 hr forecast starting at 00:00 [1]_.

    Estimate DHI from BRL model as in [2]_, then get DNI from closure equation.

    Calculate expected POA and compare with expected GHI.
    When predicted GHI > predicted POA for the coming hour, stow tracker horizontally for that hour.
    When predicted POA <= predicted GHI for the coming hour, track normally.

    Q: forecast data: right- left- center-label? all 3 (tracker ref, poa, ghi) should follow same convention

    
    Parameters
    ----------
    tracker_theta_ref : numeric
        Reference tracker rotation angle. [°]
    poa_predicted_hrly : numeric
        Predicted hourly plane-of-array irradiance at the reference tracker angle. [W/m^2]
    ghi_predicted_hrly : numeric
        Predicted hourly global horizontal irradiance. [W/m^2]
    tracker_speed: float, default 18.3
        Maximum tracker speed in degrees per minute. [°/min]
    persistence_min: float, default 5
        Time in minutes to hold a condition before changing set angle

    Returns
    -------
    tracker_theta : numeric
        New tracker rotation angle. [°]
    
    References
    ----------
    .. [1] J. Antonanzas, R. Urraca, F. J. Martinez-de-Pison, and F. Antonanzas,
        “Optimal solar tracking strategy to increase irradiance in the plane of 
        array under cloudy conditions: A study across Europe,” Sol. Energy, vol. 
        163, pp. 122-130, Mar. 2018,
        :doi: `10.1016/j.solener.2018.01.080.`
    """

    # stow module horizontally when predicted GHI > predicted POA for the coming hour
    flat_tracking_flag = ghi_predicted_hrly > poa_predicted_hrly

    # when the flat_tracking_flag is true, set tracker angle to zero. 
    tracker_theta = np.where(flat_tracking_flag, 0.0, tracker_theta_ref)


    # Q: how to then take and test against "real" data, what is the best approach?

    return tracker_theta



BRL_COEFFICIENTS = {
    'generic': {'b0': -5.38, 'b1': 6.63, 'b2': 0.006,
                'b3': -0.007, 'b4': 1.75, 'b5': 1.31},
}

def BRL_model_2010(timestamps,ghi,ghi_extra,ast,solar_zenith,model='generic'):
    r"""
    Implement the BRL 2010 decomposition model.
    Find DHI fraction from clearness index, daily clearness index, 
    solar altitude, apparent solar time and persistence of global radiation.

    Note: this approach was originally developed for hourly data, 
    but is applied here to any timestep.

    Parameters
    ----------
    timestamps: pandas Datetime
        Timestamps of the input data.
    ghi : pd.Series
        Global horizontal irradiance. [W/m^2]
    ghi_extra : pd.Series
        Extraterrestrial irradiance. [W/m^2]
    ast : pd.Series
        Apparent solar time. [hours]
    solar_zenith : pd.Series
        Solar zenith angle. [°]
    model : str, default 'generic'
        The coefficient model to apply.

    Returns
    -------
    d : numeric
        Diffuse fraction. [dimensionless]

    References
    ----------
    .. [1] B. Ridley, J. Boland, and P. Lauret, “Modelling of diffuse solar fraction 
        with multiple predictors,” Renew. Energy, vol. 35, no. 2, pp. 478-483, Feb. 2010.
        :doi: `10.1016/j.renene.2009.07.018`
    
    """

    # assign BRL coefficients
    c = BRL_COEFFICIENTS[model]

    # elevation angle from zenith angle
    solar_elevation = 90 - solar_zenith

    #get hourly clearness index, kt = GHI / GHI_extra
    kt = (ghi / ghi_extra).where(ghi_extra > 0) # kt for each daytime hour

    #group data by day and get daily Kt
    dates = timestamps.normalize() #get list of dates
    daily_ghi = ghi.groupby(dates).transform('sum')         #sum of ghi for the day
    daily_h0 = ghi_extra.groupby(dates).transform('sum')    #sum of extraterrestrial irradiance for the day
    daily_kt = (daily_ghi / daily_h0).where(daily_h0 > 0)         #ratio of daily GHI to daily extraterrestrial irradiance, K_t


    # calculate "persistence", phi, the average of the lead/lag of the clearness index.
    # take average of each point with the value 2 indices later, and shift forward by 1
    # eq. 6 from [1]
    kt_p1 = kt.shift(1)
    kt_m1 = kt.shift(-1)
    psi = pd.concat([kt_m1, kt_p1], axis=1).mean(axis=1, skipna=True)

    # calculate the diffuse fraction, d
    # eq. 9 from [1]
    z = (c['b0']
        + c['b1'] * kt
        + c['b2'] * ast
        + c['b3'] * solar_elevation
        + c['b4'] * daily_kt
        + c['b5'] * psi)
    d = 1.0 / (1.0 + np.exp(z))

    #daytime filter
    day = (solar_elevation > 0) & (ghi_extra > 0)
    d = d.where(day, 0.0)

    return d

def apparentsolartime(timestamps, lon, eqn_of_time):
    r"""
    Calculate the apparent solar time from the timestamp (center-label), longitude, and equation of time.

    Parameters
    ----------
    timestamps: pandas Datetime
        Timestamps of the input data (center-label).
    lon : numeric
        Longitude of the location. [°]
    eqn_of_time : numeric
        Equation of time. [minutes]
    
    Returns
    -------
    ast : pandas.Series
        Apparent solar time, indexed by timestamps. [hours]
    """

    hour_angle = pvlib.solarposition.hour_angle(timestamps, lon, eqn_of_time)
    ast_values = (hour_angle / 15 + 12) % 24 

    ast = pd.Series(ast_values, index=timestamps)

    return ast

def decomposition_from_DHIfrac(d,solar_zenith,ghi):
    r"""
    Take diffuse fraction, solar zenith angle, and GHI to calculate DHI and DNI.

    Parameters
    ----------
    d : Series
        Diffuse fraction. [dimensionless]   
    solar_zenith : Series
        Solar zenith angle. [degrees]
    ghi : Series
        Global horizontal irradiance. [W/m^2]

    Returns
    -------
    dhi : Series
        Diffuse horizontal irradiance. [W/m^2]
    dni : Series
        Direct normal irradiance. [W/m^2]
    """

    # DHI = GHI * dhi_fraction (d)
    dhi = d*ghi

    # direct irradiance in horizontal plane = GHI - DHI
    # DNI = direct irradiance in horizontal plane / cos(solar zenith)
    dni = pvlib.irradiance.dni(ghi, dhi, solar_zenith)

    # for high zenith angles, dni will return NaN, so replace with 0.
    dni = dni.fillna(0)

    return dhi, dni

def muneer_overcast_diffuse(surface_tilt, dhi):
    r"""
    Calculate diffuse irradiance POA component for overcast conditions from
    eq. 14 of Antonanzas (2018).
    
    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt. [°]
    dhi : numeric
        Diffuse horizontal irradiance. [W/m^2]

    Returns
    -------
    poa_diffuse : numeric
        Diffuse irradiance on the tilted surface. [W/m^2]
        
    References
    ----------
    .. [1] J. Antonanzas, R. Urraca, F. J. Martinez-de-Pison, and F. Antonanzas, 
    “Optimal solar tracking strategy to increase irradiance in the plane of 
    array under cloudy conditions: A study across Europe,” Sol. Energy, 
    vol. 163, pp. 122-130, Mar. 2018, 
    doi: 10.1016/j.solener.2018.01.080.
    
    """


    beta = np.radians(surface_tilt)
    T = (1 + np.cos(beta)) / 2 + 0.25227 * (np.sin(beta) - beta*np.cos(beta) - np.pi*(np.sin(beta/2)**2))
    poa_diffuse = dhi * T

    return poa_diffuse

def Antonanzas2018_POAcalc(surface_tilt,surface_azimuth,solar_zenith,solar_azimuth,ghi,dhi,dni,dni_extra,gcr,albedo=0.24, a_r=0.17):
    r"""
    Calculate effective POA on a tilted plane as described in Antonanzas 2018, section 2.

    Calculate direct, ground-reflected, and diffuse irradiance components on a tilted plane.
    Apply angle of incidence losses on each component separately.
    Sum the three effective irradiance components for total effective POA.

    1. DNI on poa: G_{direct} = DNI * cos(theta_i) (eq. 12 adapted for DNI)
    2. Reflected irradiance on poa: G_{ground} = GHI * albedo * (1 - cos(beta))/2 (eq. 13)
    3. Diffuse irradiance on poa: Muneer model (1990)(eq. 14-16, taken from pvlib PR here)
    4. Apply Martin Ruiz (2001) IAM model to each component (eq. 20-25, taken from pvlib here)
    5. Total irradiance = G_{poa} = G_{direct} + G_{ground} + G_{sky\,diffuse} (eq. 26)


    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt from horizontal. [°]
    surface_azimuth : numeric
        Surface azimuth, degrees east of north. [°]
    solar_zenith : numeric
        Solar zenith angle. [°]
    solar_azimuth : numeric
        Solar azimuth angle, degrees east of north. [°]
    ghi : numeric
        Global horizontal irradiance. [W/m^2]
    dhi : numeric
        Diffuse horizontal irradiance. [W/m^2]
    dni : numeric
        Direct normal irradiance. [W/m^2]
    dni_extra : numeric
        Extraterrestrial normal irradiance. [W/m^2]
    gcr : float
        Ground coverage ratio. [unitless]
    albedo : float, default 0.24
        Ground albedo. [1] uses 0.24 for grass and 0.8 for snow cover. [unitless]
    a_r : float, default 0.17
        Angular loss coefficient of the Martin and Ruiz model, default taken from [1]. [unitless]

    Returns
    -------
    poa_total_eff : numeric
        Effective plane-of-array irradiance after AOI losses. [W/m^2]

    References
    ----------
    .. [1] J. Antonanzas, R. Urraca, F. J. Martinez-de-Pison, and F. Antonanzas, 
        “Optimal solar tracking strategy to increase irradiance in the plane of 
        array under cloudy conditions: A study across Europe,” Sol. Energy, 
        vol. 163, pp. 122-130, Mar. 2018, doi: 10.1016/j.solener.2018.01.080.
        [2] textbook that gave muneer 2004

    """
    #get clearness index to create "overcast" flag
    kt = pvlib.irradiance.clearness_index(ghi,solar_zenith,dni_extra)
    airmass = pvlib.atmosphere.get_relative_airmass(solar_zenith,model='simple') #(eq. 18)
    kt_prime = pvlib.irradiance.clearness_index_zenith_independent(kt,airmass) # (eq. 17)
    overcast_ts = (kt_prime < 0.3) & (solar_zenith < 90) #flag of overcast conditions (threshold from Ineichen 2011 via Antonanzas 2018)

    #calculate aoi losses according to Martin Ruiz (2001) IAM model (eq. 20-25)
    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
    iam_b = pvlib.iam.martin_ruiz(aoi,a_r=a_r) # get direct angular loss factor
    iam_d, iam_r = pvlib.iam.martin_ruiz_diffuse(surface_tilt,a_r=a_r) #get diffuse and reflected loss factors

    #get direct irradiance on tilted plane
    direct_poa = np.maximum(dni * cosd(aoi),0) #eq. 12, modified
    direct_poa_eff = direct_poa * iam_b # eq. 23

    #reflected irradiance on tilted plane
    reflected_poa = ghi * albedo * (1 - cosd(surface_tilt))/2 #eq. 13
    reflected_poa_eff = reflected_poa * iam_r # eq. 24

    #apply muneer model for diffuse irradiance on tilted plane eq. 15-16
    diffuse_sunny = muneer2004(surface_tilt, surface_azimuth, dhi, ghi, dni_extra, solar_zenith=solar_zenith, solar_azimuth=solar_azimuth)

    #if overcast, apply correction for inter table shading (eq. 19)
    epsilon = atand(gcr*sind(surface_tilt)/(1-gcr*cosd(surface_tilt))) #derived from 2D geometry
    beta_prime = 180 - epsilon #complementary angle

    # eq. 19 - add some inter table shading to diffuse irradiance, but only when overcast
    cloudy_diffuse_poa_uncorr = muneer_overcast_diffuse(surface_tilt, dhi)
    cloudy_diffuse_poa_factor = muneer_overcast_diffuse(beta_prime, dhi) # diffuse irradiance blocked by adjacent rows
    diffuse_overcast = cloudy_diffuse_poa_uncorr - cloudy_diffuse_poa_factor

    # when kt_prime <0.3, use eq. 14. Otherwise use regular Muneer 2004 (eq. 15-16)
    diffuse_poa = np.where(overcast_ts, diffuse_overcast, diffuse_sunny)

    # apply direct IAM to diffuse irradiance when it's sunny and diffuse IAM when it's overcast (eq. 25)
    iam_diffuse = np.where(overcast_ts, iam_d, iam_b) #eq. 25

    diffuse_poa_eff = diffuse_poa * iam_diffuse 

    #take sum of 3 irradiance components to get total POA
    poa_total_eff = direct_poa_eff + reflected_poa_eff + diffuse_poa_eff

    return poa_total_eff




"""
The following implementation of the muneer model is taken from the PR here:
https://github.com/pvlib/pvlib-python/pull/2184

written by BernatNicolau.
"""

def muneer2004(surface_tilt, surface_azimuth, dhi, ghi, dni_extra,
               solar_zenith=None, solar_azimuth=None, projection_ratio=None,
               location="southern_europe"):
    '''
    Determine sky diffuse irradiance on a tilted surface using the
    Muneer model.

    This Muneer transposition model is originally described in [1]_, and
    improved for low solar elevations in [2]_.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angles in decimal degrees. ``surface_tilt`` must
        be >=0 and <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    surface_azimuth : numeric
        Surface azimuth angles in decimal degrees. ``surface_azimuth``
        must be >=0 and <=360. The azimuth convention is defined as degrees
        east of north (e.g. North = 0, South=180 East = 90, West = 270).

    dhi : numeric
        Diffuse horizontal irradiance. [W/m^2]

    ghi : numeric
        Global horizontal irradiance. [W/m^2]

    dni_extra : numeric
        Extraterrestrial normal irradiance. [W/m^2]

    location : string, default 'northern_europe'
        A string which selects the desired equation for calculating the second
        term of T of Muneer model based on location. If location is not
        provided as an input, the default, 'northern_europe' will be used.
        All possible location selections are:

        * 'northern_europe'
        * 'southern_europe'
        * 'japan'
        * 'globe'

    solar_zenith : numeric
        Solar apparent (refraction-corrected) zenith angles in decimal
        degrees. Must supply ``solar_zenith`` and ``solar_azimuth`` or
        supply ``projection_ratio``.

    solar_azimuth : numeric
        Solar azimuth angles in decimal degrees. Must supply
        ``solar_zenith`` and ``solar_azimuth`` or supply
        ``projection_ratio``.

    projection_ratio : numeric, optional
        Ratio of cosine of angle of incidence to cosine of solar zenith
        angle. [unitless]

    Returns
    -------
    poa_sky_diffuse : numeric
        In-plane diffuse irradiance from the sky. [W/m^2]

    References
    ----------
    .. [1] Muneer, T., Gueymard, C., & Kambezidis, H., 2004. Hourly horizontal
       irradiation and illuminance. 157-158
       :doi:`10.1016/B978-075065974-1/50011-5`

    .. [2] Gracia, A., & Huld, T., 2013. Performance comparison of different
       models for the estimation of global irradiance on inclined surfaces:
       validation of the model implemented in PVGIS. Centro Común de
       Investigación, Instituto de Energía y Transporte.
       :doi:`10.2790/91554`

    '''
    available_locations = ["northern_europe", "southern_europe", "japan",
                           "globe"]
    if location not in available_locations:
        raise ValueError
    desired_index = available_locations.index(location)

    cos_solar_zenith = tools.cosd(solar_zenith)
    # if necessary, calculate ratio of titled and horizontal beam irradiance
    if projection_ratio is None:
        cos_tt = pvlib.irradiance.aoi_projection(surface_tilt, surface_azimuth,
                                solar_zenith, solar_azimuth)
        cos_tt = np.maximum(cos_tt, 0)  # GH 526
        Rb = cos_tt / np.maximum(cos_solar_zenith, 0.01745)  # GH 432
    else:
        Rb = projection_ratio

    F = pvlib.irradiance.clearness_index(
        ghi - dhi, solar_zenith, dni_extra)

    solar_elevation = np.pi/2 - np.radians(solar_zenith)
    low_elevation_condition = solar_elevation < 0.1

    aoi_ = pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
               solar_zenith, solar_azimuth)
    overcast_condition = aoi_ >= 90
    F = np.where(overcast_condition, 0, F)

    T_term1 = (1 + tools.cosd(surface_tilt)) * 0.5

    # term 2 depending on the selected location
    T_term2_model0 = 0.00333 - 0.415 * F - 0.6987 * F ** 2
    T_term2_model1 = 0.00263 - 0.712 * F - 0.6883 * F ** 2
    T_term2_model2 = 0.08000 - 1.050 * F - 2.8400 * F ** 2
    T_term2_model3 = 0.04000 - 0.820 * F - 2.0260 * F ** 2
    T_term2 = np.array([T_term2_model0, T_term2_model1,
                       T_term2_model2, T_term2_model3])
    T_term2 = T_term2[desired_index]
    T_term2 = np.where(overcast_condition, 0.25227, T_term2)

    T_term3 = (
        tools.sind(surface_tilt)
        - np.radians(surface_tilt) * tools.cosd(surface_tilt)
        - np.pi * (1 - tools.cosd(surface_tilt)) * 0.5
    )

    T = T_term1 + T_term2 * T_term3

    numer_low = tools.sind(surface_tilt) * \
        tools.cosd(surface_azimuth - solar_azimuth)
    denom_low = 0.1 - 0.008 * solar_elevation

    sky_diffuse_low = dhi*(T*(1-F) + F*(numer_low/denom_low))
    sky_diffuse_high = dhi*(T*(1-F) + F*Rb)

    sky_diffuse = np.where(low_elevation_condition,
                           sky_diffuse_low, sky_diffuse_high)
    if isinstance(ghi, pd.Series):
        sky_diffuse = pd.Series(sky_diffuse,
                                index=ghi.index)
    return sky_diffuse



def muneer1990(surface_tilt, surface_azimuth, dhi, ghi, dni_extra, b=5.73,
            solar_zenith=None, solar_azimuth=None, projection_ratio=None):
    r"""
    Determine sky diffuse irradiance on a tilted surface using the
    Muneer model.

    This Muneer transposition model is described in [1]_.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angles in decimal degrees. ``surface_tilt`` must
        be >=0 and <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    surface_azimuth : numeric
        Surface azimuth angles in decimal degrees. ``surface_azimuth``
        must be >=0 and <=360. The azimuth convention is defined as degrees
        east of north (e.g. North = 0, South=180 East = 90, West = 270).

    dhi : numeric
        Diffuse horizontal irradiance. [W/m^2]Expand commentComment on line R1030Resolved

    ghi : numeric
        Global horizontal irradiance. [W/m^2]

    dni_extra : numeric
        Extraterrestrial normal irradiance. [W/m^2]

    b : numeric, default 5.73
        Radiance distribution index, introduced by Moon and Spencer [2]_
        to model luminance distribution of overcast sky. [unitless]
        Recommended values from [1]_:

           - isotropic: b = 0
           - shaded surface: b = 5.73 (default)
           - sunlit surface, overcast sky: b = 1.68
           - sunlit surface, non-overcast sky: b = -0.62

    solar_zenith : numeric
        Solar apparent (refraction-corrected) zenith angles in decimal
        degrees. Must supply ``solar_zenith`` and ``solar_azimuth`` or
        supply ``projection_ratio``.

    solar_azimuth : numeric
        Solar azimuth angles in decimal degrees. Must supply
        ``solar_zenith`` and ``solar_azimuth`` or supply
        ``projection_ratio``.

    projection_ratio : numeric, optional
        Ratio of cosine of angle of incidence to cosine of solar zenith
        angle. [unitless]

            Returns
    -------
    poa_sky_diffuse : numeric
        In-plane diffuse irradiance from the sky. [W/m^2]Expand commentComment on line R1065Resolved

    References
    ----------
    .. [1] Muneer, T., 1990. Solar radiation model for Europe. Building
       Services Engineering Research and Technology 11, 153-163.
       :doi:`10.1177/014362449001100405`

    .. [2] Moon, P., and Spencer, D. E., 1942. Illumination from a
       non-uniform sky.
       Trans. Illum. Eng. Soc. (London) 37, 707-725.
    """

    cos_solar_zenith = tools.cosd(solar_zenith)
    # if necessary, calculate ratio of titled and horizontal beam irradiance
    if projection_ratio is None:
        cos_tt = pvlib.irradiance.aoi_projection(surface_tilt, surface_azimuth,
                                solar_zenith, solar_azimuth)
        cos_tt = np.maximum(cos_tt, 0)  # GH 526
        Rb = cos_tt / np.maximum(cos_solar_zenith, 0.01745)  # GH 432
    else:
        Rb = projection_ratio

    T_term1 = (1 + tools.cosd(surface_tilt)) * 0.5
    T_term2 = 2 * b / (np.pi * (3 + 2 * b))
    T_term3 = (
        tools.sind(surface_tilt)
        - np.radians(surface_tilt) * tools.cosd(surface_tilt)
        - np.pi * (1 - tools.cosd(surface_tilt)) * 0.5
    )
    T = T_term1 + T_term2 * T_term3

    F = pvlib.irradiance.clearness_index(
        ghi - dhi, solar_zenith, dni_extra)

    solar_elevation = np.pi/2 - np.radians(solar_zenith)

    numer_low = tools.sind(surface_tilt) * \
        tools.cosd(surface_azimuth - solar_azimuth)
    denom_low = 0.1 - 0.008 * solar_elevation

    sky_diffuse_low = dhi*(T*(1-F) + F*(numer_low/denom_low))
    sky_diffuse_high = dhi*(T*(1-F) + F*Rb)

    low_elevation_condition = solar_elevation < 0.1
    sky_diffuse = np.where(low_elevation_condition,
                           sky_diffuse_low, sky_diffuse_high)
    if isinstance(sky_diffuse_low, pd.Series):
        sky_diffuse = pd.Series(sky_diffuse,
                                index=sky_diffuse_low.index)

    return sky_diffuse