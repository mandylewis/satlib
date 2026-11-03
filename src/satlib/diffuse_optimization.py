import stat

import numpy as np
import pandas as pd
import pvlib
import satlib
from satlib._trigd import sind, cosd
from satlib.limits import clip_speed

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

def munoz2024_diffuse(tracker_theta_ref,solar_zenith,solar_azimuth, dni, dhi, ghi, albedo, axis_azimuth=180):
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
    dni_extra = pvlib.irradiance.get_extra_radiation(solar_zenith)
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

def Antonanzas2018_model1(tracker_theta_ref,poa_eff,ghi, timestamps,tracker_speed = 18.3,persistence_min = 5):
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
    want_flat = pd.Series(np.asarray(ghi) > np.asarray(poa_eff), index=idx)

    # Minutes between timesteps; the first row has no predecessor, so assume 1 min
    # Q: how does pvlib handle similar assumptions? verify.
    dt = idx.to_series().diff().dt.total_seconds().div(60).fillna(1.0)

    # Label each tracking block with a unique id
    run_id = (want_flat != want_flat.shift()).cumsum()

    # Minutes the current condition has lasted (sum of dt for each block)
    run_min = dt.groupby(run_id).cumsum()

    # if GHI > POA for over persistence_min, enter = true
    enter = want_flat & (run_min >= persistence_min)

    # if GHI <= POA for over persistence_min, enter = false
    leave = ~want_flat & (run_min >= persistence_min)

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


def Antonanzas2018_model2():





return tracker_theta