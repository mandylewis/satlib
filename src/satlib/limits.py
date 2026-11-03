import pandas as pd
import numpy as np


def _forward_smear(x, limit):
    # x: np.array
    # limit: speed limit, in degrees per time interval
    diff = np.diff(x)
    carryover = 0
    for i in range(0, len(diff)):
        desired_step = diff[i] + carryover
        actual_step = np.clip(desired_step, -limit, limit)
        diff[i] = actual_step
        carryover = desired_step - actual_step
        
    out = np.concat([x[:1], x[0] + np.cumsum(diff)])
    return out


def _forward_smear_until_change(x, limit):
    # x: np.array
    # limit: speed limit, in degrees per time interval
    diff = np.diff(x)
    carryover = 0
    n_smeared = len(diff)
    actual_step = diff[0]
    for i in range(0, len(diff)):
        desired_step = diff[i] + carryover
        # use actual_step from previous iteration
        if (np.sign(desired_step) != np.sign(actual_step)) and (actual_step != 0):
            # we've reached a "change"
            n_smeared = i
            break

        actual_step = np.clip(desired_step, -limit, limit)
        diff[i] = actual_step
        carryover = desired_step - actual_step

    out = np.concat([x[:1], x[0] + np.cumsum(diff[:n_smeared])])
    return out


def _smear_until_change(x, limit, direction):
    if direction == "forward":
        return _forward_smear_until_change(x, limit)
    elif direction == "backward":
        return _forward_smear_until_change(x[::-1], limit)[::-1]
    raise ValueError(direction)


def _successive_minima(theta, limit, iteration_limit):

    out = theta.copy()
    is_solved = np.full(out.shape, False)

    n_iteration = 0
    while not np.all(is_solved):
        # find smallest value we haven't solved yet
        aux = theta.copy()
        aux[is_solved] = 1e10  # ignore solved points
        idx = np.argmin(np.abs(aux))
        
        if idx < len(out) - 1:
            forward_smear = _smear_until_change(out[idx:], limit, "forward")
            n = len(forward_smear)
            out[idx:idx+n] = forward_smear
            is_solved[idx:idx+n] = True

        if idx > 0:
            backward_smear = _smear_until_change(out[:idx+1], limit, "backward")
            n = len(backward_smear)
            out[idx-n+1:idx+1] = backward_smear
            is_solved[idx-n+1:idx+1] = True

        n_iteration += 1
        if n_iteration > iteration_limit:
            raise RuntimeError

    return out


def clip_speed(theta, limit, how="shallower"):
    """
    Apply rotation speed limits to a tracker position time
    series.

    Where the input time series moves faster than the specified
    limit, the returned time series:

    * "lead": leads the input time series
    * "lag": lags the input time series
    * "shallower": leads and lags such that the output is shallower
      tilt than the input

    Parameters
    ----------
    theta : pandas.Series
        The input time series.  Must have datetime index with constant
        time step. [degrees]
    limit : float
        Maximum rotation rate. [degrees per minute]
    how : {"shallower", "lead", "lag"}, default "shallower"
        Which method to apply.

    Returns
    -------
    limited : pandas.Series
        The time series, subject to the specified rotation limit.
        [degrees]
    """
    # degrees per minute * minutes per step = degrees per step
    dt = theta.index[1] - theta.index[0]
    limit = limit * (dt / pd.Timedelta('1min'))

    if how == "lag":
        out = _forward_smear(theta.values, limit)
    elif how == "lead":
        out = _forward_smear(theta.values[::-1], limit)[::-1]
    elif how == "shallower":
        out = _successive_minima(theta.values, limit, iteration_limit=1000)
    else:
        raise ValueError(f'Invalid "how": "{how}"')

    return pd.Series(out, index=theta.index)


def clip_position(theta, limit):
    """
    Apply rotation limits to a tracker position time series.

    Parameters
    ----------
    theta : numeric
        The tracker position time series. [degrees]
    limit : float or tuple of numeric
        The max/min position limit.  If a scalar is passed,
        it is used symmetrically for min and max limits.
        Otherwise, a tuple of (min, max) limits must be passed.

    Returns
    -------
    clipped : numeric
        The position time series with limits applied.
    """
    if np.isscalar(limit):
        limit = (-limit, limit)
    return np.clip(theta, limit[0], limit[1])
