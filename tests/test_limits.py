import satlib
import numpy as np
import pandas as pd

import pytest


def test_clip_position():
    thetas = np.array([-70, -30, 0, 30, 70])
    out = satlib.limits.clip_position(thetas, limit=50)
    assert np.allclose(out, np.array([-50, -30, 0, 30, 50]))
    out = satlib.limits.clip_position(thetas, limit=(-40, 50))
    assert np.allclose(out, np.array([-40, -30, 0, 30, 50]))
    

def test_clip_position_types():
    thetas = np.array([-70, -30, 0, 30, 70, np.nan])
    expect = np.array([-50, -30, 0, 30, 50, np.nan])

    # np array
    out = satlib.limits.clip_position(thetas, 50)
    assert isinstance(out, np.ndarray)
    assert all(np.isclose(out, expect) | np.isnan(expect))

    # pd series
    out = satlib.limits.clip_position(pd.Series(thetas), 50)
    pd.testing.assert_series_equal(out, pd.Series(expect))

    # scalar
    out = satlib.limits.clip_position(-70., 50)
    assert isinstance(out, float)
    assert out == -50


@pytest.fixture
def toy_thetas():
    vals = [-12, -11, -10, -2, -1, 0, 1, 2, 10, 11, 12]
    times = pd.date_range("2020-06-01 12:00", freq="1min", periods=len(vals))
    thetas = pd.Series(vals, index=times)
    return thetas


@pytest.mark.parametrize("how,expected",[
    ("lead", [-12, -11, -7, -2, -1, 0, 1, 5, 10, 11, 12]),
    ("lag", [-12, -11, -10, -5, -1, 0, 1, 2, 7, 11, 12]),
    ("shallower", [-12, -11, -7, -2, -1, 0, 1, 2, 7, 11, 12]),
])
def test_clip_speed(toy_thetas, how, expected):
    expected = pd.Series(expected, index=toy_thetas.index)
    clipped = satlib.limits.clip_speed(toy_thetas, 5, how=how)
    pd.testing.assert_series_equal(clipped, expected)


@pytest.fixture
def discretized():
    psza = np.arange(-90, 90, 0.1)
    times = pd.date_range("2020-06-01 12:00", freq="1min", periods=len(psza))
    psza = pd.Series(psza, index=times)
    discretized = satlib.discretization.discretize_angle(psza, interval=5)
    return discretized


def _verify_shallower_requirements(clipped, theta, limit, match=None):
    assert (clipped.abs() <= theta.abs() + 1e-6).all()  # achieves "shallower" goal
    speed = clipped.diff() / (clipped.index.diff() / pd.Timedelta("1min"))
    assert speed.abs().max() <= (limit + 1e-6)  # achieves speed limit goal
    assert np.all(~np.isnan(clipped))  # no nans
    if match is not None:
        assert np.mean(np.isclose(clipped, theta)) > match  # at least some match


@pytest.mark.parametrize("how", ["lead", "lag", "shallower"])
def test_clip_speed_discrete(discretized, how):
    limit = 1
    clipped = satlib.limits.clip_speed(discretized, limit=limit, how=how)
    speed = clipped.diff()  # index is 1-min, so this is deg/min
    assert speed.abs().max() <= limit
    if how == "shallower":
        _verify_shallower_requirements(clipped, discretized, limit, match=0.9)


def test_clip_speed_shallower():
    # shallower is more complicated than lead/lag;
    # test a tracker curve that hits speed limits in all four quadrants of
    # the (tracker angle, tracker velocity) plane.
    n = 500
    theta = pd.Series(25 * np.sin(np.linspace(0, 4*np.pi, n)), index=pd.date_range("2020-06-01 12:00", freq="6s", periods=n))
    theta = np.clip(theta, -20, 20)
    limit = 5

    clipped = satlib.limits.clip_speed(theta, limit=limit, how="shallower")
    _verify_shallower_requirements(clipped, theta, limit, match=0.3)


@pytest.fixture
def psbt():
    times = pd.date_range("2020-06-01 08:00", "2020-06-01 16:00", freq="1min")
    psza = np.linspace(-90, 90, len(times))
    bt = satlib.backtracking.backtrack_psbt(psza, gcr=0.4, allowed_shade_fractions=[0.0, 0.5])
    theta = np.where(np.isnan(bt), psza, bt)
    return pd.Series(theta, index=times)


def test_clip_speed_shallower_discontinuities(psbt):
    # handling discontinuities is tricky.
    # use PSBT as an example
    limit = 1
    clipped = satlib.limits.clip_speed(psbt, limit=limit, how="shallower")
    _verify_shallower_requirements(clipped, psbt, limit, match=0.75)


def test_clip_speed_zeros(psbt):
    psbt.iloc[:10] = 0
    psbt.iloc[-10:] = 0
    psbt.iloc[50:60] = 0
    limit = 1
    clipped = satlib.limits.clip_speed(psbt, limit=limit, how="shallower")
    _verify_shallower_requirements(clipped, psbt, limit, match=0.7)


def test_clip_speed_nans(psbt):
    psbt.iloc[:10] = np.nan
    psbt.iloc[-10:] = np.nan
    psbt.iloc[50:60] = np.nan
    limit = 1
    clipped = satlib.limits.clip_speed(psbt, limit=limit, how="shallower")

    assert ((clipped.abs() <= psbt.abs() + 1e-6) |
            (clipped.isnull() & psbt.isnull())).all()  # achieves "shallower" goal
    speed = clipped.diff() / (clipped.index.diff() / pd.Timedelta("1min"))
    assert speed.abs().max() <= (limit + 1e-6)  # achieves speed limit goal
    assert np.mean(np.isclose(clipped, psbt)) > 0.75


def test_clip_speed_invalid(toy_thetas):
    with pytest.raises(ValueError, match="Invalid"):
        satlib.limits.clip_speed(toy_thetas, limit=5, how="not a method")
