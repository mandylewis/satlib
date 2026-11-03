"""
Smearing
========

Applying speed limits to tracker position time series.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import satlib
import pvlib

# %%
# First, let's take a look at the basic "lead" and "lag" methods.
# The result is sometimes shallower and sometimes steeper than
# the target angles.

# %%

times = pd.date_range("2026-12-21 7:30:00", "2026-12-21 16:30:00", freq="1min", tz="UTC")
sp = pvlib.solarposition.get_solarposition(times, 45, 0)
sp = sp.loc[sp["apparent_elevation"] > 0, :]

tt = satlib.tracking.astronomical(sp.apparent_zenith, sp.azimuth, axis_tilt=0, axis_azimuth=180)
bt = satlib.backtracking.backtrack_standard(tt, gcr=0.4)

theta = np.where(np.isnan(bt), tt, bt)
theta = pd.Series(theta, index=sp.index)

theta.plot(c='k', label='baseline')
for how in ["lead", "lag"]:
    smeared = satlib.limits.clip_speed(theta, 1, how=how)
    smeared.plot(label=how)
    plt.legend()


# %%
# Compare that with the "shallower" method, which always produces positions
# shallower than the target:

# %%

theta.plot(c='k', label='baseline')
smeared = satlib.limits.clip_speed(theta, 1, how="shallower")
smeared.plot(label="shallower")
plt.legend()

# %%

bt = satlib.backtracking.backtrack_psbt(tt, gcr=0.4, allowed_shade_fractions=[0.0, 0.5])

theta = np.where(np.isnan(bt), tt, bt)
theta = pd.Series(theta, index=sp.index)
theta = satlib.discretization.discretize_angle(theta, interval=5)

theta.plot(c='k', label='baseline')
smeared = satlib.limits.clip_speed(theta, 1, how="shallower")
smeared.plot(label="shallower")
plt.legend()
