import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import savgol_filter
from scipy.misc import derivative

def power_from_angle(angle):
    angle = np.abs(np.rad2deg(angle)+90.0)
    sigma = 14.0
    return np.exp(-0.5 * np.power(angle / sigma, 2.0))

#homo_len = np.load("defocus__homo_length.npy")
#inhomo_len = np.load("defocus_inhomo_length.npy")

#homo_xpos = np.load("defocus_deep_homo_xpos.npy")
#inhomo_xpos = np.load("defocus_deep_inhomo_xpos.npy")

#homo_xpos = np.load("defocus_homo_xpos.npy")
#inhomo_xpos = np.load("defocus_inhomo_xpos.npy")

homo_xpos = np.load("defocus_homo_xpos_0.0.npy")
inhomo_xpos = np.load("defocus_inhomo_xpos_0.0.npy")

###
### X Pos
###
nrays = 1000
dir_steps = np.linspace(-np.deg2rad(90.0), -np.deg2rad(65.0), nrays)

running_sum = np.arange(len(inhomo_xpos))
new_x = np.linspace(0.0, 400.0, 1000)

f_inhomo_xpos = scipy.interpolate.interp1d(inhomo_xpos, running_sum, kind='cubic', bounds_error=False, fill_value="extrapolate")
new_inhomo_xpos = f_inhomo_xpos(new_x)
new_inhomo_xpos = savgol_filter(new_inhomo_xpos, 51, 3) # window size 51, polynomial order 3
new_inhomo_xpos_p = np.gradient(new_inhomo_xpos)
new_inhomo_xpos_p = savgol_filter(new_inhomo_xpos_p, 51, 3) # window size 51, polynomial order 3

beamed_power_profile = np.load("output_meep_output_0_beamed_power_profile_31456.npy")
angle_steps = np.linspace(-np.pi/2.0, 0.0, 100)
beamed_powers = []

for i in range(len(beamed_power_profile)):
    beamed_powers += [np.sum(np.power(beamed_power_profile[i], 2.0))]

beamed_powers = np.array(beamed_powers) / np.max(beamed_powers)

plt.plot(angle_steps, beamed_powers)
f_power_profile = scipy.interpolate.interp1d(angle_steps, beamed_powers, kind='cubic', bounds_error=False, fill_value="extrapolate")
power_at_steps = f_power_profile(dir_steps)

f_inhomo_power = scipy.interpolate.interp1d(inhomo_xpos, power_at_steps, kind='cubic', bounds_error=False, fill_value="extrapolate")
inhomo_power = f_inhomo_power(new_x)

new_inhomo_xpos_p *= np.power(inhomo_power, 0.2)

f_homo_xpos = scipy.interpolate.interp1d(homo_xpos, running_sum, kind='cubic', bounds_error=False, fill_value="extrapolate")
new_homo_xpos = f_homo_xpos(new_x)
new_homo_xpos = savgol_filter(new_homo_xpos, 51, 3) # window size 51, polynomial order 3
new_homo_xpos_p = np.gradient(new_homo_xpos)
new_homo_xpos_p = savgol_filter(new_homo_xpos_p, 51, 3) # window size 51, polynomial order 3

f_homo_power = scipy.interpolate.interp1d(homo_xpos, power_at_steps, kind='cubic', bounds_error=False, fill_value="extrapolate")
homo_power = f_homo_power(new_x)

new_homo_xpos_p *= np.power(homo_power, 0.2)

plt.figure()
#plt.plot(new_x, new_homo_xpos_pp, color="red", alpha=0.5, label="Homo.")
#plt.plot(new_x, new_inhomo_xpos_pp, color="blue", alpha=0.5, label="Inhomo.")
plt.plot(new_x, new_inhomo_xpos_p / new_homo_xpos_p, color="red", alpha=0.5, label="Focusing Factor")
plt.xlabel("X Position [m]")
plt.xlim(0.0, 350.0)
plt.ylim(1.1, 1.4)
plt.grid()
plt.legend()

plt.show()
exit()

###
### Distance correction
###
### Turned out not to matter
###

f_homo_len = scipy.interpolate.interp1d(homo_xpos, homo_len, kind='cubic', bounds_error=False, fill_value="extrapolate")
new_homo_len = f_homo_len(new_x)
new_homo_len = savgol_filter(new_homo_len, 51, 3) # window size 51, polynomial order 3

f_inhomo_len = scipy.interpolate.interp1d(inhomo_xpos, inhomo_len, kind='cubic', bounds_error=False, fill_value="extrapolate")
new_inhomo_len = f_inhomo_len(new_x)
new_inhomo_len = savgol_filter(new_inhomo_len, 51, 3) # window size 51, polynomial order 3

homo_power = np.power(1.0 / new_homo_len, 2.0)
inhomo_power = np.power(1.0 / new_inhomo_len, 2.0)

plt.figure()
#plt.plot(new_x, new_homo_len, color="red", alpha=0.5, label="Homo.")
#plt.plot(new_x, new_inhomo_len, color="blue", alpha=0.5, label="InHomo.")
#plt.plot(new_x, homo_power, color="red", alpha=0.5, label="Homo.")
#plt.plot(new_x, inhomo_power, color="blue", alpha=0.5, label="InHomo.")
plt.plot(new_x, inhomo_power / homo_power, color="red", alpha=0.5, label="Homo.")
plt.xlabel("X Position [m]")
plt.legend()
plt.show()

exit()

plt.hist(inhomo_xpos, range=(0.0, 400.0), bins=100)
plt.show()

exit()

plt.scatter(inhomo_len, inhomo_xpos)
plt.show()
