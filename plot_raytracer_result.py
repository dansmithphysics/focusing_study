import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import savgol_filter
from scipy.misc import derivative

beamed_result = False

def unbined_signal_to_power(raw_signal):

    running_sum = np.arange(len(raw_signal))
    f_raw_signal = scipy.interpolate.interp1d(raw_signal, running_sum, kind='cubic', bounds_error=False, fill_value="extrapolate") # Effectively integrate

    new_x = np.linspace(0.0, 400.0, 1000)
    new_raw_signal = f_raw_signal(new_x) # Upsample
    new_raw_signal = savgol_filter(new_raw_signal, 51, 3) # Smooth it out
    new_raw_signal_p = np.gradient(new_raw_signal) # Derivative 
    new_raw_signal_p = savgol_filter(new_raw_signal_p, 51, 3) # Smooth out again

    return new_raw_signal_p

if(beamed_result):
    homo_xpos   = np.load("data_focus_homo_beamed_raytracer_xpos.npy")
    inhomo_xpos = np.load("data_focus_inhomo_beamed_raytracer_xpos.npy")
else:
    homo_xpos   = np.load("data_focus_homo_raytracer_xpos.npy")
    inhomo_xpos = np.load("data_focus_inhomo_raytracer_xpos.npy")
    
ub_inhomo_xpos = unbined_signal_to_power(inhomo_xpos)
ub_homo_xpos   = unbined_signal_to_power(homo_xpos)

new_x = np.linspace(0.0, 400.0, 1000)

if(beamed_result):
    beamed_power_profile = np.load("data_focus_efield_beamed_power_profile_meep_31456.npy")

    #for i in range(len(beamed_power_profile)):
    #    plt.plot(beamed_power_profile[i] / 1.0 + i, color="blue")
    #plt.show()

    power_slice = 100
    beamed_powers = []
    for i in range(len(beamed_power_profile)):
        beamed_fft = np.fft.rfft(beamed_power_profile[i])
        #plt.plot(10.0 * np.log10(np.abs(beamed_fft)) + i, color="blue", alpha=0.5)
    
        beamed_powers += [np.power(np.abs(beamed_fft[power_slice]), 2.0)]
    #plt.show()
    
    beamed_powers /= np.max(beamed_powers) # Normalize
    
    angle_steps = np.linspace(-np.pi/2.0, 0.0, 100) # As determined when running the MEEP sim
    f_power_profile = scipy.interpolate.interp1d(angle_steps, beamed_powers, kind='cubic', bounds_error=False, fill_value="extrapolate")

    #plt.plot(angle_steps, beamed_powers)
    #plt.show()
    
    nrays = 10000
    dir_steps = np.linspace(-np.deg2rad(90.0), -np.deg2rad(65.0), nrays) # As determined when running the raytracer
    power_at_steps = f_power_profile(dir_steps)

    f_inhomo_power = scipy.interpolate.interp1d(inhomo_xpos, power_at_steps, kind='cubic', bounds_error=False, fill_value="extrapolate")
    inhomo_power = f_inhomo_power(new_x)

    f_homo_power = scipy.interpolate.interp1d(homo_xpos, power_at_steps, kind='cubic', bounds_error=False, fill_value="extrapolate")
    homo_power = f_homo_power(new_x)

    ub_inhomo_xpos *= inhomo_power
    ub_homo_xpos   *= homo_power

plt.figure()
plt.plot(new_x, ub_inhomo_xpos / ub_homo_xpos, color="red", alpha=0.5, label="Focusing Factor")
plt.xlabel("X Position [m]")
plt.xlim(0.0, 350.0)
plt.ylim(1.1, 1.4)
plt.grid()
plt.legend()

plt.figure()
plt.title("Numerical Integral vs. X Pos")
plt.plot(homo_xpos, np.arange(len(homo_xpos)), color="red", alpha=0.5, label="Homo.")
plt.plot(inhomo_xpos, np.arange(len(inhomo_xpos)), color="blue", alpha=0.5, label="Inhomo.")
plt.xlabel("X Position [m]")
plt.xlim(0.0, 350.0)
#plt.ylim(1.1, 1.4)
plt.grid()
plt.legend()

plt.show()
