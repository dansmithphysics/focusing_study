import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import savgol_filter
from scipy.misc import derivative

beamed_result = True

#ff_homo = np.load("data_focus_efield_homo_meep_31456.npy", allow_pickle=True)
#ff_inhomo = np.load("data_focus_efield_inhomo_meep_31456.npy", allow_pickle=True)
ff_homo = np.load("data_focus_efield_homo_beamed_meep_31456.npy", allow_pickle=True)
ff_inhomo = np.load("data_focus_efield_inhomo_beamed_meep_31456.npy", allow_pickle=True)

sx = 800
sy = 1000
pml_thicc = 20.0
x_pos = np.linspace(0.0, sx / 2.0 - pml_thicc, 40)

power_ratio = []
power_slice = 20

plt.figure()

running_i = 0
#for j in range(80, 120):
for j in range(40, 80):

    ts = [time_step / 10.0 for time_step in range(len(ff_homo[j]))]    
    #plt.plot(ts, np.array(ff_homo[j]) / 0.1 + j, color="red", alpha=0.5)
    ts = [time_step / 10.0 for time_step in range(len(ff_inhomo[j]))]    
    #plt.plot(ts, np.array(ff_inhomo[j]) / 0.1 + j, color="blue", alpha=0.5)    
    #plt.show()

    fpulse_homo   = np.array([ff_homo[j][time_step] if ts[time_step] < 140 + 2000 else 0.0 for time_step in range(len(ts))])
    fpulse_inhomo   = np.array([ff_inhomo[j][time_step] if ts[time_step] < 320 + 2000 else 0.0 for time_step in range(len(ts))])
    #plt.plot(ts, np.array(fpulse_homo) / 0.01 + j, color="red", alpha=0.5)
    #plt.plot(ts, np.array(fpulse_inhomo) / 0.01 + j, color="blue", alpha=0.5)
    
    #plt.plot(np.array(ff_homo[j]) / 0.3 + j, color="red", alpha=0.5)
    #plt.plot(np.array(ff_inhomo[j]) / 0.3 + j, color="blue", alpha=0.5)
    
    ff_homo_rfft = np.fft.rfft(fpulse_homo) #ff_homo[j])
    ff_inhomo_rfft = np.fft.rfft(fpulse_inhomo) #ff_inhomo[j])

    freqs = np.fft.rfftfreq(len(ff_homo[j]), (ts[1]- ts[0]) / 3.0)
    
    #plt.plot(freqs, 10.0 * np.log10(np.abs(ff_homo_rfft)) + j, color="red", alpha=0.5)
    #plt.plot(freqs, 10.0 * np.log10(np.abs(ff_inhomo_rfft)) + j, color="blue", alpha=0.5)
    plt.plot(10.0 * np.log10(np.abs(ff_homo_rfft)) + j, color="red", alpha=0.5)
    plt.plot(10.0 * np.log10(np.abs(ff_inhomo_rfft)) + j, color="blue", alpha=0.5)

    power_ratio += [np.power(np.abs(ff_inhomo_rfft[power_slice]), 2.0) / np.power(np.abs(ff_homo_rfft[power_slice]), 2.0)]
    #power_ratio += [np.sum(np.power(fpulse_inhomo, 2.0)) / np.sum(np.power(fpulse_homo, 2.0))]
    print(power_ratio[-1])

meep_power_ratio = power_ratio
meep_x_pos = np.flip(x_pos, 0)

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

    beamed_powers = []
    for i in range(len(beamed_power_profile)):
        beamed_fft = np.fft.rfft(beamed_power_profile[i])    
        beamed_powers += [np.power(np.abs(beamed_fft[power_slice]), 2.0)]
    
    beamed_powers /= np.max(beamed_powers) # Normalize
    
    angle_steps = np.linspace(-np.pi/2.0, 0.0, 100) # As determined when running the MEEP sim
    f_power_profile = scipy.interpolate.interp1d(angle_steps, beamed_powers, kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    nrays = 10000
    dir_steps = np.linspace(-np.deg2rad(90.0), -np.deg2rad(65.0), nrays) # As determined when running the raytracer
    power_at_steps = f_power_profile(dir_steps)

    f_inhomo_power = scipy.interpolate.interp1d(inhomo_xpos, power_at_steps, kind='cubic', bounds_error=False, fill_value="extrapolate")
    inhomo_power = f_inhomo_power(new_x)

    f_homo_power = scipy.interpolate.interp1d(homo_xpos, power_at_steps, kind='cubic', bounds_error=False, fill_value="extrapolate")
    homo_power = f_homo_power(new_x)

    ub_inhomo_xpos *= np.sqrt(inhomo_power)
    ub_homo_xpos   *= np.sqrt(homo_power)

plt.figure()
plt.title("Focusing Factor for Beamed Source in MEEP (@"+str(int(1000.0*freqs[power_slice]))+"MHz) and in Ray Tracer")
plt.xlabel("Distance on Surface from Source [m]")
plt.ylabel("Focusing Factor: Inhomo. Ice Power / Homo. Ice Power")
plt.plot(new_x, ub_inhomo_xpos / ub_homo_xpos, color="red", alpha=0.5, label="Ray Tracer")
plt.plot(meep_x_pos, meep_power_ratio, label="MEEP Result")
plt.grid()
plt.xlim(0.0, 350.0)
plt.ylim(1.0, 1.3)
plt.legend()


'''
plt.figure()
plt.title("Numerical Integral vs. X Pos")
plt.plot(homo_xpos, np.arange(len(homo_xpos)), color="red", alpha=0.5, label="Homo.")
plt.plot(inhomo_xpos, np.arange(len(inhomo_xpos)), color="blue", alpha=0.5, label="Inhomo.")
plt.xlabel("X Position [m]")
plt.xlim(0.0, 350.0)
#plt.ylim(1.1, 1.4)
plt.grid()
plt.legend()
'''

plt.show()
