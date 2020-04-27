import numpy as np
import matplotlib.pyplot as plt
                   
#ff_homo = np.load("data_focus_efield_homo_meep_31456.npy", allow_pickle=True)
#ff_inhomo = np.load("data_focus_efield_inhomo_meep_31456.npy", allow_pickle=True)
ff_homo = np.load("data_focus_efield_homo_beamed_meep_31456.npy", allow_pickle=True)
ff_inhomo = np.load("data_focus_efield_inhomo_beamed_meep_31456.npy", allow_pickle=True)

sx = 800
sy = 1000
pml_thicc = 20.0
x_pos = np.linspace(0.0, sx / 2.0 - pml_thicc, 40)

power_ratio = []
power_slice = 100

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

    freqs = np.fft.rfftfreq(len(ff_homo[j]), (ts[1]- ts[0]) / 3.33)
    
    #plt.plot(freqs, 10.0 * np.log10(np.abs(ff_homo_rfft)) + j, color="red", alpha=0.5)
    #plt.plot(freqs, 10.0 * np.log10(np.abs(ff_inhomo_rfft)) + j, color="blue", alpha=0.5)
    plt.plot(10.0 * np.log10(np.abs(ff_homo_rfft)) + j, color="red", alpha=0.5)
    plt.plot(10.0 * np.log10(np.abs(ff_inhomo_rfft)) + j, color="blue", alpha=0.5)

    power_ratio += [np.power(np.abs(ff_inhomo_rfft[power_slice]), 2.0) / np.power(np.abs(ff_homo_rfft[power_slice]), 2.0)]
    #power_ratio += [np.sum(np.power(fpulse_inhomo, 2.0)) / np.sum(np.power(fpulse_homo, 2.0))]
    print(power_ratio[-1])

    
plt.figure()
plt.title("Focusing Factor for Isotropic Source in MEEP (@"+str(int(1000.0*freqs[power_slice]))+"MHz) and in Ray Tracer")
plt.xlabel("Distance on Surface from Source [m]")
plt.ylabel("Focusing Factor: Inhomo. Ice Power / Homo. Ice Power")
plt.grid()
plt.xlim(0.0, 350.0)
plt.ylim(1.1, 1.4)
plt.plot(np.flip(x_pos, 0), power_ratio, label="MEEP Result")
plt.legend()
plt.show()
