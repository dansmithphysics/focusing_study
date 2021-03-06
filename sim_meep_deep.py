from __future__ import division
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import scipy.fftpack
import scipy.interpolate
import pandas as pd
import glob
import pickle 
import sys, os

seed = 31456
np.random.seed(seed+1)

homo = False
save_files = False
file_base = "./"

resolution = 1
if(homo):
    end_time = 5750
else:
    end_time = 7500
sx = 4000  # 100 m, size of cell in X direction
sy = 880 # 4 km, size of cell in Y direction

cell = mp.Vector3(sx,sy,0)
size = mp.Vector3(0,0.0)
size_objects = 1.0 # a metre

if(homo):
    output_file_title = "deep_homo"
else:
    output_file_title = "deep_inhomo"

if(mp.am_master()):
    print("Seed = ", seed)
    print("End Time =", end_time)
    print("Resolution =", resolution)
    print("(sx, sy) = (" + str(sx) + ", " + str(sy) + ")")
    print("Output file = ", file_base, output_file_title)
    if(homo):
        print("Homo.")
    else:
        print("InHomo.")

pml_thicc = 20.0
pml_layers = [mp.PML(pml_thicc)]

symmetries = []

source_center = mp.Vector3(-sx/2.0 + 2.0 * pml_thicc, -sy/2.0 + 820.0, 0.0)

center_slice_list = []
start_times = []
for i, jjj in enumerate(np.linspace(- sx / 2.0 + pml_thicc, sx / 2.0 - pml_thicc, 100)):
    center_slice_list += [mp.Vector3(jjj, -sy/2.0 + 120.0, 0.0)] # Reciever at 100 m
    if(homo):
        start_times += [-100.0 + 1.35 * np.sqrt(np.sum(np.power(center_slice_list[-1] - source_center, 2.0)))]
    else:
        start_times += [-100.0 + 1.75 * np.sqrt(np.sum(np.power(center_slice_list[-1] - source_center, 2.0)))]

for i, jjj in enumerate(np.linspace(- sx / 2.0 + pml_thicc, sx / 2.0 - pml_thicc, 100)):
    center_slice_list += [mp.Vector3(jjj, -sy/2.0 + 170.0, 0.0)] # Reciever at 150 m
    if(homo):
        start_times += [-100.0 + 1.35 * np.sqrt(np.sum(np.power(center_slice_list[-1] - source_center, 2.0)))]
    else:
        start_times += [-100.0 + 1.75 * np.sqrt(np.sum(np.power(center_slice_list[-1] - source_center, 2.0)))]

for i, jjj in enumerate(np.linspace(- sx / 2.0 + pml_thicc, sx / 2.0 - pml_thicc, 100)):
    center_slice_list += [mp.Vector3(jjj, -sy/2.0 + 220.0, 0.0)] # Reciever at 200 m
    if(homo):
        start_times += [-100.0 + 1.35 * np.sqrt(np.sum(np.power(center_slice_list[-1] - source_center, 2.0)))]
    else:
        start_times += [-100.0 + 1.75 * np.sqrt(np.sum(np.power(center_slice_list[-1] - source_center, 2.0)))]

sigma = 5.0 #1.5 # beam width

def gaussian_beam(sigma, x0):
    def _gaussian_beam(x):
        return np.exp(-x.dot(x) / (2.0*sigma**2))
    return _gaussian_beam

src_pt = mp.Vector3(0.0, -sy/2.0 + 2.0 * pml_thicc, 0.0)

def f_source(t):
    t0 = 10.0
    sig = 1.0
    return np.exp(-0.5*np.power((t - t0) / sig, 2.0))

sources = [mp.Source(mp.CustomSource(src_func = f_source, start_time=0, end_time = 40),
                     component=mp.Ez,
                     center=source_center, 
                     size=mp.Vector3(0,0,0))]

geometry = []                
block_size = size_objects

for y_ in range(0, 1000, 1): 
    center = mp.Vector3(0, y_ - sy / 2.0, 0)

    # Now, find a depth, and set index of refraction from that
    if(homo):
        index = 1.35915 
    else:
        index = 1.35915 + 0.429568*(1.0 -np.exp(-0.0134459 * (y_ - pml_thicc))) # Make it 60 meters deep at top

    geometry.append(mp.Block(center=center,
                             size=mp.Vector3(sx, 1.0, 0),
                             material=mp.Medium(index=index)))    
# Cut down reflections at the tippy top

geometry.append(mp.Block(center=mp.Vector3(0.0, -sy / 2.0 + pml_thicc / 2.0, 0.0), 
                         size=mp.Vector3(sx, pml_thicc, 0),
                         material=mp.Medium(index=1.35915)))

# Set up framework to get data from simulation
slice_from_sim = [[] for i in range(len(center_slice_list))]
def get_slice_from_sim(sim):
    for i in range(len(center_slice_list)):
        if(sim.meep_time() > start_times[i] and sim.meep_time() < start_times[i]+500.0):
            slice_from_sim[i] += [sim.get_array(center=center_slice_list[i], size=size, component=mp.Ez)]
        
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    eps_averaging=True,
                    sources=sources,
                    resolution=resolution)

sim.run(mp.at_every(1.0/10.0, get_slice_from_sim), until=end_time)

if(not(save_files)):
    print("Plotting", output_file_title)    
    
    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    
    if(mp.am_master()):
        
        ex_x_lw = -sx / 2.0 / size_objects
        ex_x_up =  sx / 2.0 / size_objects
        ex_y_lw = -sy / 2.0 / size_objects
        ex_y_up =  sy / 2.0 / size_objects
        
        plt.figure(dpi=100)        
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary', extent=(ex_x_lw, ex_x_up, ex_y_lw, ex_y_up))
        #plt.imshow(ez_running.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
        
        plt.figure(dpi=100)
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary', extent=(ex_x_lw, ex_x_up, ex_y_lw, ex_y_up))
        plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9, extent=(ex_x_lw, ex_x_up, ex_y_lw, ex_y_up))
        for center in center_slice_list[::-1]:
            plt.plot(center.x / size_objects, -1.0 * center.y / size_objects, marker='o', markersize=5, color='orange')
        plt.xlabel("Length [m]")
        plt.ylabel("Length [m]")

        plt.figure(dpi=100)            
        for i in range(len(center_slice_list)):
            time = [j * 1.0/resolution for j in range(len(slice_from_sim[i]))]
            if(slice_from_sim[i] != []):
                plt.plot(time, np.array(slice_from_sim[i]) / np.max(np.abs(slice_from_sim[i])) + float(i)*1.0, alpha=0.5)

        plt.xlabel("time")
        plt.ylabel("Electric field")
        plt.axhline(y= 0)
        plt.show()
        
else:        
    if(mp.am_master()):
        print(output_file_title)
        
        output_file_name = file_base + "/data_focus_efield_" + output_file_title + "_meep_" + str(seed) + ".npy"
        np.save(output_file_name, slice_from_sim)
        
        geometry_file_name = file_base + "/data_focus_geometry_" + output_file_title + "_meep_" + str(seed) + ".pkl"
        with open(geometry_file_name, 'wb') as config_dictionary_file:    
            pickle.dump(geometry, config_dictionary_file)
            
        config_file_name = file_base + "/data_focus_config_" + output_file_title + "_meep_" + str(seed) + ".csv"
        df = pd.DataFrame({"sx": [sx],
                           "sy": [sy],
                           "resolution": [resolution],
                           "it_is_the_end_times": [end_time],
                           "seed": [seed]})

        df.to_csv(config_file_name)

        print("Done with", output_file_title)


