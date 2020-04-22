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

beamed_profile = False
homo = True
save_files = True
file_base = "./"

resolution = 2
end_time = 4000 
sx = 800  # 800 m, size of cell in X direction
sy = 1000 # 1 km, size of cell in Y direction

cell = mp.Vector3(sx,sy,0)
size = mp.Vector3(0,0.0)
size_objects = 1.0 # a metre

if(beamed_profile):
    output_file_title = "beamed_power_profile"
else:
    if(homo):
        output_file_title = "homo_beamed"
    else:
        output_file_title = "inhomo_beamed"

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

    if(beamed_profile):
        print("Beamed Power")
    else:
        print("Isotropic Power")

pml_thicc = 20.0
pml_layers = [mp.PML(pml_thicc)]#, mp.Y)]

symmetries = [mp.Mirror(direction=mp.X, phase=1)]

center_slice_list = []
start_times = []

if(beamed_profile):
    angle_steps = np.linspace(-np.pi/2.0, 0.0, 100)
    r = 300
    for i, jjj in enumerate(angle_steps):
        center_slice_list += [mp.Vector3(r * np.cos(angle_steps[i]), -sy/2.0 + pml_thicc - r * np.sin(angle_steps[i]), 0.0)]

        if(homo):
            start_times += [0.0]
        else:
            start_times += [0.0]
else:

    for i, jjj in enumerate(np.linspace(- sx / 2.0 + pml_thicc + 10.0, 0.0, 40)):
        center_slice_list += [mp.Vector3(jjj, -sy/2.0 + pml_thicc, 0.0)] # Top 
        if(homo):
            start_times += [0.0]
        else:
            start_times += [0.0]
            
    for i, jjj in enumerate(np.linspace(- sx / 2.0 + pml_thicc + 10.0, 0.0, 40)):    
        center_slice_list += [mp.Vector3(jjj, -sy/2.0 + pml_thicc, 0.0)] # Top, returned
        if(homo):
            start_times += [2700 - 250.0]
        else:
            start_times += [3300 - 250.0]

    for i, jjj in enumerate(np.linspace(- sx / 2.0 + pml_thicc + 10.0, 0.0, 40)):    
        center_slice_list += [mp.Vector3(jjj, sy/2.0 - 4.0 * pml_thicc, 0.0)] # Bottom
        if(homo):
            start_times += [1100.0]
        else:
            start_times += [1400.0]
    
sigma = 5.0 # Beam width
src_pt = mp.Vector3(0.0, -sy/2.0 + 2.0 * pml_thicc, 0.0)

def gaussian_beam(sigma, x0):
    def _gaussian_beam(x):
        return np.exp(-x.dot(x) / (2.0*sigma**2))
    return _gaussian_beam
    
def f_source(t):
    t0 = 10.0
    sig = 1.0
    return np.exp(-0.5*np.power((t - t0) / sig, 2.0))

sources = [mp.Source(mp.CustomSource(src_func = f_source, start_time=0, end_time = 40),
                     component=mp.Ez,
                     center=mp.Vector3(0.0, -sy/2.0 + pml_thicc, 0.0),
                     size=mp.Vector3(sx - 2.0 * pml_thicc, 0, 0), 
                     amp_func=gaussian_beam(sigma, src_pt))]


geometry = []                
block_size = size_objects

for y_ in range(0, 1000, 1): 
    center = mp.Vector3(0, y_ - sy / 2.0 + pml_thicc, 0)

    # Now, find a depth, and set index of refraction from that
    if(homo):
        index = 1.35915 
    else:
        index = 1.35915 + 0.429568*(1.0 -np.exp(-0.0134459 * y_))

    geometry.append(mp.Block(center=center,
                             size=mp.Vector3(sx, 1.0, 0),
                             material=mp.Medium(index=index)))    
# Cut down reflections at the tippy top by replacing air with top-of-fern / loose snow index ice
geometry.append(mp.Block(center=mp.Vector3(0.0, -sy / 2.0 + pml_thicc / 2.0, 0.0), 
                         size=mp.Vector3(sx, pml_thicc, 0),
                         material=mp.Medium(index=1.35915)))

# Perfect bottom, lol
geometry.append(mp.Block(center=mp.Vector3(0, sy / 2.0 - pml_thicc, 0),
                         size=mp.Vector3(sx, 2.0 * pml_thicc, 0), 
                         material=mp.Medium(index=1e10)))

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
                    symmetries=symmetries,
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


