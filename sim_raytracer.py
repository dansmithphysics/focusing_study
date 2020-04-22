import matplotlib.pyplot as plt
import numpy as np
import copy 

homo = False
plot = False
beamed = False

nrays = 10000

# starting points and direction
x0 = 0.0
if(beamed):
    z0 = 80.0
else:
    z0 = 0
    
norm_vec = np.array([0.0, 0.0, 1.0]) # Vector norm

nsteps = 5000
ice_layer = -900.0

dir_steps = np.linspace(-np.deg2rad(90.0), -np.deg2rad(65.0), nrays)

def step_length(z, ice_layer):
    max_step = 10.0 

    if(np.abs(z - ice_layer) < 2.0):
        step_length_ = 0.1
    elif(np.abs(z - ice_layer) < 2.0 * max_step):
        step_length_ = 1.0
    elif(np.abs(z) < 2.0):
        step_length_ = 0.1
    elif(np.abs(z) < 2.0 * max_step ):
        step_length_ = 1.0
    elif(np.abs(z) < 4.0 * max_step):
        step_length_ = 2.0
    else:
        step_length_ = max_step

    return step_length_

def index_function(z, ice_layer, homo):
    n2 = 1.35915

    if(z >= 0.0):
        n2 = 1.35915
    elif(z < 0.0 and z > ice_layer):
        if(homo):
            n2 = 1.35915
        else:
            n2 = 1.35915 + 0.429568*(1.0 - np.exp(0.0134459 * z))
    else:
        n2 = 10e10

    return n2

if(plot):
    f1 = plt.figure(figsize=(7, 6))


last_points_x = []
total_lengths = []

for i, dir_0 in enumerate(dir_steps):

    print(str(i)+") "+str(i / float(len(dir_steps))*100.0)+"%, "+str(dir_0)+" "+str(x0))

    x_plot, z_plot = [], []
    pos = np.array([x0, 0.0, z0])
    total_length = 0.0
    
    i_vec = np.array([np.cos(dir_0), 0.0, np.sin(dir_0)])
    t_vec = np.array([0.0, 0.0, 0.0])

    n_old = index_function(pos[2], ice_layer, homo)
    for i in range(nsteps):

        #print(i, pos, i_vec, n_old)

        n1 = n_old
        n2 = index_function(pos[2], ice_layer, homo)
        
        if(pos[2] < ice_layer):
            i_vec[2] *= -1
            pos[2] = ice_layer
            n2 = index_function(pos[2], ice_layer, homo)
                
        # Stupid solution to not having the correct normal vector sign
        going_down = False
        if(i_vec[2] < 0.0):
            going_down = True
                
        if not(going_down):
            i_vec[2] *= -1

        t_vec = np.sqrt(1.0 - np.power(n1 / n2, 2.0)*(1.0 - np.power(np.dot(norm_vec, i_vec), 2.0))) * norm_vec
        t_vec += (n1/n2)*(i_vec - np.dot(norm_vec, i_vec)*norm_vec)

        t_vec /= np.sqrt(np.sum(np.power(t_vec, 2.0)))
        
        if(going_down):
            t_vec[2] *= -1.0 

        step_length_ = step_length(pos[2], ice_layer)

        total_length += step_length_
            
        pos += step_length_ * t_vec
        i_vec = copy.deepcopy(t_vec)
        n_old = n2
       
        if(plot):
            x_plot += [pos[0]]
            z_plot += [pos[2]]

        if(pos[2] > 0.0 and not(going_down)):
            break

    if(plot):
        plt.plot(x_plot, z_plot, color='red', alpha=0.1)

    last_points_x += [pos[0]]
    total_lengths += [total_length]

if(beamed):
    if(homo):
        np.save("data_focus_homo_beamed_raytracer_xpos.npy", last_points_x)
        np.save("data_focus_homo_beamed_raytracer_length.npy", total_lengths)
    else:
        np.save("data_focus_inhomo_beamed_raytracer_xpos.npy", last_points_x)
        np.save("data_focus_inhomo_beamed_raytracer_length.npy", total_lengths)
else:
    if(homo):
        np.save("data_focus_homo_raytracer_xpos.npy", last_points_x)
        np.save("data_focus_homo_raytracer_length.npy", total_lengths)
    else:
        np.save("data_focus_inhomo_raytracer_xpos.npy", last_points_x)
        np.save("data_focus_inhomo_raytracer_length.npy", total_lengths)

if(plot):
    plt.grid(True)

    if(homo):
        plt.title("Ray Tracing Result in Homogeneous Ice")
    else:
        plt.title("Ray Tracing Result in Inhomogeneous Ice")
    plt.xlim(0.0, 450.0)
    plt.ylim(-950.0, 50.0)
    plt.ylabel("Depth [m]")
    plt.xlabel("Surface Distance From Source [m]")

    plt.figure()
    plt.hist(last_points_x, bins=200, range=(0.0, 400.0))
    plt.xlabel("X-Position at Ice Exit [m]")

    plt.figure()
    plt.hist(total_lengths, bins=200, range=(np.min(total_lengths), np.max(total_lengths)))
    plt.xlabel("Total Propagation Distance [m]")

    plt.show() 
