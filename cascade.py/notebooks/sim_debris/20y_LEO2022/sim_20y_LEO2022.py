import sys, os
# We parse the script inputs
try:
    cpus = int(sys.argv[1])
    runs = int(sys.argv[2])
    years = float(sys.argv[3])
except: 
    print("USAGE:")
    print("{0} <cpus> <runs> <years>".format(sys.argv[0]))
    sys.exit(2)

# core imports
import pykep as pk
import numpy as np
import pickle as pkl
import cascade as csc
from copy import deepcopy

#initializing the counters
collision_count_vec = np.zeros(runs) 
decay_count_vec = np.zeros(runs)

for run in range(runs):
    
    # create a folder to store the results of the simulation
    folder_name = 'run_'+str(run+1)
    os.mkdir(folder_name)
    
    # loading the initial LEO population: 
    # - position and collision radius are in [m]
    # - velocity is in [m/s]
    with open("../../data/debris_simulation_ic_LEO_2022.pk", "rb") as file:
        r_ic, v_ic, collision_radius, to_satcat_index, satcat, debris = pkl.load(file)
    
    # adding a noise to the positions with a uniformly distributed number in [−1000, 1000] meters
    for i in range(len(r_ic)):
        noise = np.random.uniform(-1000, 1000, 3)  
        r_ic[i, :] += noise  # add the noise to the ith row

    # reference epoch for the initial conditions
    t0_jd = pk.epoch_from_iso_string("20220301T000000").jd # Julian date corresponding to 2022-Mar-01 00:00:00

    # We now extract from the satcat list and store it into separate arrays the objects' BSTAR coefficients.
    # Array containing the BSTAR coefficient in the SI units used
    BSTARS = []
    for idx in to_satcat_index:
        BSTARS.append(float(satcat[idx]["BSTAR"]))
    # We transform the BSTAR in SI units
    BSTARS = np.array(BSTARS) / pk.EARTH_RADIUS
    # .. and remove negative BSTARS (this can happen for objects that where performing orbital manouvres during the tle definition) setting the value to zero in those occasions.
    BSTARS[BSTARS<0] = 0.

    #Building the dynamical system to integrate
    dyn =  csc.dynamics.simple_earth(J2=True, J3=True, J4=True, C22S22=True, sun=True, moon=True, SRP=True, drag=True)
    csc.set_nthreads(cpus)

    # We now define the radius that will be used to check for decayed objects. We will assume that once the position of some object is below 150km altitude, the object can be considered as decayed.
    def remove_particle(idx, r_ic, v_ic, BSTARS, to_satcat_index, collision_radius):
        r_ic = np.delete(r_ic, idx, axis=0)
        BSTARS = np.delete(BSTARS, idx, axis=0)
        v_ic = np.delete(v_ic, idx, axis=0)
        to_satcat_index = np.delete(to_satcat_index, idx, axis=0)
        collision_radius = np.delete(collision_radius, idx, axis=0)
        return r_ic, v_ic, BSTARS, to_satcat_index, collision_radius

    # Before starting we need to remove all particles inside our playing field
    min_radius = pk.EARTH_RADIUS + 150000. # all the particles inside the radius of 150 km need to be removed
    inside_the_radius = np.where(np.linalg.norm(r_ic, axis=1) < min_radius)[0]
    # print("Removing ", len(inside_the_radius), " orbiting objects:")
    # for idx in inside_the_radius:
    #     print(satcat[to_satcat_index[idx]]["OBJECT_NAME"], "-", satcat[to_satcat_index[idx]]["OBJECT_ID"])
    r_ic, v_ic, BSTARS, to_satcat_index, collision_radius = remove_particle(inside_the_radius, r_ic, v_ic, BSTARS, to_satcat_index, collision_radius)

    # Prepare the data in the shape expected by the simulation object.
    ic_state = np.hstack([r_ic, v_ic, collision_radius.reshape((r_ic.shape[0], 1))])
    # BSTARS
    p0 = BSTARS.reshape((r_ic.shape[0], 1)) # BSTAR coefficient (SI units) 

    # Cr*(A/m)
    Cr = 1
    # Volume of all satellites starting from the collision radius
    volume = (4/3) * (np.pi) * ((collision_radius.reshape((r_ic.shape[0], 1))) ** 3) # [m^3]
    # iridium satellite charateristics
    mass_iridium = 689 # [kg] (mass of a first-generation Iridium satellite)
    radius_iridium = 1.0281416736168953 # [m] (radius of a first-generation Iridium satellite)
    # average density of the satellites
    av_den = mass_iridium / ((4/3) * (np.pi) * (radius_iridium ** 3)) # [kg/m^3]
    # we compute M for each debris
    mass = av_den * volume # [m]
    # computing the AOM for each satellite
    A = (np.pi) * ((collision_radius.reshape((r_ic.shape[0], 1))) ** 2)
    AOM = A / mass
    # defining p1 as cr * A/m 
    p1 = Cr * AOM

    # building the pars vectors
    pars = np.hstack((p0, p1))
    collisional_step = 225

    # we now set up the simulation
    sim = csc.sim(ic_state, collisional_step, dyn=dyn, pars=pars, reentry_radius=min_radius, n_par_ct = 25)
    cascade.set_logger_level_trace()

    # We define here the simulation starting time knowing that in the dynamics t=0 corresponds to 1st Jan 2000 12:00. 
    t0 = (t0_jd - pk.epoch_from_iso_string("20000101T120000").jd) * pk.DAY2SEC
    sim.time = t0

    # Running the simulation
    import time
    final_t = t0 + ((365.25 * years) * pk.DAY2SEC) # propagating for "years" years. this is not the time step.
    # print("Starting the simulation:", flush=True)
    start = time.time()

    #initializing the counters inside the cycle
    collision_count = 0
    decay_count = 0
    current_year = 0

    while sim.time < final_t:

        years_elapsed = (sim.time - t0) * pk.SEC2DAY // 365.25

        if years_elapsed == current_year:
            with open("run_"+str(run+1)+"/year_"+str(current_year)+".pk", "wb") as file:
                pkl.dump((sim.state, sim.pars, to_satcat_index), file)
            current_year += 1

        # performing a step of the simulation
        oc = sim.step()

        # if the simulation is interrupted by a collision, we remove the 2 particles involved
        if oc == csc.outcome.collision:

            collision_count +=1

            pi, pj = sim.interrupt_info
            # We log the event to file
            satcat_idx1 = to_satcat_index[pi]
            satcat_idx2 = to_satcat_index[pj]
            days_elapsed = (sim.time - t0) * pk.SEC2DAY
            with open("run_"+str(run+1)+"/collision_log.txt", "a") as file_object:
                file_object.write(
                    f"{days_elapsed}, {satcat_idx1}, {satcat_idx2}, {sim.state[pi]}, {sim.state[pj]}\n")
            # We remove the objects and restart the simulation
            sim.remove_particles([pi,pj])
            to_satcat_index = np.delete(to_satcat_index, [max(pi,pj)], axis=0)        
            to_satcat_index = np.delete(to_satcat_index, [min(pi,pj)], axis=0)

        # if the simulation is interrupted by a reentry, we remove the particle involved
        elif oc == csc.outcome.reentry:
            
            decay_count += 1

            pi = sim.interrupt_info
            # we log the event to file
            satcat_idx = to_satcat_index[pi]
            days_elapsed = (sim.time - t0) * pk.SEC2DAY
            with open("run_"+str(run+1)+"/decay_log.txt", "a") as file_object:
                file_object.write(f"{days_elapsed},{satcat_idx}\n")
            # We remove the re-entered object and restart the simulation
            sim.remove_particles([pi])
            to_satcat_index = np.delete(to_satcat_index, [pi], axis=0)  

    collision_count_vec[run] = collision_count
    decay_count_vec[run] = decay_count
    print("run number", run + 1,", number of collisions:",collision_count,", number of decays:",decay_count, "\n")

end = time.time()
elapsed = end - start
# print("Elapsed [s]: ", end - start)
# print("Time projected to simulate 20 years is ", elapsed / 30 * 20 *365.25 / 60 / 60, " hours")
print("The number of collisions for each run is: ", collision_count_vec, "while its mean is: ", np.mean(collision_count_vec))
print("The number of decays for each run is: ", decay_count_vec, "while its mean is: ", np.mean(decay_count_vec))

# we store the results in a .txt file
with open("20y_LEO2022.txt", 'w') as file:
    output_str = "The number of collisions for each run is: {}\nwhile its mean is: {}\n".format(collision_count_vec, np.mean(collision_count_vec))
    file.write(output_str)
    output_str = "The number of decays for each run is: {}\nwhile its mean is: {}\n".format(decay_count_vec, np.mean(decay_count_vec))
    file.write(output_str)
