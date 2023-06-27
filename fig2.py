from neuron import h
import matplotlib.pyplot as plt
import numpy as np

# Functions
# ----------

def gaussian_density(x: np.ndarray, mu: float, sigma: float):
    '''
    Get the value of gaussian densitiy(mu, sigma) at points x.

    Parameters:
    ----------
    x: np.ndarray
        Values to compute density at.

    mu: float 
        Mean of the gaussian.

    sigma: float 
        Standard deviation of the gaussian.

    Returns:
    ----------
    out: float
        The gaussian densitiy(mu, sigma) at points x.
    '''
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x-mu)**2) / (2 * sigma**2))

def generate_spike_times(num_spike_volley_times: int, alpha_iv: int, dt: float, sigma_iv: float, cv_t: float, P: float):
    '''
    Generate spike times.

    Parameters:
    ----------
    num_spike_volley_times: int
        Number of spike volleys to generate.

    alpha_iv: int
        Mean number of spikes per volley.

    dt: float
        Discretization step.

    sigma_iv: float
        Standard deviation of the spike probability.

    cv_t: float
        Coefficient of variation of inter-volley time.

    P: float
        Mean of inter-volley time.

    Returns:
    ----------
    grid: np.ndarray
        Time grid with step = dt (ms).
    
    spike_times: np.ndarray
        Spike times for this grid.

    '''

    # Generate the spike volley times (ms)
    spike_volley_times = np.zeros(num_spike_volley_times)
    for i in range(1, len(spike_volley_times)):
        spike_volley_times[i] = spike_volley_times[i-1] + np.random.normal(P, cv_t * P)

    # Discretize the grid
    grid = []
    for i in range(1, len(spike_volley_times)):
        num_bins =  int((spike_volley_times[i] - spike_volley_times[i-1]) / dt)
        grid.append(np.linspace(spike_volley_times[i-1], spike_volley_times[i], num_bins))
    grid = np.hstack(grid).flatten()

    # Find the indexes of spike_volley_times in the grid
    spike_volley_times_inds_in_the_grid = []
    for i in range(len(spike_volley_times)):
        ind = np.argmin(np.abs(grid - spike_volley_times[i]))
        spike_volley_times_inds_in_the_grid.append(ind)
    spike_volley_times_inds_in_the_grid = np.array(spike_volley_times_inds_in_the_grid)

    # Define and apply Gaussian filter with length of 40ms, centered around spike volleys
    spike_time_probs = []
    spike_time_inds_for_probs = [] # Some densities may overalap, and the density at overlapping values needs to be recomputed
    for i in range(len(spike_volley_times)):
        filter_center = grid[spike_volley_times_inds_in_the_grid[i]]
        filter_left = grid[spike_volley_times_inds_in_the_grid[i]] - 20
        filter_right = grid[spike_volley_times_inds_in_the_grid[i]] + 20
        gaus = gaussian_density(grid[np.argmin(np.abs(grid - filter_left)) : np.argmin(np.abs(grid - filter_right))], 
                                filter_center, sigma_iv)
        spike_time_probs.append(gaus)
        spike_time_inds_for_probs.append(np.arange(np.argmin(np.abs(grid - filter_left)), np.argmin(np.abs(grid - filter_right))))

    # Fix overlapping densities
    overlapping_inds = []
    overlapping_densities = []
    for i in range(1, len(spike_volley_times)):
        inter_inds = np.intersect1d(spike_time_inds_for_probs[i-1], spike_time_inds_for_probs[i])
        filter_center = grid[spike_volley_times_inds_in_the_grid[i-1]] + grid[spike_volley_times_inds_in_the_grid[i]]
        gaus = gaussian_density(grid[inter_inds], filter_center, 2 * sigma_iv)
        overlapping_inds.append(inter_inds)
        overlapping_densities.append(gaus)

    # Merge everything into one array
    spike_time_probs = np.hstack(spike_time_probs).flatten()
    spike_time_inds_for_probs = np.hstack(spike_time_inds_for_probs).flatten()
    overlapping_inds = np.hstack(overlapping_inds).flatten()
    overlapping_densities = np.hstack(overlapping_densities).flatten()

    final_spike_time_probs = []
    processed_overlap = []
    for i in range(len(spike_time_inds_for_probs)):
        if spike_time_inds_for_probs[i] in overlapping_inds:
            if spike_time_inds_for_probs[i] in processed_overlap: continue
            final_spike_time_probs.append(float(overlapping_densities[overlapping_inds == spike_time_inds_for_probs[i]]))
            processed_overlap.append(spike_time_inds_for_probs[i])
        else:
            final_spike_time_probs.append(float(spike_time_probs[i]))

    final_spike_time_probs.insert(0, 0) # To match the grid length
    final_spike_time_probs = np.array(final_spike_time_probs)

    # Generate spike times using Poisson process
    # https://www.researchgate.net/profile/Jorge-Jose/publication/11485024_Information_Transfer_in_Entrained_Cortical_Neurons/links/53ea851a0cf2dc24b3cd4c19/Information-Transfer-in-Entrained-Cortical-Neurons.pdf
    # (page 42)
    spike_times = []
    i = 0
    while i < len(grid):
        upper_ind = np.argmin(np.abs(grid - (grid[i] + alpha_iv * dt)))
        firing_rate = np.sum(final_spike_time_probs[i:upper_ind])
        if np.random.uniform() <= firing_rate:
            spike_times.append(i)
        i = upper_ind + 1

    return grid, spike_times

def plot_spiketrains(grid: np.ndarray, spike_times: np.ndarray):
    trace = np.zeros_like(grid)
    trace[spike_times] = 1
    plt.stem(grid, trace, markerfmt = ' ')
    plt.show()


# Script
# ----------
# (Comment / Uncomment lines)

# Generate spike times
# ----------

# Spike volley parameters
P = 26.10 # (ms) mean time in ms between spike volleys, from appendix A.4. fig 2
cv_t = 0.095 # Coefficient of variation calculated as sqrt(var/mean), from appendix fig 2.
sigma_iv = 2 # (ms) standard deviation of gaussian distribution for fig 1b
# stdev = CV_t * mean
num_spike_volleys = 8 # the number of spike volleys generated
alpha_iv = 25
dt = 0.01

# The grid is discretized with step dt. 
# The spike times are in the grid's timeframe (i.e, indices of spikes in the grid)
# To convert spike times to milliseconds, do: spike_times * dt
grid, spike_times = generate_spike_times(num_spike_volleys, alpha_iv, dt, sigma_iv, cv_t, P)

# Plot the spiketrains
# plot_spiketrains(grid, spike_times)

# Cell
# ----------

h.load_file('nrngui.hoc')
h.load_file('stdrun.hoc')

'''Building Soma'''
h.dt = 0.01  # (ms) Timestep as stated in page 4 of the Tiesinga paper (same as their bin width)
h.v_init = -65  # (mV) Initial membrane potential

soma = h.Section(name = 'soma')

# ** = from calculations based on Fellous et al. 2010 – 79.788 for L and diam
soma.L = 79.788  # soma length µm –– walt got this to work as 20
soma.diam = 79.788  # soma diameter µm  –– walt got this to work as 5
soma.cm = 1  # membrane capacitance µF/cm2  
soma.Ra = 100  # ohm-cm  **

soma.insert('leak')  # Leak channel, using the .mod file 
soma.glbar_leak = 0.1 / 1000 # (S/cm2 )
soma.el_leak = -65  # (mV)

soma.insert('na')  # Sodium channel, using the .mod file
soma.gnabar_na = 35 / 1000  # (S/cm2)
soma.ena = 55 # (mV)

soma.insert('k')  # Potassium channel, using the .mod file
soma.gkbar_k = 9 / 1000 # (S/cm2)
soma.ek = -90 # (mV)

'''Current Clamp (Background)'''
current = 4 # (uA/cm2) from fig 2 in appendix A.4. For 2a it's 4, for 2b it's 1
# surface_area = (np.pi * soma.diam ** 2 + 2 * np.pi * (soma.diam / 2)**2) / 10 ** 8 # (cm2)
current_inj = h.IClamp(soma(0.5)) #injected into the middle of the soma
current_inj.delay = 50 # (ms) To check the dynamics
current_inj.dur = spike_times[-1] * dt + 50 # (ms), 50ms after the last spike occurs
'''
FIGURING OUT AMP VALUE:
From Tiesinga, current injection is a distributed current with 4 uA/cm^2 for figure 2.
Here, we will convert this distributed current into a point process (as this is the IClamp function of NEURON).
IClamp documentation:  https://www.neuron.yale.edu/neuron/static/py_doc/modelspec/programmatic/mechanisms/mech.html
The default units taken by NEURON for a current clamp is nA, as stated here: https://www.neuron.yale.edu/neuron/static/docs/units/unitchart.html

First, we determine the surface area. The soma is a cylinder, but we don't include the top and bottom circles in NEURON.
Thus the surface area is 2*pi*r*h. h (or soma.L) = d (or soma.diam) = 79.788 um, as per Fellous et al. 2010.
Since h=2r here, we can simplify the equation to pi * h**2.
To convert the surface area from nm^2 to cm^2, multiply by 10**-8.
To convert current injection from uA/cm^2 to nA, multiply by the surface area (in cm^2) and then multiply by 1000 (uA to nA).
'''
surface_area = np.pi * soma.L**2 * 10**-8 #cm^2
current_inj.amp = current * surface_area * 1000 #nA calculated from fig 2 in appendix A.4


# Create a NetStim object
# stim = h.NetStim()

# Putting the spike train in through a VecStim object
stimulus = h.ExpSyn(soma(0.5)) # Inhibitory synapse into the middle of the cell
# Documentation for an exponential synapse (ExpSyn): https://www.neuron.yale.edu/neuron/static/py_doc/modelspec/programmatic/mechanisms/mech.html#ExpSyn
stimulus.tau = 10 # (ms) decay time constant, as per fig.s  2-10 in the appendix of Tiesinga
stimulus.e = -75 # (mV) reversal potential, from pg. 3 of Tiesinga, this is the Erev for a GABA synapse
vec = h.Vector(spike_times)
vecstim = h.VecStim() # Uses the vecevent.mod file
vecstim.play(vec)
netcon = h.NetCon(vecstim, stimulus) 

# spike_detector = h.NetCon(soma, None)  # Assuming 'soma' is the section where spikes are detected
# spike_times_vec = h.Vector()  # Create a vector to store spike times
# spike_detector.record(spike_times_vec)  # Record spike events in spike_times_vec
# print(spike_times_vec)

# netcon.delay = min(results[0]) #TODO: fix this value
netcon.weight[0] = 0.044 / 1000 # (S/cm2) #TODO: check this value

# # Set the properties of the NetStim object
# stim.interval = 10  # Average time between spikes in ms
# stim.number = 10  # Number of spikes
# stim.start = 100  # Start time of first spike
# stim.noise = 1  # Fraction of interval variability (1 for Poisson)

# # Create a NetCon object to connect the NetStim to the synapse
# nc = h.NetCon(stim, syn)

# # 'esyn': -80,    # synaptic channels reversal potential (mV)
# #     'gmax': 0,    # synaptic channels maximum conductance (uS) (default: 10e-3~50e-3)
# #     'tau1': 10,    # rise time (ms) (default: 10)
# #     'tau2': 20    # decay time (ms) (default: 20)

# Define vectors for recording variables
t_vec = h.Vector().record(h._ref_t)  # hoc object, Time stamp vector
v_vec = h.Vector().record(soma(0.5)._ref_v)  # hoc object, Membrane potential vector

# Run the simulation
h.tstop = grid[-1] + 50 # (ms)
h.run()

#change the IClamp injection value
current = 1 # (uA/cm2) from fig 2 in appendix A.4. For 2a it's 4, for 2b it's 1
current_inj.amp = current * surface_area * 1000 #nA calculated from fig 2 in appendix A.4

t_vec_2 = h.Vector().record(h._ref_t)  # hoc object, Time stamp vector
v_vec_2 = h.Vector().record(soma(0.5)._ref_v)
h.tstop = grid[-1] + 50
h.run()

#TODO: is it ok if our "spikes" are just the first time that the Vm crosses 0? Or should we make it the max? We could take the derivative and plot every 0 (but only the 0s where it goes from pos to negative)
'''
Another possibility for recording spikes:
spike_times = h.Vector()
nc = h.NetCon(axon(0.1)._ref_v, None, sec=axon)
nc.threshold = 0 * mV
nc.record(spike_times)
'''
#For now, we'll quantify spikes by counting every time Vm goes above 0.
#When this happens, we need to save the timestamp in a list (maybe change to an array)
output_t1 = [] #list with output spike times
c = 0  # a counter to make sure we only count each spike once

for i in range(len(t_vec)):
    if v_vec[i] > 0 and c == 0:
        output_t1.append(t_vec[i])
        c = 1
    elif v_vec[i] < 0:
        c = 0

output_t2 = [] #list with output spike times
c = 0  # a counter to make sure we only count each spike once

for i in range(len(t_vec_2)):
    if v_vec_2[i] > 0 and c == 0:
        output_t2.append(t_vec_2[i])
        c = 1
    elif v_vec_2[i] < 0:
        c = 0

outputs = [] #a list of lists of spike times for each trial
outputs.append(output_t1)
outputs.append(output_t2)
'''
we keep track of the first t_vec value when v_vec > 0 and the last value
average the t_vec
'''

# Plot membrane potentials
fig, ax = plt.subplots(nrows = 3, ncols = 1, sharex = True)
ax[0].plot(np.array(t_vec), np.array(v_vec))
ax[1].plot(np.array(t_vec_2), np.array(v_vec_2))

colors1 = [f'C{i}' for i in range(2)]
ax[2].eventplot(outputs, colors=colors1)


# Previous labels to plot membrane potential over time
#axs[3].plot(t_vec, v_vec)
#axs[3].set_ylim([-75, -73])
#axs[3].set_xlabel('Time (ms)')
#axs[3].set_ylabel('Membrane Potential (mV)')


plt.savefig('Fig2btrial')
plt.show()