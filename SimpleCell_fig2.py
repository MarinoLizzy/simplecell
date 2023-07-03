import neuron
from neuron import h
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def gaussian_density(x: np.ndarray, mu: float, sigma: float):
    '''
    Purpose: Get the value of gaussian densitiy(mu, sigma) at points x.
    Parameters:
        x: (np.ndarray) Values to compute density at.
        mu: (float) Mean of the gaussian.
        sigma: (float) Standard deviation of the gaussian.
    Returns:
        out: (float) The gaussian densitiy(mu, sigma) at points x.
    '''
    # return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x-mu)**2) / (2 * sigma**2))
    return (1/(np.sqrt(2*np.pi*sigma**2))) * np.exp(-((x-mu)**2) / (2*sigma**2))

def generate_spike_times():
    '''
    Purpose: To generate spike volley times based on P (mean interval) and CV_t (coefficient of variance)
    Values:
        spike_volleys (array): timestamps of the mean of each spike volley
        STP_x (list of arrays): 
        STP_gaus (list of floats): a Gaussian curve for each spike volley
        spike_probs (list): flattened version of STP_x and STP_gaus
        spike_times (list of floats): timestamps of each individual spike
        flat_x: a flattened version of STP_x
    Output: flat_x, volley_trace, spike_probs, spike_trace, spike_times
    '''
    spike_volleys = np.zeros(num_spike_volleys) #Generates an empty array that's the size of the number of spike volleys
    for i in range(1, len(spike_volleys)): #Fills the spike_volleys array with spike times - what you see in fig 1a
        spike_volleys[i] = spike_volleys[i - 1] + np.random.normal(P, CV_t * P)

    # Define and apply Gaussian filter w/ length of 40ms, centered around spike volleys
    STP_x = [] #spike time probability, list of arrays
    STP_gaus = [] #list of arrays (one array for each spike volley)
    spike_probs = []
    
    for v_time in range(1, len(spike_volleys) - 1):
        gaus_start = (spike_volleys[v_time] + spike_volleys[v_time - 1]) / 2 #avg time of two spike volleys, on the left side
        gaus_end = (spike_volleys[v_time] + spike_volleys[v_time + 1]) / 2 #avg time of two spike volleys, on the right side
        x_length = int(gaus_end - gaus_start + 1) 
        x = np.linspace(gaus_start, gaus_end, x_length) #calculates x_length evenly-spaced samples between gaus_start and gaus_end
        
        if (spike_volleys[v_time] > 300) and (spike_volleys[v_time] < 700): #sigma_iv is 8ms before 300 and after 700. It is 2ms within this interval
            # sigma_IV = 2
            gaus = gaussian_density(x, spike_volleys[v_time], 2) #generating Gaussian curves (as defined above)
            # well it works with this mega small sigma_IV: 0.002 ##TODO: fix this! sigma_IV should be 2 here :(
            #also works at 0.005 (sometimes) or 0.006
        else:
            # sigma_IV = 8
            gaus = gaussian_density(x, spike_volleys[v_time], 8)

        STP_x.append(x)
        STP_gaus.append(gaus)
    

    # Flatten into 1D arrays for plotting
    flat_x = [x for volley in STP_x for x in volley]
    spike_probs = []
    for volley in STP_gaus:
        spike_probs.extend((volley / np.max(volley)).tolist())
    #flat_gaus = [gaus for volley in STP_gaus for gaus in volley]
    #spike_probs.append(np.array(flat_gaus) / np.max(flat_gaus))
    #spike_probs = [prob for volley in spike_probs for prob in volley]

    volley_trace = np.zeros_like(flat_x)
    for i in range(1, len(spike_volleys) - 1):
        volley_trace[np.argmin(np.abs(flat_x - spike_volleys[i]))] = 1

    # Generate spike times using Poisson process
    spike_times = []
    for i in range(len(flat_x)):
        #spike_count = np.random.poisson(flat_gaus[i])  # sampling from a Poisson distribution
        #spike_times.extend([flat_x[i]] * spike_count)  # add each spike time multiple times according to spike_count
        if np.random.uniform() <= spike_probs[i]:
            spike_times.append(i)

    spike_trace = np.zeros_like(flat_x)
    spike_trace[spike_times] = 1

    return flat_x, volley_trace, spike_probs, spike_trace, spike_times

'''Spike Volley Parameters'''
P = 26.10 #ms mean time (ms) between spike volleys, from appendix fig 2
CV_t = 0.095 #Coefficient of variance calculated as sqrt(var/mean), from appendix fig 2.
# sigma_IV = 3 #ms standard deviation of gaussian distribution for fig 1b
# stdev = CV_t * mean
num_spike_volleys = 40 #the number of spike volleys generated


# flat_x: the x-axis (time)
# volley_trace: the mean of each spike volley
# spike_probs: gaussian curves for each volley
# spike_trace: 
# spike_times: timestamps of each individual spike

'''Initialize the soma'''
h.load_file('nrngui.hoc')
h.load_file('stdrun.hoc')

h.dt = 0.01  # (ms) Timestep as stated in page 4 of the Tiesinga paper (same as their bin width)
h.v_init = -65  # (mV) Initial membrane potential

soma = h.Section(name = 'soma')

# ** = from calculations based on Fellous et al. 2010 – 79.788 for L and diam (turns out this value doesn't matter-see note at the bottom)
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

'''insert the spike train via an inhibitory synapse'''
spike_input = h.ExpSyn(soma(0.5))  # Inhibitory synapse into the middle of the cell
# Documentation for an exponential synapse (ExpSyn): https://www.neuron.yale.edu/neuron/static/py_doc/modelspec/programmatic/mechanisms/mech.html#ExpSyn
spike_input.tau = 10 # (ms) decay time constant, as per fig.s  2-10 in the appendix of Tiesinga
spike_input.e = -75 # (mV) reversal potential, from pg. 3 of Tiesinga, this is the Erev for a GABA synapse
# vecstim = h.VecStim() # Uses the vecevent.mod file


## Here's an option of recording spikes:
# netcon.threshold = 20 
# spike_recording = h.Vector()
# netcon.record(spike_recording)

'''Add an excitatory background current via a current clamp'''
current = 4 # (uA/cm2) from fig 2 in appendix A.4. For 2a it's 4, for 2b it's 1
# surface_area = (np.pi * soma.diam ** 2 + 2 * np.pi * (soma.diam / 2)**2) / 10 ** 8 # (cm2)
current_inj = h.IClamp(soma(0.5)) #injected into the middle of the soma
current_inj.delay = 0 # (ms) To check the dynamics
current_inj.dur = 900 # (ms), 50ms after the last spike occurs
# '''
# FIGURING OUT AMP VALUE:
# From Tiesinga, current injection is a distributed current with 4 uA/cm^2 for figure 2.
# Here, we will convert this distributed current into a point process (as this is the IClamp function of NEURON).
# IClamp documentation:  https://www.neuron.yale.edu/neuron/static/py_doc/modelspec/programmatic/mechanisms/mech.html
# The default units taken by NEURON for a current clamp is nA, as stated here: https://www.neuron.yale.edu/neuron/static/docs/units/unitchart.html

# First, we determine the surface area. The soma is a cylinder, but we don't include the top and bottom circles in NEURON.
# Thus the surface area is 2*pi*r*h. h (or soma.L) = d (or soma.diam) = 79.788 um, as per Fellous et al. 2010.
# Since h=2r here, we can simplify the equation to pi * h**2.
# To convert the surface area from nm^2 to cm^2, multiply by 10**-8.
# To convert current injection from uA/cm^2 to nA, multiply by the surface area (in cm^2) and then multiply by 1000 (uA to nA).
# '''
surface_area = np.pi * soma.L * soma.diam * 10**-8 #cm^2
current_inj.amp = current * surface_area * 1000 #nA calculated from fig 2 in appendix A.4
# print(current_inj.amp)


'''Running and graphing multiple trials'''
#Trial 1 will be plotted as graph a
#Trials 1-10 will be included in the raster plot
#Trial 11 will have a lowered current injection and be plotted as graph b
num_trials = 11
runtime = 900 #ms
fig, axs = plt.subplots(nrows=4, sharex=True)
raster_spikes = []
colors1 = [f'C{i}' for i in range(num_trials)]

for i in range(num_trials): 
    if i == 10:
        current_inj = None
        current = 1 # (uA/cm2) from fig 2 in appendix A.4. For 2a it's 4, for 2b it's 1
        # surface_area = (np.pi * soma.diam ** 2 + 2 * np.pi * (soma.diam / 2)**2) / 10 ** 8 # (cm2)
        current_inj = h.IClamp(soma(0.5)) #injected into the middle of the soma
        current_inj.delay = 0 # (ms) To check the dynamics
        current_inj.dur = 900 # (ms), 50ms after the last spike occurs
    netcon = None #clear the h.NetCon object
    vecstim = None
    flat_x, volley_trace, spike_probs, spike_trace, spike_times = generate_spike_times()  
    # print(spike_times[-1])
    vecstim = h.VecStim() # Uses the vecevent.mod file
    vecstim.play(h.Vector(spike_times))
    netcon = h.NetCon(vecstim, spike_input) 
    netcon.weight[0] = 0.044 * surface_area * 1000 # (uS - the default units for netcon.weight, stated here: https://www.neuron.yale.edu/phpBB/viewtopic.php?t=1187)
    #^given by Tiesinga as 0.044 mS/cm^2 0.044 * surface area(cm^2) * 1000

    t_vec = h.Vector().record(h._ref_t)  # hoc object, Time stamp vector
    v_vec = h.Vector().record(soma(0.5)._ref_v)  # hoc object, Membrane potential vector

    # Run the simulation
    h.tstop = runtime
    h.run()

    '''Recording when spikes occur'''
    output_t = [] #list with output spike times
    c = 0  # a counter to make sure we only count each spike once

    for j in range(len(t_vec) - 1): #Subtract 1 to not include the final trial where we decrease the current inj
        if v_vec[j] > 0 and c == 0:
            output_t.append(t_vec[j])
            c = 1
        elif v_vec[j] < 0:
            c = 0
    raster_spikes.append(output_t)

    # plot first spikes
    if i == 0:
        axs[0].plot(np.array(t_vec), np.array(v_vec), color=colors1[i])
    
    if i == 10:
        axs[1].plot(np.array(t_vec), np.array(v_vec), color = colors1[i])

# raster_spikes_flipped = []
# for i in range(len(raster_spikes)):
#     raster_spikes_flipped.append(raster_spikes.pop(-1))
raster_spikes.reverse()
colors1.reverse()
axs[-1].eventplot(raster_spikes, colors=colors1)
# plt.savefig('fig2_abd')

'''Bin Plots (but they're bar graphs right now, we can change later)'''
#bin width 0.01ms
bin_x_values = []
bin_counter = []
time_ctr = 0
dt = 1 #ms
for i in range(int(runtime/dt)):
    bin_count = 0
    bin_x_values.append(time_ctr)
    for trial in raster_spikes:
        for spike in trial:
            if time_ctr <= spike < time_ctr + dt:
                bin_count += 1
    time_ctr += dt
    bin_counter.append(bin_count)

axs[2].bar(bin_x_values, bin_counter, dt)
axs[2].set(ylim=(0, 4))
# plt.savefig('fig2')
plt.show()