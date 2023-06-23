import neuron
from neuron import h
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

'''Spike Volley Parameters'''
P = 26.10 #ms mean time (ms) between spike volleys, from appendix fig 2
CV_t = 0.095 #Coefficient of variance calculated as sqrt(var/mean), from appendix fig 2.
sigma_IV = 3 #ms standard deviation of gaussian distribution for fig 1b
# stdev = CV_t * mean
num_spike_volleys = 8 #the number of spike volleys generated

def gaussian(x, mu, sigma):
    '''
    Purpose: Generate a Gaussian curve
    Parameters:
        x (array): inputed values
        mu (float): mean of the Gaussian curve
        sigma (float): standard deviation
    Output (float): the probability density for the Gaussian distribution
    '''
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
    Output (list): [flat_x, volley_trace, spike_probs, spike_trace, spike_times]

    '''
    spike_volleys = np.zeros(num_spike_volleys) #Generates an empty array that's the size of the number of spike volleys
    for i in range(1, len(spike_volleys)): #Fills the spike_volleys array with spike times - what you see in fig 1a
        spike_volleys[i] = spike_volleys[i - 1] + np.random.normal(P, CV_t * P)
    # Define and apply Gaussian filter w/ length of 40ms, centered around spike volleys
    STP_x = [] #spike time probability
    STP_gaus = []
    spike_probs = []

    for v_time in range(1, len(spike_volleys) - 1):
        gaus_start = (spike_volleys[v_time] + spike_volleys[v_time - 1]) / 2 #avg time of two spike volleys, on the left side
        gaus_end = (spike_volleys[v_time] + spike_volleys[v_time + 1]) / 2 #avg time of two spike volleys, on the right side
        x_length = int(gaus_end - gaus_start + 1) 
        x = np.linspace(gaus_start, gaus_end, x_length) #calculates x_length evenly-spaced samples between gaus_start and gaus_end
        
        if (v_time < num_spike_volleys//2): #to make the first half of fig1b wide, and the second half narrow
             gaus = gaussian(x, spike_volleys[v_time], sigma_IV * 2.5) #generating Gaussian curves (as defined above)
        else:
            gaus = gaussian(x, spike_volleys[v_time], sigma_IV)

        STP_x.append(x)
        STP_gaus.append(gaus)


    # Flatten into 1D arrays for plotting
    flat_x = [x for volley in STP_x for x in volley]
    flat_gaus = [gaus for volley in STP_gaus for gaus in volley]

    spike_probs.append(np.array(flat_gaus) / np.max(flat_gaus))

    spike_probs = [prob for volley in spike_probs for prob in volley]

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
    # Plot spike times
    return [flat_x, volley_trace, spike_probs, spike_trace, spike_times]

'''Making Graphs Using MatPlot'''
results = generate_spike_times()
# fig, axs = plt.subplots(nrows = 4, ncols = 1, sharex = True)
# axs[0].stem(results[0], results[1], markerfmt = ' ')
# axs[1].plot(results[0], results[2])
# axs[2].stem(results[0], results[3], markerfmt = ' ')

h.load_file('nrngui.hoc')
h.load_file('stdrun.hoc')


'''Building Soma'''
h.dt = 0.01  # ms Timestep as stated in page 4 of the Tiesinga paper (same as their bin width)
h.v_init = -65  # mV Initial membrane potential

soma = h.Section(name='soma')

# ** = from calculations based on Fellous et al. 2010
soma.L = 79.788  # soma length µm  **
soma.diam = 79.788  # soma diameter µm  **
soma.cm = 1  # membrane capacitance µF/cm2  
soma.Ra = 100  # ohm-cm  **

soma.insert('leak')  # Leak channel, using the .mod file 
soma.glbar_leak = 0.0001  # S/cm2 
soma.el_leak = -65  # mV

soma.insert('na')  # Sodium channel, using the .mod file
soma.gnabar_na = 0.035  # S/cm2
soma.ena = 55 # mV

soma.insert('k')  # Potassium channel, using the .mod file
soma.gkbar_k = 0.009  # S/cm2
soma.ek = -90 # mV

'''Current Clamp (Background)'''
# Can delete below, there's no excitatory input
# Documentation for a synapse:
# https://www.neuron.yale.edu/neuron/static/py_doc/modelspec/programmatic/mechanisms/mech.html#ExpSyn
# ^ the h.Expsyn class needs its own mod file. We took ours from this link, no edits


# current clamp -- currently injected in middle of soma
current = 4.0 #μA/cm2
surface_area = (np.pi * soma.diam**2 + 2 * np.pi * (soma.diam/2)**2) / 10**8 #in cm^2
current_inj = h.IClamp(soma(0.5))
current_inj.delay = 0 #ms
current_inj.dur = results[0][-1] + 50 #ms, 50ms after the last spike occurs
current_inj.amp = current * 1000 * surface_area #nA current, calculated from fig 2 in appendix A.4

# # Create a NetStim object
# stim = h.NetStim()

#Putting the spike train in through a VecStim object
stimulus = h.ExpSyn(soma(0.5)) #inhibitory synapse into the middle of the cell
stimulus.tau =  10 # ms decay time constant, as per fig.s  2-10 in the appendix of Tiesinga
stimulus.e = -75 #mV reversal potential, from pg. 3 of Tiesinga, this is the Erev for a GABA synapse
vec = h.Vector(results[4])
print(vec)
vecstim=h.VecStim() # need vecevent.mod file
vecstim.play(vec)
netcon = h.NetCon(vecstim, soma) 

netcon.delay = min(results[0])
netcon.weight[0] = 0.5

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
t_vec = h.Vector().record(h._ref_t)  #hoc object, Time stamp vector
v_vec = h.Vector().record(soma(0.5)._ref_v)  #hoc object, Membrane potential vector

# Run the simulation
# h.tstop = len(results[0]) + 50 # ms
h.tstop = results[0][-1] + 50 #ms
h.run()

#plot membrane conductance:
#Δginh = 0.044 mS/cm2 – each pulse had a "unitary peak conductance"
#0.44 mS/cm2 – the time-average of the inhibitory conductance (and the value for appendix fig )
#"The neuron was not spontaneously active in the absence of synaptic inputs; hence, in order to make it spike in the presence of inhibition, a constant depolarizing current I = 4.0 μA/cm2 was also injected."
#"Each input spike produced an exponentially decaying conductance pulse, Δginhexp(−t/τinh) in the postsynaptic cell"


# Plot membrane potential
fig, ax = plt.subplots()
ax.plot(t_vec, v_vec)

# plot membrane potential over time
#axs[3].plot(t_vec, v_vec)
#axs[3].set_ylim([-75, -73])
#axs[3].set_xlabel('Time (ms)')
#axs[3].set_ylabel('Membrane Potential (mV)')


plt.savefig("Spike Times Generation")
plt.show()


'''
SD (width) of Gaussian dist abt the mean of a volley: sigma_iv
area of the Gaussian dist abt the mean (sum of the bins):  was a_IV*Δt.
bin width (Δt) = 0.01
The Gaussian filter was 40 ms long in order to accommodate at least 2 standard deviations for the maximum sigma_IV used in the simulations (peak at 20ms)
'''    
