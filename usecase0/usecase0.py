# In[import all packages] :
"""
Pints and myokit should be installed before (follow instruction from Github)
"""

import numpy as np
import matplotlib.pyplot as plt
import pints
import pints.plot
import myokit
import pandas as pd
# In[Load Data]
File_path = 'PK_Excercise3_1Dose.csv'
# load data from file
with open(File_path) as f:
    data = pd.read_csv(f, sep=',')
    data.head()
    f.close()

type(data)
data = data.to_numpy()

timeReadIn = data[:, 0]
concReadIn = data[:, 1]
doseReadIn = data[:, 2]

doseLevel_mgPerKg = np.array(np.unique(doseReadIn))
mice_weight = 0.025
doseLevel_ng = doseLevel_mgPerKg*mice_weight*1e6

# reshape the data such that it is the same format as the model output
NumberSamplingPoints = data.shape[0] / len(doseLevel_ng)
timeFinal = timeReadIn[0:int(NumberSamplingPoints)]

concFinal = np.reshape(
    concReadIn, (int(len(concReadIn)/NumberSamplingPoints), int(NumberSamplingPoints))).T


Original_data = {'time': timeFinal,
                 'observation': concFinal}
colorlist = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C0']
plt.figure()
for i in range(len(doseLevel_ng)):
    plt.semilogy(Original_data['time'], Original_data['observation']
                 [:, i], 'o',  label=doseLevel_mgPerKg[i])

plt.legend()
plt.show()

# In[Initialization] :
"""
Call model from myokit

"""

m = myokit.load_model('ThreeCompartment_IV_Model.mmt')  # path for the model file
# path for the protocol file(e.g. dose regimen)
p = myokit.load_protocol('protocol_New.mmt')

ref = myokit.Simulation(m, p)  # set up myokit model: input model and protocol
save_state = ref.state()  # save the original initial state

# Change DoseLevel to dosing per time unit
# Extract duration of the dosing event from the protocol
event_duration = p.events()[0].duration()
DoseLevel = doseLevel_ng / event_duration


# In[Define Model]
"""
set up model compatible for pints, i.e. a class that takes parameters and
times
"""


class MyokitModel(pints.ForwardModel):

    def __init__(self):
        # path for the model file
        m = myokit.load_model('ThreeCompartment_IV_Model.mmt')
        # path for the protocol file(e.g. dose regimen)
        p = myokit.load_protocol('protocol_New.mmt')

        # define simulation (i.e. run the model via myokit)
        self.simulation = myokit.Simulation(m, p)

    def n_parameters(self):
        return 6  # number of parameters to Fit

    def n_outputs(self):
        return len(DoseLevel)

    def simulate(self, parameters, times):
        Drug_Central = []

       # setting up simulation parameters
        self.simulation.set_state(save_state)
        self.simulation.reset()
        self.simulation.set_time(0)
        self.simulation.set_constant(
            'AllCompartment.CL', parameters[0])  # define parameter
        self.simulation.set_constant(
            'AllCompartment.Vc', parameters[1])  # define parameter
        self.simulation.set_constant(
            'AllCompartment.Qp1', parameters[2])  # define parameter
        self.simulation.set_constant(
            'AllCompartment.Vp1', parameters[3])  # define parameter
        self.simulation.set_constant(
            'AllCompartment.Qp2', parameters[4])  # define parameter
        self.simulation.set_constant(
            'AllCompartment.Vp2', parameters[5])  # define parameter

        # simulating multiple dose levels
        var_to_log = 'AllCompartment.Drug_Concentration_Central'

        DoseAmounts = DoseLevel  # Define dose level, can move this line to the initialization part
        for i in range(len(DoseAmounts)):
            self.simulation.reset()
            self.simulation.set_constant('dose.doseAmount', float(DoseAmounts[i]))
            Output = self.simulation.run(times[-1]+1, log=[var_to_log], log_times=times)
            Drug_Central.append(Output[var_to_log])

        return np.array(Drug_Central).T


# Then create an instance of our new model class
ThreeCompModel = MyokitModel()


# In[Optimization]

times = Original_data['time']
values = Original_data['observation']

problem = pints.MultiOutputProblem(ThreeCompModel, times, values)

score = pints.SumOfSquaresError(problem)

# initial guess: use knoweldge from the PK lecture to get a better initial guess
x0 = [3, 6, 1, 25, 10, 6]

boundaries = pints.RectangularBoundaries([0.01, 0.01, 0.01, 0.01, 0.01, 0.01], [
                                         100, 100, 100, 100, 100, 100])

# Perform an optimization
found_parameters, found_value = pints.optimise(
    score, x0, boundaries=boundaries, method=pints.CMAES)
print('Score at true solution:')
# print(score(x0))

print('Found solution:          initial guess:')
for k, x in enumerate(found_parameters):
    print(pints.strfloat(x) + '    ' + pints.strfloat(x0[k]))

# true parameters : [5.2,8.65,1.6,32.2,15,7.9] used for simulating the data


# In[Plot Fitted over Observation]

times_Simulation = np.linspace(0.001, 24, 500)

# run a test simulation
FittedSimulation = ThreeCompModel.simulate(found_parameters, times_Simulation)

# Plot the result
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i, trace in enumerate(FittedSimulation.T):
    plt.plot(times_Simulation, trace, label=DoseLevel[i])
    plt.semilogy(Original_data['time'], Original_data['observation']
                 [:, i], 'o',  label=doseLevel_mgPerKg[i])
ax.set_yscale('log')
plt.legend()
plt.show()

FittedSimulation = ThreeCompModel.simulate(found_parameters, times).T

fig = plt.figure()

for i in range(len(doseLevel_ng)):
    plt.plot(Original_data['observation'][:, i],
             FittedSimulation[i, :], 'o', label=DoseLevel[i])

plt.plot([0.1, np.amax(FittedSimulation)], [0.1, np.amax(FittedSimulation)])
plt.legend()
plt.show()

# In[Estimate Posterior Distribution]

Noise_sigma = 2

found_parameters_noise = np.array(list(found_parameters) + [Noise_sigma])

xs = [
    found_parameters_noise * 1,
    found_parameters_noise * 0.9,
    found_parameters_noise * 1.05,
    found_parameters_noise * 0.95,
    found_parameters_noise * 1.1,
]


log_prior = pints.UniformLogPrior(np.array(
    xs[0]*0.5).tolist(), np.array(xs[0]*2).tolist())  # a uniform distribution for prior
# Create a log-likelihood function
log_likelihood = pints.UnknownNoiseLogLikelihood(problem)
# Create a posterior log-likelihood (log(likelihood * prior))
log_posterior = pints.LogPosterior(log_likelihood, log_prior)


# Define MCMC
mcmc = pints.MCMCSampling(log_posterior, len(
    xs), xs, method=pints.AdaptiveCovarianceMCMC)

# Add stopping criterion
mcmc.set_max_iterations(50000)
initial_point = 3000  # Start adapting after 3000 iterations

mcmc.set_initial_phase_iterations(initial_point)
mcmc.set_chain_filename('./Chain_Excercise4_DenseSampling.csv')
mcmc.set_log_pdf_filename('./LogPDF_Excercise4_DenseSampling.csv')

# Disable verbose mode
# mcmc.set_verbose(False)

# Run!
print('Running...')
chains = mcmc.run()
print('Done!')

names = ['CL', 'Vc', 'Qp1', 'Vp1', 'Qp2', 'Vp2', 'noise sigma']
fig, ax = pints.plot.trace(chains, parameter_names=names)

fig.savefig('trace.png')
