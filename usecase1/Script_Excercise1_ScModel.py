# In[import all packages] :
"""
Pints and myokit should be installed before (follow instruction from Github)
"""

import numpy as np
import matplotlib.pyplot as plt
import pints
import myokit


# In[Initialization] :
"""
Call model from myokit 

"""

m = myokit.load_model('ThreeCompartment_OralSCModel.mmt') #path for the model file 
p = myokit.load_protocol('protocol_New.mmt') #path for the protocol file(e.g. dose regimen)

ref = myokit.Simulation(m, p) #set up myokit model: input model and protocol 
save_state = ref.state()  #save the original initial state 

# Total  amount of drug dosed: 1mg/kg, 5mg/kg and 15mg/kg dose for mice (unit ng)
TotalDoseLevel= np.array([25000, 125000, 375000])

# Change DoseLevel to dosing per time unit
event_duration = p.events()[0].duration() # Extract duration of the dosing event from the protocol
DoseLevel = TotalDoseLevel / event_duration



# In[Define Model]
"""
set up model compatible for pints, i.e. a class that takes parameters and
times        
"""
class MyokitModel(pints.ForwardModel):
    
    def __init__(self):
        m = myokit.load_model('ThreeCompartment_OralSCModel.mmt') #path for the model file 
        p = myokit.load_protocol('protocol_New.mmt')#path for the protocol file(e.g. dose regimen)
      
        self.simulation = myokit.Simulation(m, p) #define simulation (i.e. run the model via myokit)

    def n_parameters(self):
        return 7 # number of parameters to Fit

    def simulate(self, parameters, times):
        Drug_Central = []
        
       #setting up simulation parameters 
        self.simulation.set_state(save_state)
        self.simulation.reset()     
        self.simulation.set_time(0)
        self.simulation.set_constant('doseCompart.Ka', parameters[0]) # define parameter 
        self.simulation.set_constant('AllCompartment.CL', parameters[1]) # define parameter 
        self.simulation.set_constant('AllCompartment.Vc', parameters[2]) # define parameter 
        self.simulation.set_constant('AllCompartment.Qp1', parameters[3]) # define parameter 
        self.simulation.set_constant('AllCompartment.Vp1', parameters[4]) # define parameter 
        self.simulation.set_constant('AllCompartment.Qp2', parameters[5]) # define parameter 
        self.simulation.set_constant('AllCompartment.Vp2', parameters[6]) # define parameter
                                 
        # simulating multiple dose levels
        var_to_log = 'AllCompartment.Drug_Concentration_Central'
        
        DoseAmounts=DoseLevel # Define dose level, can move this line to the initialization part 
        for i in range(len(DoseAmounts)):
            self.simulation.reset()  
            self.simulation.set_constant('dose.doseAmount', float(DoseAmounts[i]))
            Output = self.simulation.run(times[-1]+1, log=[var_to_log], log_times = times)
            Drug_Central.append(Output[var_to_log])
        
        return np.array(Drug_Central).T

                    


# Then create an instance of our new model class
ThreeCompModel = MyokitModel()

# In[Run a test simulation]
#param_test=[2,3,5,8,50,2,10] 
param_test=[2, 5.2,8.65,1.6,32.2,15,7.9]#[3,5,8,50,2,10] [5.2,8.65,1.6,32.2,15,7.9]
#times=[0.083,0.25,0.5,1,2,3,4,6,7,8,10,12,15,18,24]
times=[0.083,0.25,0.5,1,2,3,4,5,6,7,10,11,12,15,16,18,20,24]

#times= np.linspace(0,24,500)

# run a test simulation
TestSimulation=ThreeCompModel.simulate(param_test ,times)

# Plot the result
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i, trace in enumerate(TestSimulation.T):
    plt.plot(times, trace, 'o', label=TotalDoseLevel[i])
ax.set_yscale('log')
plt.legend()
plt.show()


