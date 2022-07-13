# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:15:08 2019

@author: wangk39
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pints
import math
import myokit
import pandas as pd



# In[Initialization] :
"""
Call model from myokit 

"""

m = myokit.load_model('./AdpativeModelWithThreeCompartmentLinearPK.mmt') #path for the model file 
p = myokit.load_protocol('./protocol_QID.mmt')#path for the protocol file(e.g. dose regimen)

ref = myokit.Simulation(m, p) #set up myokit model: input model and protocol 
save_state = ref.state()  #save the original initial state 

doseLevel_mgPerKg = np.array([0,25,50,100,150,200,250,375])#in mg/kg i.e. 25 mg/kg for each dose and QID will be giving 25 mg/kg every 6 hrs 


InitialCFU =[1102000,1102000,1102000,1102000,1102000,1102000,1102000,1102000,1102000,1102000] # initial value of the CFU (one of the state variables)


mice_weight = 0.026
doseLevel_ng = doseLevel_mgPerKg*mice_weight*1e6 #calculate the flat dose in ng unit 
event_duration = p.events()[0].duration()
DoseLevel = doseLevel_ng / event_duration

PK_parameters = [6,30,24,1,10,0.03,0.7] # true paramter for synthesized data 

PD_parameters = [0.5,1.35E+09,1.5,0.1,4,2,0.15,0.5,1]#true paramter for synthesized data



#%% # Define solving 

class MyokitModel(pints.ForwardModel):
    def __init__(self):
        m = myokit.load_model('./AdpativeModelWithThreeCompartmentLinearPK.mmt') #path for the model file 
        p = myokit.load_protocol('./protocol_QID.mmt')
        
        
        self.simulation = myokit.Simulation(m, p) #define simulation (i.e. run the model via myokit)
        
    def n_parameters(self):
        return 9 # number of parameters to Fit
    
    def n_outputs(self):
        return len(DoseLevel) 
    
    def simulate(self, PK_parameters,PD_parameters,times):
        total_CFU = []
        Drug_Central = []
        
        self.simulation.set_state(save_state)
        self.simulation.reset() 
        P1 = m.get('PDCompartment.P1')
        self.simulation.set_time(0)
        #self.simulation.set_constant('doseCompart.Ka', PK_parameters[0]) # define parameter 
        #self.simulation.set_constant('PKCompartment.fu', 0.014) # define parameter
        #self.simulation.set_constant('PKCompartment.CL', PK_parameters[1]) # define parameter 
        
        #self.simulation.set_constant('PKCompartment.Vc', PK_parameters[2]) # define parameter 
        self.simulation.set_constant('doseCompart.Ka', PK_parameters[0]) # define parameter 
        self.simulation.set_constant('PKCompartment.CL', PK_parameters[1]) # define parameter 
        self.simulation.set_constant('PKCompartment.Vc', PK_parameters[2]) # define parameter 
        self.simulation.set_constant('PKCompartment.Qp1',PK_parameters[3]) # define parameter         
        self.simulation.set_constant('PKCompartment.Vp1', PK_parameters[4]) # define parameter 
        self.simulation.set_constant('PKCompartment.Qp2', PK_parameters[5]) # define parameter 
        self.simulation.set_constant('PKCompartment.Vp2', PK_parameters[6]) # define parameter
        self.simulation.set_constant('PKCompartment.fu', 0.014) # define parameter, this parameter is the free fraction of a drug, it is a measured parameter, not fitted from the data 

        
        self.simulation.set_constant('PDCompartment.KNetgrowth', PD_parameters[0]) 
        self.simulation.set_constant('PDCompartment.tvbmax', PD_parameters[1]) 
        self.simulation.set_constant('PDCompartment.Kmax',PD_parameters[2]) 
        self.simulation.set_constant('PDCompartment.EC50k', PD_parameters[3])        
        self.simulation.set_constant('PDCompartment.gamma', PD_parameters[4]) 
        self.simulation.set_constant('PDCompartment.beta',PD_parameters[5])
        self.simulation.set_constant('PDCompartment.tau',PD_parameters[6])         
        self.simulation.set_constant('PDCompartment.Kdeath',PD_parameters[7])  
        self.simulation.set_constant('PDCompartment.Ksr_max', PD_parameters[8]) 
               
          
        # simulating multiple dose levels
        PD_var_to_log = 'PDCompartment.Total_Bacterial'
        PK_var_to_log = 'PKCompartment.Drug_Concentration_Central'
        
        DoseAmounts=DoseLevel # Define dose level, can move this line to the initialization part 
        for i in range(len(DoseAmounts)):
            self.simulation.reset() 
            P1.set_state_value(InitialCFU[i])
            updated_state = m.state()
            self.simulation.set_state(updated_state)
            self.simulation.set_constant('dose.doseAmount', float(DoseAmounts[i]))
            Output = self.simulation.run(times[-1]+1, log_times = times)
            
            total_CFU.append(Output[PD_var_to_log]) 
            Drug_Central.append(Output[PK_var_to_log])       
                
        return  np.array(Drug_Central).T, np.array(total_CFU).T
        
AdaptiveModel = MyokitModel()


colorlist=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C0', 'C1', 'C2']


#%% # run a test simulation
times_step=0.5
times_Simulation = np.arange(0, 24,times_step)

Plasma_PK, CFU=AdaptiveModel.simulate(PK_parameters,PD_parameters, times_Simulation)


# Plot the result
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i, trace in enumerate(Plasma_PK.T):
    plt.plot(times_Simulation, trace, color=colorlist[i],label='Simulation--Dose:' + str(doseLevel_mgPerKg[i])+ ' mg/kg')    
plt.show()
    
#ax.set_yscale('log')
plt.legend()
plt.xlabel('time (hr)')
plt.ylabel('Free Plasma concentration (ng/mL)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)  
LogCFU_24hr=np.empty ((len(doseLevel_mgPerKg),0)) 
LogCFU_reduction=[]
for j, trace in enumerate(CFU.T):
    plt.plot(times_Simulation, np.log10(trace),color=colorlist[j], label='Simulation--Dose:' + str(doseLevel_mgPerKg[j])+ ' mg/kg')
    LogCFU_24hr=np.append(LogCFU_24hr,np.log10(trace)[-1])
    LogCFU_reduction.append(np.log10(trace)[-1]-np.log10(trace)[0])

plt.legend()    
    
#ax.set_yscale('log')
plt.legend()
plt.xlabel('time (hr)')
plt.ylabel('CFU')
plt.ylim (3,10)
plt.show()



    
#