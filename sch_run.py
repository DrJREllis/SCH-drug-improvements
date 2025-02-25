
from sch_simulation.helsim_RUN_KK import SCH_Simulation
import json
import numpy as np

numReps=200
R0=4
k=0.24
paramFileName='Mansoni-high_adult_burden.txt'
outputFileName = 'mansoni-highAdultBurden-70prev-'

filename = outputFileName+'baseline'
paramsOverride=dict(JuvenileDrugEfficacy=0, R0=R0, k=k, coverage=np.array([0,0.75,0.75,0.75]))

[df, all_results,params] = SCH_Simulation(paramFileName=paramFileName, demogName='KenyaKDHS',paramsOverride=paramsOverride, numReps=numReps)

df.to_json('Data_output/' + filename + '_df_results.json')

np.save('Data_output/' + filename + '_egg_results',np.array(all_results['egg_results']))
np.save('Data_output/' + filename + '_worm_results',np.array(all_results['worm_results']))
np.save('Data_output/' + filename + '_params',np.array(params))

del df, all_results

juvEffArr = [0.863, 0.6, 0.4, 0.2]

for JuvenileDrugEfficacy in juvEffArr:
    filename = outputFileName+'juvenileDrugEfficacy_'+str(int(JuvenileDrugEfficacy*100))
    paramsOverride=dict(JuvenileDrugEfficacy=JuvenileDrugEfficacy, R0=R0, k=k, coverage=np.array([0,0.75,0.75,0.75]))

    [df, all_results,params] = SCH_Simulation(paramFileName=paramFileName, demogName='KenyaKDHS',paramsOverride=paramsOverride, numReps=numReps)

    df.to_json('Data_output/' + filename + '_df_results.json')

    np.save('Data_output/' + filename + '_egg_results',np.array(all_results['egg_results']))
    np.save('Data_output/' + filename + '_worm_results',np.array(all_results['worm_results']))
    np.save('Data_output/' + filename + '_params',np.array(params))

    del df, all_results


longLastArr = [3, 5, 10, 15]

for longLastWeeks in longLastArr:
    filename = outputFileName+'longLastingEfficacy_'+str(int(longLastWeeks))

    paramsOverride=dict(longLastEfficacy=0.863, longLastTimescale=longLastWeeks/52, R0=R0, k=k, coverage=np.array([0,0.75,0.75,0.75]), )

    [df, all_results,params] = SCH_Simulation(paramFileName=paramFileName, demogName='KenyaKDHS',paramsOverride=paramsOverride, numReps=numReps)

    df.to_json('Data_output/' + filename + '_df_results.json')

    np.save('Data_output/' + filename + '_egg_results',np.array(all_results['egg_results']))
    np.save('Data_output/' + filename + '_worm_results',np.array(all_results['worm_results']))
    np.save('Data_output/' + filename + '_params',np.array(params))

    del df, all_results


