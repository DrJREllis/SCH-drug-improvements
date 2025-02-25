from joblib import Parallel, delayed
import multiprocessing
import copy

from sch_simulation.helsim_FUNC_KK import *

num_cores = multiprocessing.cpu_count()

def loadParameters(paramFileName, demogName, paramsOverride=None, numReps=None):

    '''
    This function loads all the parameters from the input text
    files and organizes them in a dictionary.
    Parameters
    ----------
    paramFileName: str
        name of the input text file with the model parameters;
    demogName: str
        subset of demography parameters to be extracted;
    r0: double
        r0 parameter override
    k: double
        k parameter override
    Returns
    -------
    params: dict
        dictionary containing the parameter names and values;
    '''
    
    #paramslist = [readParams(paramFileName=paramFileName, demogName=demogName) for x in range(numReps)]

    #### load the parameters
    params = readParams(paramFileName=paramFileName, demogName=demogName)
    
    if numReps is not None:
        params['numReps'] = numReps

    if paramsOverride is not None:
        params = params | paramsOverride
        
  
    # params['N'] = int(np.mean(paramsOverride[paramsOverride['TargetPop']=='Total']['PopReq']))

    # overwrite r0
    ##if r0 is not None:
    ##    params['R0'] = r0 
        
    # overwrite k
    ##if k is not None:
    ##    params['k'] = k 
    
 #   for i in range(len(paramslist)):
 #       paramslist[i]['R0'] = r0[i]
 #       paramslist[i]['k'] = k[i]
 #       # configure the parameters
 #       paramslist[i] = configure(paramslist[i])
 #       
 #       # update the parameters
 #       paramslist[i]['psi'] = getPsi(paramslist[i])
 #       paramslist[i]['equiData'] = getEquilibrium(paramslist[i])


    # configure the parameters
    params = configure(params)

    # update the parameters
    params['psi'] = getPsi(params)
    params['equiData'] = getEquilibrium(params)

    return params

def doRealization(params, i):

    '''
    This function generates a single simulation path.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    i: int
        iteration number, provides the seed;
    Returns
    -------
    results: list
        list with simulation results;
    '''
    
    # set seed
    #np.random.seed(i)

    # setup simulation data
    simData = setupSD(params)

    # start time
    t = 0

    # end time
    maxTime = copy.deepcopy(params['maxTime'])

    # time at which to update the freelive population
    freeliveTime = t

    # times at which data should be recorded
    outTimes = copy.deepcopy(params['outTimings'])

    # time when data should be recorded next
    nextOutIndex = np.argmin(outTimes)
    nextOutTime = outTimes[nextOutIndex]

    # time at which individuals' age is advanced next
    ageingInt = 1 / 52
    nextAgeTime = 1 / 52
    maxStep = 1 / 52
    
    # time at which individuals receive next chemotherapy
    currentchemoTiming = copy.deepcopy(params['chemoTimings'])

    nextChemoIndex = np.argmin(currentchemoTiming)

    nextChemoTime = currentchemoTiming[nextChemoIndex]
    lastChemoTime = np.nan
    # next event
    nextStep = np.min([nextOutTime, t + maxStep, nextChemoTime, nextAgeTime])

    results = [] # initialise empty list to store results
    elimCount = 0

    # run stochastic algorithm
    while t < maxTime:

        rates = calcRates(params, simData)
        sumRates = np.sum(rates)

        # if the rate is such that nothing's likely to happen in the next 10,000 years,
        # just fix the next time step to 10,000
        if sumRates < 1e-4:

            dt = 10000

        else:

            dt = np.random.exponential(scale=1 / sumRates, size=1)[0]

        if t + dt < nextStep:

            t += dt

            simData = doEvent(rates, simData, params, t-lastChemoTime)

        else:

            simData = doFreeLive(params, simData, nextStep - freeliveTime)

            t = nextStep
            freeliveTime = nextStep
            timeBarrier = nextStep

            # ageing and death
            if timeBarrier >= nextAgeTime:

                simData = doDeath(params, simData, t)

                nextAgeTime += ageingInt

            # chemotherapy
            if timeBarrier >= nextChemoTime:

                simData = doDeath(params, simData, t)

                lastChemoTime = t

                simData = doChemo(params, simData, t, params['coverage'])

                currentchemoTiming[nextChemoIndex] = maxTime + 10
                nextChemoIndex = np.argmin(currentchemoTiming)
                nextChemoTime = currentchemoTiming[nextChemoIndex]

                if nextChemoIndex == params['numFirstDrug'] + 1:
                    params['DrugEfficacy'] = params['DrugEfficacy2']
                    params['JuvenileDrugEfficacy'] = params['JuvenileDrugEfficacy2']
                        
           

            if timeBarrier >= nextOutTime:

                results.append(dict(
                    iteration=i,
                    time=t,
                    worms=copy.deepcopy(simData['worms']),
                    hosts=copy.deepcopy(simData['demography']),
                    lifetimeWorms=copy.deepcopy(simData['lifetimeWorms']),
                    yearsInfected=copy.deepcopy(simData['yearsInfected']),
                    freeLiving=copy.deepcopy(simData['freeLiving']),
                    # adherenceFactors=copy.deepcopy(simData['adherenceFactors']),
                    # compliers=copy.deepcopy(simData['compliers'])
                ))

                outTimes[nextOutIndex] = maxTime + 10
                nextOutIndex = np.argmin(outTimes)
                nextOutTime = outTimes[nextOutIndex]

            nextStep = np.min([nextOutTime, t + maxStep, nextChemoTime,nextAgeTime])

    results.append(dict(# attendanceRecord=np.array(simData['attendanceRecord']),
                        # ageAtChemo=np.array(simData['ageAtChemo']),
                        # adherenceFactorAtChemo=np.array(simData['adherenceFactorAtChemo'])
                   ))

    return results

def SCH_Simulation(paramFileName, demogName, paramsOverride=None, seed=0, numReps=None):

    '''
    This function generates multiple simulation paths.
    Parameters
    ----------
    paramFileName: str
        name of the input text file with the model parameters;
    demogName: str
        subset of demography parameters to be extracted;
    demogName: int
        seed for random number generation
    r0: double
        r0 parameter override
    k: double
        k parameter override
    numReps: int
        number of simulations;
    Returns
    -------
    df: data frame
        data frame with simulation results;
    '''

    # initialize the parameters
    params = loadParameters(paramFileName, demogName, paramsOverride, numReps)


    # run the simulations
    results = Parallel(n_jobs=num_cores)(delayed(doRealization)(params, i) for i in range(seed*numReps,(seed+1)*numReps))
    # results = [doRealization(params, i) for i in range(seed*numReps,(seed+1)*numReps)]

    # process the output
    output = extractHostData(results)

    # transform the output to data frame
    [df, all_results] = getPrevalence(output, params, numReps, villageSampleSize=params['N'])
   # [M,P,wormBurdens] = getDistribution(output)

    return df, all_results, params
