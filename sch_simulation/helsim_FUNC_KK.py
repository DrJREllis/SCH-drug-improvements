import numpy as np
import pandas as pd
from scipy.optimize import bisect
import warnings
import pkg_resources
warnings.filterwarnings('ignore')
np.seterr(divide='ignore')

import sch_simulation.ParallelFuncs as ParallelFuncs

def readParam(fileName):

    '''
    This function extracts the parameter values stored
    in the input text files into a dictionary.
    Parameters
    ----------
    fileName: str
        name of the input text file;
    Returns
    -------
    params: dict
        dictionary containing the parameter names and values;
    '''

    DATA_PATH = pkg_resources.resource_filename('sch_simulation', 'data/')

    with open(DATA_PATH + fileName) as f:
        
        contents = f.readlines()

    params = []

    for content in contents:

        line = content.split('\t')

        if len(line) >= 2:

            try:
                
                line[1] = np.array([float(x) for x in line[1].split(' ')])

                if len(line[1]) == 1:
                    
                    line[1] = line[1][0]

            except:
                
                pass

            params.append(line[:2])

    params = dict(params)

    return params

def readParams(paramFileName, demogFileName='Demographies.txt', demogName='Default'):

    '''
    This function organizes the model parameters and
    the demography parameters into a unique dictionary.
    Parameters
    ----------
    paramFileName: str
        name of the input text file with the model parameters;
    demogFileName: str
        name of the input text file with the demography parameters;
    demogName: str
        subset of demography parameters to be extracted;
    Returns
    -------
    params: dict
        dictionary containing the parameter names and values;
    '''

    demographies = readParam(demogFileName)
    parameters = readParam(paramFileName)

    chemoTimings = np.array([parameters['treatStart'] + parameters['burnIn'] + x * parameters['treatInterval']
    for x in range(int(parameters['nRounds']))])


    params = {'numReps': int(parameters['repNum']),
              'maxTime': parameters['nYears']+parameters['burnIn'],
              'N': int(parameters['nHosts']),
              'R0': float(parameters['R0']),
              'lambda': float(parameters['lambda']),
              'gamma': float(parameters['gamma']),
              'k': float(parameters['k']),
              'sigma': float(parameters['sigma']),
              'LDecayRate': float(parameters['ReservoirDecayRate']),
              'DrugEfficacy': float(parameters['drugEff']),
              'JuvenileDrugEfficacy': float(parameters['drugEffJuv']),
              'contactAgeBreaks': parameters['contactAgeBreaks'],
              'contactRates': parameters['betaValues'],
              'rho': parameters['rhoValues'],
              'treatmentAgeBreaks': parameters['treatmentBreaks'],
              'coverage': parameters['coverage'],
              'treatInterval': float(parameters['treatInterval']),
              'treatStart': parameters['treatStart']+parameters['burnIn'],
              'nRounds': int(parameters['nRounds']),
              'chemoTimings': chemoTimings,
              'outTimings': np.linspace(parameters['burnIn'],parameters['nYears']+parameters['burnIn'],int(1+parameters['nYears']/parameters['outputInterval'])),
              'burdenThreshold': parameters['burdenThreshold'],
              'highBurdenBreaks': parameters['highBurdenBreaks'],
              'highBurdenValues': parameters['highBurdenValues'],
              'demogType': demogName,
              'hostMuData': demographies[demogName + '_hostMuData'],
              'muBreaks': np.append(0, demographies[demogName + '_upperBoundData']),
              'SR': [True if parameters['StochSR'] == 'TRUE' else False][0],
              'reproFuncName': parameters['reproFuncName'],
              'z': np.exp(- parameters['gamma']),
              'psi': 1.0,
              'k_epg': parameters['k_epg'],
              'adherenceSetting': parameters['adherenceSetting'],
              'propNeverCompliers': parameters['propNeverCompliers'],
              'rhoAdherence': parameters['rhoAdherence'],
              'drugResistance': parameters['drugResistance'],
              'longLastEfficacy': parameters['longLastEfficacy'],
              'longLastTimescale': parameters['longLastTimescale'],
              'juvenilePeriod': 0.1,
              'juvenileSurvival': 0.4,
              'numFirstDrug': parameters['numFirstDrug'],
              'DrugEfficacy2': float(parameters['drugEff2']),
              'JuvenileDrugEfficacy2': float(parameters['drugEffJuv2'])
              }

    return params

def configure(params):

    '''
    This function defines a number of additional parameters.
    Parameters
    ----------
    params: dict
        dictionary containing the initial parameter names and values;
    Returns
    -------
    params: dict
        dictionary containing the updated parameter names and values;
    '''

    # level of discretization for the drawing of lifespans
    dT = 0.1

    # definition of the reproduction function
    params['reproFunc'] = getattr(ParallelFuncs, params['reproFuncName'])

    # max age cutoff point
    params['maxHostAge'] = np.min([np.max(params['muBreaks']), np.max(params['contactAgeBreaks'])])

    # full range of ages
    params['muAges'] = np.arange(start=0, stop=np.max(params['muBreaks']), step=dT) + 0.5 * dT

    params['hostMu'] = params['hostMuData'][pd.cut(x=params['muAges'], bins=params['muBreaks'],
    labels=np.arange(start=0, stop=len(params['hostMuData']))).to_numpy()]

    # probability of surviving
    params['hostSurvivalCurve'] = np.exp(-np.cumsum(params['hostMu']) * dT)

    # the index for the last age group before the cutoff in this discretization
    maxAgeIndex = np.argmax([params['muAges'] > params['maxHostAge']]) - 1

    # cumulative probability of dying
    params['hostAgeCumulDistr'] = np.append(np.cumsum(dT * params['hostMu'] * np.append(1,
    params['hostSurvivalCurve'][:-1]))[:maxAgeIndex], 1)

    params['contactAgeGroupBreaks'] = np.append(params['contactAgeBreaks'][:-1], params['maxHostAge'])
    params['treatmentAgeGroupBreaks'] = np.append(params['treatmentAgeBreaks'][:-1], params['maxHostAge'] + dT)

    if params['outTimings'][-1] != params['maxTime']:
        params['outTimings'] = np.append(params['outTimings'], params['maxTime'])

    if params['reproFuncName'] == 'epgMonog':
        params['monogParams'] = ParallelFuncs.monogFertilityConfig(params)

    return params

def setupSD(params):

    '''
    This function sets up the simulation to initial conditions
    based on analytical equilibria.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    Returns
    -------
    SD: dict
        dictionary containing the equilibrium parameter settings;
    '''

    si = np.random.gamma(size=params['N'], scale=1 / params['k'], shape=params['k'])

    lifeSpans = getLifeSpans(params['N'], params)
    trialBirthDates = - lifeSpans * np.random.uniform(low=0, high=1, size=params['N'])
    trialDeathDates = trialBirthDates + lifeSpans

    communityBurnIn = 1000

    while np.min(trialDeathDates) < communityBurnIn:

        earlyDeath = np.where(trialDeathDates < communityBurnIn)[0]
        trialBirthDates[earlyDeath] = trialDeathDates[earlyDeath]
        trialDeathDates[earlyDeath] += getLifeSpans(len(earlyDeath), params)

    demography = {'birthDate': trialBirthDates - communityBurnIn, 'deathDate': trialDeathDates - communityBurnIn}

    contactAgeGroupIndices = pd.cut(x=-demography['birthDate'], bins=params['contactAgeGroupBreaks'],
    labels=np.arange(start=0, stop=len(params['contactAgeGroupBreaks']) - 1)).to_numpy()

    treatmentAgeGroupIndices = pd.cut(x=-demography['birthDate'], bins=params['treatmentAgeGroupBreaks'],
    labels=np.arange(start=0, stop=len(params['treatmentAgeGroupBreaks']) - 1)).to_numpy()

    compliers = np.random.uniform(low=0, high=1, size=params['N']) > params['propNeverCompliers']
    if params['adherenceSetting'] == "betabinomial": 
        ageCoverage = np.where(params['coverage'][treatmentAgeGroupIndices]==0, np.nan, params['coverage'][treatmentAgeGroupIndices]/(1 - params['propNeverCompliers']))
        adherenceFactors = np.random.beta(a=ageCoverage* (1 - params['rhoAdherence'])/params['rhoAdherence'],
                                        b=(1 - ageCoverage)*(1 - params['rhoAdherence'])/params['rhoAdherence'], size=params['N']) 
        adherenceFactors[~compliers] = 0
    else: adherenceFactors = np.random.uniform(low=0, high=1, size=params['N'])

    meanBurdenIndex = pd.cut(x=-demography['birthDate'], bins=np.append(0, params['equiData']['ageValues']),
    labels=np.arange(start=0, stop=len(params['equiData']['ageValues']))).to_numpy()

    wTotal = np.random.poisson(lam=si * params['equiData']['stableProfile'][meanBurdenIndex] * 2, size=params['N'])

    wJuvenile =  np.random.poisson(lam=si * params['equiData']['stableProfile'][meanBurdenIndex] * 2 * params['juvenilePeriod'] * params['sigma'], size=params['N'])

    worms = dict(total=wTotal, female=np.random.binomial(n=wTotal, p=0.5, size=params['N']), juvenile=wJuvenile )

    lifetimeWorms = [wTotal[i] * np.trapz(params['equiData']['stableProfile'][0:meanBurdenIndex[i]], params['equiData']['ageValues'][0:meanBurdenIndex[i]]) 
                              /params['equiData']['stableProfile'][meanBurdenIndex[i]] for i in range(len(meanBurdenIndex))]  
    
    yearsInfected = np.zeros(len(wTotal))

    stableFreeLiving = params['equiData']['L_stable'] * 2


    SD = {'si': si,
          'worms': worms,
          'freeLiving': stableFreeLiving,
          'demography': demography,
          'contactAgeGroupIndices': contactAgeGroupIndices,
          'treatmentAgeGroupIndices': treatmentAgeGroupIndices,
          'adherenceFactors': adherenceFactors,
          'compliers': compliers,
          'attendanceRecord': [],
          'ageAtChemo': [],
          'adherenceFactorAtChemo': [],
          'lifetimeWorms': lifetimeWorms,
          'yearsInfected': yearsInfected}

    return SD

def calcRates(params, SD):

    '''
    This function calculates the event rates; the events are
    new worms and worms death.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    SD: dict
        dictionary containing the equilibrium parameter values;
    Returns
    -------
    array of event rates;
    '''

    hostInfRates = SD['freeLiving'] * SD['si'] * params['contactRates'][SD['contactAgeGroupIndices']] 
    maturationRate = np.sum(SD['worms']['juvenile']/params['juvenilePeriod'])
    juvenileDeathRate = ((1-params['juvenileSurvival'])/params['juvenileSurvival'])*(1/params['juvenilePeriod']) * np.sum(SD['worms']['juvenile'])
    deathRate = params['sigma'] * np.sum(SD['worms']['total'])

    return np.append(hostInfRates, (maturationRate, juvenileDeathRate, deathRate))

def doEvent(rates, SD, params, timeSinceChemo):

    '''
    This function enacts the event; the events are
    new worms and worms death.
    Parameters
    ----------
    rates: float
        array of event rates;
    SD: dict
        dictionary containing the initial equilibrium parameter values;
    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''

    # determine which event takes place; if it's 1 to N, it's a new worm, otherwise it's a worm death
    event = np.argmax(np.random.uniform(low=0, high=1, size=1) * np.sum(rates) < np.cumsum(rates))

    if event == len(rates) - 1: # worm death event

        deathIndex = np.argmax(np.random.uniform(low=0, high=1, size=1) * np.sum(SD['worms']['total']) < np.cumsum(SD['worms']['total']))

        SD['worms']['total'][deathIndex] -= 1

        if np.random.uniform(low=0, high=1, size=1) < SD['worms']['female'][deathIndex] / SD['worms']['total'][deathIndex]:
            SD['worms']['female'][deathIndex] -= 1

    elif event == len(rates) - 2: # juvenile death event
    
        deathIndex = np.argmax(np.random.uniform(low=0, high=1, size=1) * np.sum(SD['worms']['juvenile']) < np.cumsum(SD['worms']['juvenile']))

        SD['worms']['juvenile'][deathIndex] -= 1


    elif event == len(rates) - 3: # worm maturation event
    
        maturationIndex = np.argmax(np.random.uniform(low=0, high=1, size=1) * np.sum(SD['worms']['juvenile']) < np.cumsum(SD['worms']['juvenile']))
        recentChemo = False
        if timeSinceChemo < params['longLastTimescale'] and params['JuvenileDrugEfficacy']==0:
            recentChemo= SD['attendanceRecord'][-1][maturationIndex] * (np.random.uniform(low=0, high=1, size=1)<params['longLastEfficacy'])
        # if timeSinceChemo < 1.5*params['longLastTimescale']:
        #     newEfficacy = params['longLastEfficacy'] * np.exp(-(0.005/params['longLastTimescale'])* np.exp((5/params['longLastTimescale']*timeSinceChemo))*timeSinceChemo)
        #     recentChemo= SD['attendanceRecord'][-1][maturationIndex] * (np.random.uniform(low=0, high=1, size=1)<newEfficacy)

        if ~recentChemo:
            SD['worms']['total'][maturationIndex] += 1

            if np.random.uniform(low=0, high=1, size=1) < 0.5:
                SD['worms']['female'][maturationIndex] += 1

        SD['worms']['juvenile'][maturationIndex] -= 1



    else: # new worm event

        SD['worms']['juvenile'][event] += 1

        #SD['lifetimeWorms'][event] +=1
        if params['longLastEfficacy']*params['JuvenileDrugEfficacy']>0:
            recentChemo = False
            if timeSinceChemo < params['longLastTimescale']:
                recentChemo= SD['attendanceRecord'][-1][event] * (np.random.uniform(low=0, high=1, size=1)<params['longLastEfficacy'])

            if recentChemo:
                SD['worms']['juvenile'][event] -= 1



    return SD

def doFreeLive(params, SD, dt):

    '''
    This function updates the freeliving population deterministically.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    SD: dict
        dictionary containing the initial equilibrium parameter values;
    dt: float
        time interval;
    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''
    # polygamous reproduction; female worms produce fertilised eggs only if there's at least one male worm around
    if params['reproFuncName'] == 'epgFertility' and params['SR']:
        productivefemaleworms = np.where(SD['worms']['total'] == SD['worms']['female'], 0, SD['worms']['female'])

    elif params['reproFuncName'] == 'epgFertility' and not params['SR']:
        productivefemaleworms = SD['worms']['female']

    # monogamous reproduction; only pairs of worms produce eggs
    elif params['reproFuncName'] == 'epgMonog':
        productivefemaleworms = np.minimum(SD['worms']['total'] - SD['worms']['female'], SD['worms']['female'])

    eggOutputPerHost = params['lambda'] * productivefemaleworms * np.exp(-SD['worms']['total'] * params['gamma'])
    eggsProdRate = 2 * params['psi'] * np.sum(eggOutputPerHost * params['rho'][SD['contactAgeGroupIndices']]) / params['N']
    expFactor = np.exp(-params['LDecayRate'] * dt)
    SD['freeLiving'] = SD['freeLiving'] * expFactor + eggsProdRate * (1 - expFactor) / params['LDecayRate']

    SD['lifetimeWorms'] = SD['lifetimeWorms'] + SD['worms']['total'] * dt

    SD['yearsInfected'] = SD['yearsInfected'] + (SD['worms']['total']>0) * dt


    return SD

def doDeath(params, SD, t):

    '''
    Death and aging function.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    SD: dict
        dictionary containing the initial equilibrium parameter values;
    t: int
        time step;
    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''

    # identify the indices of the dead
    theDead = np.where(SD['demography']['deathDate'] < t)[0]

    if len(theDead) != 0:


        # update the birth dates and death dates
        SD['demography']['birthDate'][theDead] = t - 0.001
        SD['demography']['deathDate'][theDead] = t + getLifeSpans(len(theDead), params)

        # they also need new force of infections (FOIs)
        SD['si'][theDead] = np.random.gamma(size=len(theDead), scale=1 / params['k'], shape=params['k'])

        # kill all their worms
        SD['worms']['total'][theDead] = 0
        SD['worms']['female'][theDead] = 0
        SD['worms']['juvenile'][theDead] = 0

        # refresh lifetime worm burden
        SD['lifetimeWorms'][theDead] = 0
        SD['yearsInfected'][theDead] = 0

        # update the adherence factors
        SD['adherenceFactors'][theDead] = np.random.uniform(low=0, high=1, size=len(theDead))


        # assign the newly-born to either comply or not
        SD['compliers'][theDead] = np.random.uniform(low=0, high=1, size=len(theDead)) > params['propNeverCompliers']


    # update the contact age categories
    SD['contactAgeGroupIndices'] = pd.cut(x=t - SD['demography']['birthDate'], bins=params['contactAgeGroupBreaks'],
    labels=np.arange(0, len(params['contactAgeGroupBreaks']) - 1)).to_numpy()

    # update the treatment age categories
    if params['adherenceSetting']=="systematic":
        SD['treatmentAgeGroupIndices'] = pd.cut(x=t - SD['demography']['birthDate'], bins=params['treatmentAgeGroupBreaks'],
        labels=np.arange(0, len(params['treatmentAgeGroupBreaks']) - 1)).to_numpy()
    if params['adherenceSetting']=="betabinomial":
        newTreatmentAgeGroup = pd.cut(x=t - SD['demography']['birthDate'], bins=params['treatmentAgeGroupBreaks'],
            labels=np.arange(0, len(params['treatmentAgeGroupBreaks']) - 1)).to_numpy()
        if sum(params['coverage'][SD['treatmentAgeGroupIndices']] != params['coverage'][newTreatmentAgeGroup])>0:
            agedIndices = np.where(params['coverage'][SD['treatmentAgeGroupIndices']] != params['coverage'][newTreatmentAgeGroup])
            ageCoverage = np.where(params['coverage'][newTreatmentAgeGroup[agedIndices]]==0, np.nan, params['coverage'][newTreatmentAgeGroup[agedIndices]]/(1 - params['propNeverCompliers']))
            SD['adherenceFactors'][agedIndices] = np.random.beta(
                a=ageCoverage* (1 - params['rhoAdherence'])/params['rhoAdherence'],
                b=(1 - ageCoverage)*(1 - params['rhoAdherence'])/params['rhoAdherence'], 
                size=sum(params['coverage'][SD['treatmentAgeGroupIndices']] != params['coverage'][newTreatmentAgeGroup]))
        SD['adherenceFactors'][~ SD['compliers']] = 0
        SD['treatmentAgeGroupIndices'] = newTreatmentAgeGroup

    return SD

def doChemo(params, SD, t, coverage):

    '''
    Chemoterapy function.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    SD: dict
        dictionary containing the initial equilibrium parameter values;
    t: int
        time step;
    coverage: array
        coverage fractions;
    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''

    # decide which individuals are treated, treatment is random
    if params['adherenceSetting'] == "systematic":
        attendance = np.random.uniform(low=0, high=1, size=params['N']) < coverage[SD['treatmentAgeGroupIndices']] / (1 - params['propNeverCompliers'])
    if params['adherenceSetting'] == "betabinomial":                                                                                                                                         
        attendance = np.random.uniform(low=0, high=1, size=params['N']) < SD['adherenceFactors']

    # they're compliers and it's their turn
    toTreatNow = np.logical_and(attendance, SD['compliers'])

    # calculate the number of dead worms
    femaleToDie = np.random.binomial(size=np.sum(toTreatNow), n=SD['worms']['female'][toTreatNow],
    p=params['DrugEfficacy'])

    maleToDie = np.random.binomial(size=np.sum(toTreatNow), n=SD['worms']['total'][toTreatNow] -
    SD['worms']['female'][toTreatNow], p=params['DrugEfficacy'])

    juvenileToDie = np.random.binomial(size=np.sum(toTreatNow), n=SD['worms']['juvenile'][toTreatNow], p=params['JuvenileDrugEfficacy'])

    SD['worms']['female'][toTreatNow] -= femaleToDie
    SD['worms']['total'][toTreatNow] -= (maleToDie + femaleToDie)
    SD['worms']['juvenile'][toTreatNow] -= juvenileToDie

    # save actual attendance record and the age of each host when treated
    SD['attendanceRecord'].append(toTreatNow)
    SD['ageAtChemo'].append(t - SD['demography']['birthDate'])
    SD['adherenceFactorAtChemo'].append(SD['adherenceFactors'])

    params['DrugEfficacy'] = params['DrugEfficacy'] * (1-params['drugResistance'])

    return SD

def getPsi(params):

    '''
    This function calculates the psi parameter.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    Returns
    -------
    value of the psi parameter;
    '''

    # higher resolution
    deltaT = 0.1

    # inteval-centered ages for the age intervals, midpoints from 0 to maxHostAge
    modelAges = np.arange(start=0, stop=params['maxHostAge'], step=deltaT) + 0.5 * deltaT

    # hostMu for the new age intervals
    hostMu = params['hostMuData'][pd.cut(x=modelAges, bins=params['muBreaks'], labels=np.arange(start=0,
    stop=len(params['hostMuData']))).to_numpy()]

    hostSurvivalCurve = np.exp(-np.cumsum(hostMu * deltaT))
    MeanLifespan = np.sum(hostSurvivalCurve[:len(modelAges)]) * deltaT

    # calculate the cumulative sum of host and worm death rates from which to calculate worm survival
    # intMeanWormDeathEvents = np.cumsum(hostMu + params['sigma']) * deltaT # commented out as it is not used

    modelAgeGroupCatIndex = pd.cut(x=modelAges, bins=params['contactAgeGroupBreaks'], labels=np.arange(start=0,
    stop=len(params['contactAgeGroupBreaks']) - 1)).to_numpy()

    betaAge = params['contactRates'][modelAgeGroupCatIndex]
    rhoAge = params['rho'][modelAgeGroupCatIndex]

    wSurvival = np.exp(-params['sigma'] * modelAges)

    B = np.array([np.sum(betaAge[: i] * np.flip(wSurvival[: i])) * deltaT for i in range(1, 1 + len(hostMu))]) * params['juvenileSurvival']

    return params['R0'] * MeanLifespan * params['LDecayRate'] / (params['lambda'] * params['z'] * np.sum(rhoAge * hostSurvivalCurve * B) * deltaT )

def getLifeSpans(nSpans, params):

    '''
    This function draws the lifespans from the population survival curve.
    Parameters
    ----------
    nSpans: int
        number of drawings;
    params: dict
        dictionary containing the parameter names and values;
    Returns
    -------
    array containing the lifespan drawings;
    '''

    u = np.random.uniform(low=0, high=1, size=nSpans) * np.max(params['hostAgeCumulDistr'])
    spans = np.array([np.argmax(u[i] < params['hostAgeCumulDistr']) for i in range(nSpans)])

    return params['muAges'][spans]

def getEquilibrium(params):

    '''
    This function returns a dictionary containing the equilibrium worm burden
    with age and the reservoir value as well as the breakpoint reservoir value
    and other parameter settings.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    Returns
    -------
    dictionary containing the equilibrium parameter settings;
    '''

    # higher resolution
    deltaT = 0.1

    # inteval-centered ages for the age intervals, midpoints from 0 to maxHostAge
    modelAges = np.arange(start=0, stop=params['maxHostAge'], step=deltaT) + 0.5 * deltaT

    # hostMu for the new age intervals
    hostMu = params['hostMuData'][pd.cut(x=modelAges, bins=params['muBreaks'], labels=np.arange(start=0,
    stop=len(params['hostMuData'])))]

    hostSurvivalCurve = np.exp(-np.cumsum(hostMu * deltaT))
    MeanLifespan = np.sum(hostSurvivalCurve[:len(modelAges)]) * deltaT

    modelAgeGroupCatIndex = pd.cut(x=modelAges, bins=params['contactAgeBreaks'], labels=np.arange(start=0,
    stop=len(params['contactAgeBreaks']) - 1)).to_numpy()

    betaAge = params['contactRates'][modelAgeGroupCatIndex]
    rhoAge = params['rho'][modelAgeGroupCatIndex]

    wSurvival = np.exp(-params['sigma'] * modelAges)

    # this variable times L is the equilibrium worm burden
    Q = np.array([np.sum(betaAge[: i] * np.flip(wSurvival[: i])) * deltaT for i in range(1, 1 + len(hostMu))])

    # converts L values into mean force of infection
    FOIMultiplier = np.sum(betaAge * hostSurvivalCurve) * deltaT / MeanLifespan

    # upper bound on L
    SRhoT = np.sum(hostSurvivalCurve * rhoAge) * deltaT
    R_power = 1 / (params['k'] + 1)
    L_hat = params['z'] * params['lambda'] * params['psi'] * SRhoT * params['k'] * (params['R0'] ** R_power - 1) / \
    (params['R0'] * MeanLifespan * params['LDecayRate'] * (1 - params['z']))


    # L_hat = params['z'] * params['k'] * (params['R0'] ** R_power - 1) / \
    # (params['R0']  * (1 - params['z']))

    # now evaluate the function K across a series of L values and find point near breakpoint;
    # L_minus is the value that gives an age-averaged worm burden of 1; negative growth should
    # exist somewhere below this
    L_minus = MeanLifespan / np.sum(Q * hostSurvivalCurve * deltaT)
    test_L = np.append(np.linspace(start=0, stop=L_minus, num=10), np.linspace(start=L_minus, stop=L_hat, num=20))

    def K_valueFunc(currentL, params):
    
        return params['psi'] * np.sum(params['reproFunc'](currentL * Q, params) * rhoAge * hostSurvivalCurve * deltaT) / \
        (MeanLifespan * params['LDecayRate']) - currentL

    K_values = np.vectorize(K_valueFunc)(currentL=test_L, params=params)

    # now find the maximum of K_values and use bisection to find critical Ls
    iMax = np.argmax(K_values)
    mid_L = test_L[iMax]

    if K_values[iMax] < 0:

        return dict(stableProfile=0 * Q,
                    ageValues=modelAges,
                    L_stable=0,
                    L_breakpoint=np.nan,
                    K_values=K_values,
                    L_values=test_L,
                    FOIMultiplier=FOIMultiplier)

    # find the top L
    L_stable = bisect(f=K_valueFunc, a=mid_L, b=4 * L_hat, args=(params))
    
    if L_stable < 0:

        return dict(stableProfile=0 * Q,
                    ageValues=modelAges,
                    L_stable=0,
                    L_breakpoint=np.nan,
                    K_values=K_values,
                    L_values=test_L,
                    FOIMultiplier=FOIMultiplier)
    
    if np.isnan(L_stable):

        return dict(stableProfile=0 * Q,
                    ageValues=modelAges,
                    L_stable=0,
                    L_breakpoint=np.nan,
                    K_values=K_values,
                    L_values=test_L,
                    FOIMultiplier=FOIMultiplier)

    # find the unstable L
    L_break = test_L[1] / 50

    if K_valueFunc(L_break, params) < 0: # if it is less than zero at this point, find the zero
        L_break = bisect(f=K_valueFunc, a=L_break, b=mid_L, args=(params))

    stableProfile = L_stable * Q


    return dict(stableProfile=stableProfile,
                ageValues=modelAges,
                hostSurvival=hostSurvivalCurve,
                L_stable=L_stable,
                L_breakpoint=L_break,
                K_values=K_values,
                L_values=test_L,
                FOIMultiplier=FOIMultiplier)

def extractHostData(results):

    '''
    This function is used for processing results the raw simulation results.
    Parameters
    ----------
    results: list
        raw simulation output;
    Returns
    -------
    output: list
        processed simulation output;
    '''

    output = []

    for rep in range(len(results)):

        output.append(dict(
            wormsOverTime=np.array([results[rep][i]['worms']['total'] for i in range(len(results[0]) - 1)]).T,
            femaleWormsOverTime=np.array([results[rep][i]['worms']['female'] for i in range(len(results[0]) - 1)]).T,
            juvenileWormsOverTime=np.array([results[rep][i]['worms']['juvenile'] for i in range(len(results[0]) - 1)]).T,
            freeLiving=np.array([results[rep][i]['freeLiving'] for i in range(len(results[0]) - 1)]),
            ages=np.array([results[rep][i]['time'] - results[rep][i]['hosts']['birthDate'] for i in range(len(results[0]) - 1)]).T,
            # adherenceFactors=np.array([results[rep][i]['adherenceFactors'] for i in range(len(results[0]) - 1)]).T,
            # compliers=np.array([results[rep][i]['compliers'] for i in range(len(results[0]) - 1)]).T,
            # totalPop=len(results[rep][0]['worms']['total']),
            timePoints=np.array([results[rep][i]['time'] for i in range(len(results[0]) - 1)]),
            # attendanceRecord=results[rep][-1]['attendanceRecord'],
            # ageAtChemo=results[rep][-1]['ageAtChemo'],
            # finalFreeLiving=results[rep][-2]['freeLiving'],
            # adherenceFactorAtChemo=results[rep][-1]['adherenceFactorAtChemo']
            lifetimeWorms = np.array([results[rep][i]['lifetimeWorms'] for i in range(len(results[0]) - 1)]).T,
            yearsInfected = np.array([results[rep][i]['yearsInfected'] for i in range(len(results[0]) - 1)]).T
                            ))


    return output

def getSetOfEggCounts(total, female, params, Unfertilized=False):

    '''
    This function returns a set of readings of egg counts from a vector of individuals,
    according to their reproductive biology.
    Parameters
    ----------
    total: int
        array of total worms;
    female: int
        array of female worms;
    params: dict
        dictionary containing the parameter names and values;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    Returns
    -------
    random set of egg count readings from a single sample;
    '''

    if Unfertilized:

        meanCount = female * params['lambda'] * params['z'] ** female

    else:
        if params['reproFuncName'] == 'epgFertility' and params['SR']:
            productivefemaleworms = np.where(total == female, 0, female)

        elif params['reproFuncName'] == 'epgFertility' and not params['SR']:
            productivefemaleworms = female

        # monogamous reproduction; only pairs of worms produce eggs
        elif params['reproFuncName'] == 'epgMonog':
            productivefemaleworms = np.minimum(total - female, female)
        
        # eggProducers = np.where(total == female, 0, female)
        eggProducers = productivefemaleworms   
        meanCount = eggProducers * params['lambda'] * params['z'] ** total

    return np.random.negative_binomial(size=len(meanCount), p=params['k_epg'] / (meanCount + params['k_epg']), n=params['k_epg'])

def getVillageMeanCountsByHost(villageList, timeIndex, params, nSamples=2, Unfertilized=False):

    '''
    This function returns the mean egg count across readings by host
    for a given time point and iteration.
    Parameters
    ----------
    villageList: dict
        processed simulation output for a given iteration;
    timeIndex: int
        selected time point index;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    Returns
    -------
    array of mean egg counts;
    '''

    meanEggsByHost = getSetOfEggCounts(villageList['wormsOverTime'][:, timeIndex],
        villageList['femaleWormsOverTime'][:, timeIndex], params, Unfertilized) / nSamples

    for i in range(1, nSamples):

        meanEggsByHost += getSetOfEggCounts(villageList['wormsOverTime'][:, timeIndex],
        villageList['femaleWormsOverTime'][:, timeIndex], params, Unfertilized) / nSamples

    return meanEggsByHost

def getSampledDetectedMeanEggByVillage(hostData, timeIndex, params, nSamples=2, Unfertilized=False,
    villageSampleSize=100):

    '''
    This function provides sampled, age-cat worm prevalence
    at a given time point across all iterations.
    Parameters
    ----------
    hostData: dict
        processed simulation output;
    timeIndex: int
        selected time point index;
    ageBand: int
        array with age group boundaries;
    params: dict
        dictionary containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    '''
    
    mySample = np.array([getVillageMeanCountsByHost(villageList, timeIndex, params,
        nSamples, Unfertilized) for villageList in hostData])
    
    #print(mySample)
    return mySample
    #return np.sum(mySample)/villageSampleSize

def getAgeCatSampledPrevByVillage(villageList, timeIndex, ageBand, params, nSamples=2, Unfertilized=False,
    villageSampleSize=100):

    '''
    This function provides sampled, age-cat worm prevalence
    for a given time point and iteration.
    Parameters
    ----------
    villageList: dict
        processed simulation output for a given iteration;
    timeIndex: int
        selected time point index;
    ageBand: int
        array with age group boundaries;
    params: dict
        dictionary containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    '''

    meanEggCounts = getVillageMeanCountsByHost(villageList, timeIndex, params, nSamples, Unfertilized)

    ageGroups = pd.cut(x=villageList['ages'][:, timeIndex], bins=np.append(-10, np.append(ageBand, 150)),
    labels=np.array([1, 2, 3])).to_numpy()

    currentAgeGroupMeanEggCounts = meanEggCounts[ageGroups == 2]

    if villageSampleSize < len(currentAgeGroupMeanEggCounts):
        mySample = np.random.choice(a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=False)

    else:
        mySample = np.random.choice(a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=True)

    return np.sum(nSamples * mySample > 0.9) / villageSampleSize

def getAgeCatSampledPrevHeavyBurdenByVillage(villageList, timeIndex, ageBand, params, nSamples=2, Unfertilized=False,
    villageSampleSize=100):

    '''
    This function provides sampled, age-cat worm prevalence
    for a given time point and iteration.
    Parameters
    ----------
    villageList: dict
        processed simulation output for a given iteration;
    timeIndex: int
        selected time point index;
    ageBand: int
        array with age group boundaries;
    params: dict
        dictionary containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    '''

    meanEggCounts = getVillageMeanCountsByHost(villageList, timeIndex, params, nSamples, Unfertilized)

    ageGroups = pd.cut(x=villageList['ages'][:, timeIndex], bins=np.append(-10, np.append(ageBand, 150)),
    labels=np.array([1, 2, 3])).to_numpy()

    currentAgeGroupMeanEggCounts = meanEggCounts[ageGroups == 2]

    if villageSampleSize < len(currentAgeGroupMeanEggCounts):
        mySample = np.random.choice(a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=False)

    else:
        mySample = np.random.choice(a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=True)

    return np.sum(mySample >= params['burdenThreshold']) / villageSampleSize

def getSampledDetectedPrevByVillage(hostData, timeIndex, ageBand, params, nSamples=2, Unfertilized=False,
    villageSampleSize=100):

    '''
    This function provides sampled, age-cat worm prevalence
    at a given time point across all iterations.
    Parameters
    ----------
    hostData: dict
        processed simulation output;
    timeIndex: int
        selected time point index;
    ageBand: int
        array with age group boundaries;
    params: dict
        dictionary containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    '''

    return np.array([getAgeCatSampledPrevByVillage(villageList, timeIndex, ageBand, params,
    nSamples, Unfertilized, villageSampleSize) for villageList in hostData])

def getSampledDetectedPrevHeavyBurdenByVillage(hostData, timeIndex, ageBand, params, nSamples=2, Unfertilized=False,
    villageSampleSize=100):

    '''
    This function provides sampled, age-cat worm prevalence
    at a given time point across all iterations.
    Parameters
    ----------
    hostData: dict
        processed simulation output;
    timeIndex: int
        selected time point index;
    ageBand: int
        array with age group boundaries;
    params: dict
        dictionary containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    '''

    return np.array([getAgeCatSampledPrevHeavyBurdenByVillage(villageList, timeIndex, ageBand, params,
    nSamples, Unfertilized, villageSampleSize) for villageList in hostData])

def getPrevalence(hostData, params, numReps, nSamples=2, Unfertilized=False, villageSampleSize=100):

    '''
    This function provides the average SAC and adult prevalence at each time point,
    where the average is calculated across all iterations.
    Parameters
    ----------
    hostData: dict
        processed simulation output;
    params: dict
        dictionary containing the parameter names and values;
    numReps: int
        number of simulations;
    nSamples: int
        number of diagnostic samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    data frame with SAC and adult prevalence at each time point;
    '''

    psac_results = np.array([getSampledDetectedPrevByVillage(hostData, t, np.array([2, 5]), params, nSamples,
    Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    sac_results = np.array([getSampledDetectedPrevByVillage(hostData, t, np.array([5, 14]), params, nSamples,
    Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    adult_results = np.array([getSampledDetectedPrevByVillage(hostData, t, np.array([14, 90]), params, nSamples,
    Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    psac_heavy_results = np.array([getSampledDetectedPrevHeavyBurdenByVillage(hostData, t, np.array([2, 5]), params,
    nSamples, Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    sac_heavy_results = np.array([getSampledDetectedPrevHeavyBurdenByVillage(hostData, t, np.array([5, 14]), params,
    nSamples, Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    adult_heavy_results = np.array([getSampledDetectedPrevHeavyBurdenByVillage(hostData, t, np.array([14, 90]), params,
    nSamples, Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    egg_results = np.array([getSampledDetectedPrevByVillage(hostData, t, np.array([0, 90]), params,
    nSamples, Unfertilized,villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    # egg_results = np.array([getSampledDetectedMeanEggByVillage(hostData, t, params,
    # nSamples, Unfertilized,villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    psac_prevalence = np.sum(psac_results, axis=1) / numReps
    psac_centiles = np.percentile(psac_results,[2.5,97.5],axis=1)
    sac_prevalence = np.sum(sac_results, axis=1) / numReps
    sac_centiles = np.percentile(sac_results,[2.5,97.5],axis=1)
    adult_prevalence = np.sum(adult_results, axis=1) / numReps
    adult_centiles = np.percentile(adult_results,[2.5,97.5],axis=1)
    
    
    psac_heavy_prevalence = np.sum(psac_heavy_results, axis=1) / numReps
    psac_heavy_centiles = np.percentile(psac_heavy_results,[2.5,97.5],axis=1)
    sac_heavy_prevalence = np.sum(sac_heavy_results, axis=1) / numReps
    sac_heavy_centiles = np.percentile(sac_heavy_results,[2.5,97.5],axis=1)
    adult_heavy_prevalence = np.sum(adult_heavy_results, axis=1) / numReps
    adult_heavy_centiles = np.percentile(adult_heavy_results,[2.5,97.5],axis=1)

    chemoStart = list(params['outTimings']).index(params['chemoTimings'][0]) - 1
    # who_goal_0 = np.mean(np.sum(egg_results, axis=2)==0,axis=1)
    who_goal_0 = np.mean(egg_results==0,axis=1)
    who_goal_1 = np.sum(sac_heavy_results<0.01, axis=1) / numReps
    # who_goal_5 = np.sum(sac_heavy_results<0.05, axis=1) / numReps
    who_goal_active =  np.sum(sac_heavy_results[:,egg_results[chemoStart,:]>0]<0.01, axis=1) / np.sum(egg_results[chemoStart,:]>0)


    # egg_results = np.sum(egg_results, axis=1) / numReps

    worm_results = np.zeros((len(params['outTimings']), len(hostData), params['N']))
    juvenile_results = np.zeros(np.shape(worm_results))
    # lifetime_results = np.zeros(np.shape(egg_results))
    # yearsInfected_results = np.zeros(np.shape(egg_results))
    host_ages = np.zeros(np.shape(worm_results))
    # propJuvenile1 = np.zeros((len(params['outTimings']),len(hostData)))
    worms1 = np.zeros((len(params['outTimings']),len(hostData)))
    juv1 = np.zeros((len(params['outTimings']),len(hostData)))
    free1 = np.zeros((len(params['outTimings']),len(hostData)))
    for i in range(len(hostData)):
        worm_results[:,i,:] = hostData[i]['wormsOverTime'].transpose()
        juvenile_results[:,i,:] = hostData[i]['juvenileWormsOverTime'].transpose()
        # lifetime_results[:,i,:] = hostData[i]['lifetimeWorms'].transpose()
        # yearsInfected_results[:,i,:] = hostData[i]['yearsInfected'].transpose()
        host_ages[:,i,:] = hostData[i]['ages'].transpose()
        worms1[:,i] = np.sum(hostData[i]['wormsOverTime'],axis=0)
        juv1[:,i] = np.sum(hostData[i]['juvenileWormsOverTime'],axis=0)
        # propJuvenile1[:,i] = np.sum(hostData[i]['juvenileWormsOverTime'],axis=0)/(np.sum(hostData[i]['juvenileWormsOverTime'],axis=0)+np.sum(hostData[i]['wormsOverTime'],axis=0))
        free1[:,i] =hostData[i]['freeLiving']

    # propJuvenile = np.mean(propJuvenile1, axis=1)
    wot = np.mean(worms1, axis=1)
    jot = np.mean(juv1, axis=1)
    freeLiving = np.mean(free1, axis=1)

    # adultPrevalence = np.sum(worm_results>0,axis=2)/np.size(worm_results,axis=2)
    allWormPrevalence = np.sum(juvenile_results+worm_results>0,axis=2)/np.size(juvenile_results,axis=2)
    EoT = np.sum(np.sum(worm_results,axis=2)==0,axis=1)/numReps
    # EoT_centiles = np.percentile(np.sum(worm_results,axis=2)==0,[2.5,97.5],axis=1)

    df = pd.DataFrame({'Time': hostData[0]['timePoints'],
                       'PSAC Prevalence': psac_prevalence,
                       'PSAC Prevalence_2.5': psac_centiles[0,:],
                       'PSAC Prevalence_97.5': psac_centiles[1,:],
                       'SAC Prevalence': sac_prevalence,
                       'SAC Prevalence_2.5': sac_centiles[0,:],
                       'SAC Prevalence_97.5': sac_centiles[1,:],
                       'Adult Prevalence': adult_prevalence,
                       'Adult Prevalence_2.5': adult_centiles[0,:],
                       'Adult Prevalence_97.5': adult_centiles[1,:],
                       'PSAC Heavy Intensity Prevalence': psac_heavy_prevalence,
                       'PSAC Heavy Prevalence_2.5': psac_heavy_centiles[0,:],
                       'PSAC Heavy Prevalence_97.5': psac_heavy_centiles[1,:],
                       'SAC Heavy Intensity Prevalence': sac_heavy_prevalence,
                       'SAC Heavy Prevalence_2.5': sac_heavy_centiles[0,:],
                       'SAC Heavy Prevalence_97.5': sac_heavy_centiles[1,:],
                       'Adult Heavy Intensity Prevalence': adult_heavy_prevalence,
                       'Adult Heavy Prevalence_2.5': adult_heavy_centiles[0,:],
                       'Adult Heavy Prevalence_97.5': adult_heavy_centiles[1,:],
                       'WHO elimination probability': who_goal_0,
                       'WHO 1% heavy SAC probability': who_goal_1,
                       'WHO 1% active sims': who_goal_active,
                       'WoT': wot,
                       'JoT': jot,
                       'freeLiving': freeLiving,
                       'EoT': EoT
                      })

    #df = df[(df['Time'] >= 50) & (df['Time'] <= 64)]
    #df['Time'] = df['Time'] - 50
    
    df_sac = pd.DataFrame({'Time': hostData[0]['timePoints'],
                       'SAC Prevalence': sac_prevalence,
                       'Adult Prevalence': adult_prevalence,
                       'SAC Heavy Intensity Prevalence': sac_heavy_prevalence,
                       'Adult Heavy Intensity Prevalence': adult_heavy_prevalence
                      })
    
    sac_results = pd.DataFrame(sac_results)
    sac_results = sac_results.add_suffix('_sac')

    adult_results = pd.DataFrame(adult_results)
    adult_results = adult_results.add_suffix('_adult')
    
 #   all_results = pd.concat([sac_results,adult_results],axis=1)
 #   all_results.insert(0,"time",hostData[0]['timePoints'])

    # all_results = dict(egg_results=egg_results.tolist(),worm_results=worm_results.tolist(),
    #                     juvenile_results=juvenile_results.tolist(), ages=host_ages.tolist() )
    all_results = dict(egg_results=egg_results.tolist(),worm_results=allWormPrevalence.tolist() )
    return df, all_results

def getDistribution(hostData):
    
    wormBurdens=hostData[0]['wormsOverTime']
    M = np.zeros((np.size(hostData[0]['wormsOverTime'],axis=1),len(hostData)))
    P = np.zeros((np.size(hostData[0]['wormsOverTime'],axis=1),len(hostData)))
    for i in range(1,len(hostData)):
        wormBurdens = np.concatenate((wormBurdens, hostData[i]['wormsOverTime']))

    for i in range(len(hostData)):
        M[:,i] = np.mean(hostData[i]['wormsOverTime'],axis=0)
        P[:,i] = (hostData[i]['wormsOverTime'] != 0).sum(0)/np.size(hostData[i]['wormsOverTime'],0)
    
    return  M, P, wormBurdens