import sys
import os
import numpy as np
from pathlib import Path

from TopasOpt import Optimisers as to

BaseDirectory = os.path.expanduser("~") + '/Downloads/100724_beam_model'
SimulationName = 'Emittance_model_bayes'
OptimisationDirectory = '/Users/dchen/Desktop/Desktop_nickelfish/hello/Stanford/Oct/BeamModelOptimisation'

# set up optimisation params:
optimisation_params = {}
optimisation_params['ParameterNames'] = ['e1', 'e2', 'e3', 'e1w', 'e2w',
                                         'BeamPositionCutoffX', 'BeamPositionCutoffY',
                                         'BeamPositionSpreadX', 'BeamPositionSpreadY',
                                         'BeamAngularCutoffX', 'BeamAngularCutoffY',
                                         'BeamAngularSpreadX', 'BeamAngularSpreadY']
optimisation_params['UpperBounds'] = np.array([14, 14, 14, 100, 100,
                                               20, 20,
                                               10, 10,
                                               30, 30,
                                               0.1, 0.1])
optimisation_params['LowerBounds'] = np.array([0, 0, 0, 0.01, 0.01,
                                               10, 10,
                                               0.0001, 0.0001,
                                               1, 1,
                                               0.00001, 0.00001])
# optimisation_params['ParameterNames'] = ['e1', 'e2', 'e3', 'e1w', 'e2w', 'e3w',
#                                          'SigmaX', 'SigmaXprime', 'CorrelationX',
#                                          'SigmaY', 'SigmaYPrime', 'CorrelationY']
# optimisation_params['UpperBounds'] = np.array([14, 14, 14, 100, 100, 100,
#                                                15, 1, 0.99999,
#                                                15, 1, 0.99999])
# optimisation_params['LowerBounds'] = np.array([0, 0, 0, 0, 0, 0,
#                                                0, 0, -0.99999,
#                                                0, 0, -0.99999])
# generate a random starting point between our bounds (it doesn't have to be random, this is just for demonstration purposes)
random_start_point = np.random.default_rng().uniform(optimisation_params['LowerBounds'],
                                                     optimisation_params['UpperBounds'])
optimisation_params['start_point'] = np.array([12, 6, 3, 80, 15, 20, 20, 0.4, 0.4, 30, 30, 0.05, 0.05])

# optimisation_params['start_point'] = np.array([12, 6, 3, 80, 15, 5, 4, 0.0003, 0.988, 4, 0.001, -0.99])
optimisation_params['Nitterations'] = 30
# optimisation_params['Suggestions'] # you can suggest points to test if you want - we won't here.
ReadMeText = 'This is a public service announcement, this is only a test'

Optimiser = to.BayesianOptimiser(optimisation_params=optimisation_params, BaseDirectory=BaseDirectory,
                                 SimulationName=SimulationName,
                                 G4dataLocation = '/Applications/GEANT4/G4DATA',
                                 OptimisationDirectory=OptimisationDirectory,
                                 TopasLocation='/Applications/TOPAS/OpenTOPAS-install', ReadMeText=ReadMeText, Overwrite=True)
Optimiser.RunOptimisation()