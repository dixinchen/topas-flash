from pathlib import Path

def GenerateTopasScripts(BaseDirectory, iteration, **variable_dict):
    """
    This file simply returns a list object, where each list entry corresponds to
    a line in the topas script.
    When it is called from an Optimiser object,it will receive a dictionary that contains the current values of 
    the variables you set up in optimisation_params when you initialised the optimiser.
    """
    
    beammodel = []
    beammodel.append('')
    beammodel.append('i:Ts/ShowHistoryCountAtInterval = 1000')
    beammodel.append('i:So/FLASHBeam/NumberOfHistoriesInRun = 100000')
    beammodel.append('sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-binarycascade" "g4h-elastic_HP" "g4stopping"')
    beammodel.append('Ge/QuitIfOverlapDetected = "True"')
    beammodel.append('i:Ts/NumberOfThreads = 20')
    beammodel.append('')
    beammodel.append('s:Ge/World/Material  = "Air"')
    beammodel.append('d:Ge/World/HLX       = 2 m')
    beammodel.append('d:Ge/World/HLY       = 2 m')
    beammodel.append('d:Ge/World/HLZ       = 2 m')
    beammodel.append('b:Ge/World/Invisible = "False"')
    beammodel.append('')
    beammodel.append('s:So/FLASHBeam/Type                     = "Beam"')
    beammodel.append('s:So/FLASHBeam/Component                = "BeamPosition"')
    beammodel.append('s:So/FLASHBeam/BeamParticle             = "e-"')
    # beammodel.append('dc:So/FLASHBeam/BeamEnergy               = ' + str(variable_dict['BeamEnergy']) + ' MeV')
    # beammodel.append('uc:So/FLASHBeam/BeamEnergySpread         = ' + str(variable_dict['BeamEnergySpread']))
    beammodel.append('s:So/FLASHBeam/BeamPositionDistribution = "Gaussian"')
    beammodel.append('s:So/FLASHBeam/BeamAngularDistribution  = "Gaussian"')
    beammodel.append('s:So/FLASHBeam/BeamPositionCutoffShape = "Ellipse"')
    beammodel.append('dc:So/FLASHBeam/BeamPositionCutoffX = ' + str(variable_dict['BeamPositionCutoffX']) + ' cm')
    beammodel.append('dc:So/FLASHBeam/BeamPositionCutoffY = ' + str(variable_dict['BeamPositionCutoffY']) + ' cm')
    beammodel.append('dc:So/FLASHBeam/BeamPositionSpreadX = ' + str(variable_dict['BeamPositionSpreadX']) + ' cm')
    beammodel.append('dc:So/FLASHBeam/BeamPositionSpreadY = ' + str(variable_dict['BeamPositionSpreadY']) + ' cm')
    beammodel.append('dc:So/FLASHBeam/BeamAngularCutoffX = ' + str(variable_dict['BeamAngularCutoffX']) + ' deg')
    beammodel.append('dc:So/FLASHBeam/BeamAngularCutoffY = ' + str(variable_dict['BeamAngularCutoffY']) + ' deg')
    beammodel.append('dc:So/FLASHBeam/BeamAngularSpreadX = ' + str(variable_dict['BeamAngularSpreadX']) + ' rad')
    beammodel.append('dc:So/FLASHBeam/BeamAngularSpreadY = ' + str(variable_dict['BeamAngularSpreadY']) + ' rad')

    # beammodel.append('s:So/FLASHBeam/Type                     = "Emittance"')
    # beammodel.append('s:So/FLASHBeam/Component                = "BeamPosition"')
    # beammodel.append('s:So/FLASHBeam/BeamParticle             = "e-"')
    # # beammodel.append('dc:So/FLASHBeam/BeamEnergy               = ' + str(variable_dict['BeamEnergy']) + ' MeV')
    # # beammodel.append('uc:So/FLASHBeam/BeamEnergySpread         = ' + str(variable_dict['BeamEnergySpread']))
    # beammodel.append('s:So/FLASHBeam/Distribution = "BiGaussian"')
    # beammodel.append('dc:So/FLASHBeam/SigmaX = ' + str(variable_dict['SigmaX']) + ' cm')
    # beammodel.append('dc:So/FLASHBeam/SigmaXprime = ' + str(variable_dict['SigmaXprime']))
    # beammodel.append('dc:So/FLASHBeam/CorrelationX = ' + str(variable_dict['CorrelationX']) + " cm")
    # beammodel.append('dc:So/FLASHBeam/SigmaY = ' + str(variable_dict['SigmaY']) + ' cm')
    # beammodel.append('dc:So/FLASHBeam/SigmaYPrime = ' + str(variable_dict['SigmaYPrime']))
    # beammodel.append('dc:So/FLASHBeam/CorrelationY = ' + str(variable_dict['CorrelationY']) + " cm")

    beammodel.append('s:So/FLASHBeam/BeamEnergySpectrumType     = "Continuous"')
    beammodel.append('dv:So/FLASHBeam/BeamEnergySpectrumValues = 3 ' + str(variable_dict['e1']) + " "  + str(variable_dict['e2']) + " "  + str(variable_dict['e3']) + ' MeV')
    beammodel.append('uv:So/FLASHBeam/BeamEnergySpectrumWeights = 3 ' + str(variable_dict['e1w']) + " " + str(variable_dict['e2w']) + " "
                     + str(int(100 - variable_dict['e1w'] - variable_dict['e2w'])))


    beammodel.append('')
    beammodel.append('s:Ge/BeamPosition/Parent   = "World"')
    beammodel.append('s:Ge/BeamPosition/Type     = "Group"')
    beammodel.append('d:Ge/BeamPosition/TransX   = 0. m')
    beammodel.append('d:Ge/BeamPosition/TransY   = 0. m')
    beammodel.append('d:Ge/BeamPosition/TransZ   = 0. cm')
    beammodel.append('d:Ge/BeamPosition/RotX     = 0. deg')
    beammodel.append('d:Ge/BeamPosition/RotY     = 0. deg')
    beammodel.append('d:Ge/BeamPosition/RotZ     = 180. deg')
    beammodel.append('')
    beammodel.append('s:Ge/Film1/Type     = "TsBox"')
    beammodel.append('s:Ge/Film1/Parent   = "World"')
    beammodel.append('s:Ge/Film1/Material = "G4_WATER"')
    beammodel.append('d:Ge/Film1/HLX      = 8.460559 cm')
    beammodel.append('d:Ge/Film1/HLY      = 8.460559 cm')
    beammodel.append('d:Ge/Film1/HLZ      = 0.139 mm')
    beammodel.append('d:Ge/Film1/TransX   = 0. cm')
    beammodel.append('d:Ge/Film1/TransY   = 0. cm')
    beammodel.append('d:Ge/Film1/TransZ   = 64.095 cm ')
    beammodel.append('d:Ge/Film1/RotX     = 0. deg')
    beammodel.append('d:Ge/Film1/RotY     = 0. deg')
    beammodel.append('d:Ge/Film1/RotZ     = 0. deg')
    beammodel.append('s:Ge/Film1/Color    = "skyblue"')
    beammodel.append('i:Ge/Film1/XBins = 100')
    beammodel.append('i:Ge/Film1/YBins = 100')
    beammodel.append('')
    beammodel.append('s:Ge/Film2/Type     = "TsBox"')
    beammodel.append('s:Ge/Film2/Parent   = "World"')
    beammodel.append('s:Ge/Film2/Material = "G4_WATER"')
    beammodel.append('d:Ge/Film2/HLX      = 8.460559 cm')
    beammodel.append('d:Ge/Film2/HLY      = 8.460559 cm')
    beammodel.append('d:Ge/Film2/HLZ      = 0.139 mm')
    beammodel.append('d:Ge/Film2/TransX   = 0. cm')
    beammodel.append('d:Ge/Film2/TransY   = 0. cm')
    beammodel.append('d:Ge/Film2/TransZ   = 73.956 cm ')
    beammodel.append('d:Ge/Film2/RotX     = 0. deg')
    beammodel.append('d:Ge/Film2/RotY     = 0. deg')
    beammodel.append('d:Ge/Film2/RotZ     = 0. deg')
    beammodel.append('s:Ge/Film2/Color    = "green"')
    beammodel.append('i:Ge/Film2/XBins = 100')
    beammodel.append('i:Ge/Film2/YBins = 100')
    beammodel.append('')
    beammodel.append('s:Ge/WaterTank/Type     = "TsBox"')
    beammodel.append('s:Ge/WaterTank/Parent   = "World"')
    beammodel.append('s:Ge/WaterTank/Material = "G4_WATER"')
    beammodel.append('d:Ge/WaterTank/HLX      = 10.0 cm')
    beammodel.append('d:Ge/WaterTank/HLY      = 10.0 cm')
    beammodel.append('d:Ge/WaterTank/HLZ      = 5.071535 cm')
    beammodel.append('d:Ge/WaterTank/TransX   = 0. cm')
    beammodel.append('d:Ge/WaterTank/TransY   = 0. cm')
    beammodel.append('d:Ge/WaterTank/TransZ   = 106 cm ')
    beammodel.append('d:Ge/WaterTank/RotX     = 0. deg')
    beammodel.append('d:Ge/WaterTank/RotY     = 0. deg')
    beammodel.append('d:Ge/WaterTank/RotZ     = 0. deg')
    beammodel.append('s:Ge/WaterTank/Color    = "green"')
    beammodel.append('i:Ge/WaterTank/XBins = 1')
    beammodel.append('i:Ge/WaterTank/YBins = 1')
    beammodel.append('i:Ge/WaterTank/ZBins = 600')
    beammodel.append('')
    beammodel.append('s:Sc/DoseAtFilm1/Quantity = "DoseToMedium"')
    beammodel.append('s:Sc/DoseAtFilm1/Component = "Film1"')
    beammodel.append('s:Sc/DoseAtFilm1/IfOutputFileAlreadyExists = "Overwrite"')
    beammodel.append('s:Sc/DoseAtFilm1/OutputFile =  "../Results/DoseAtFilm1_itt_' + str(iteration) + '"')
    beammodel.append('b:Sc/DoseAtFilm1/OutputToConsole = "False"')
    beammodel.append('')
    beammodel.append('s:Sc/DoseAtFilm2/Quantity = "DoseToMedium"')
    beammodel.append('s:Sc/DoseAtFilm2/Component = "Film2"')
    beammodel.append('s:Sc/DoseAtFilm2/IfOutputFileAlreadyExists = "Overwrite"')
    beammodel.append('s:Sc/DoseAtFilm2/OutputFile =  "../Results/DoseAtFilm2_itt_' + str(iteration) + '"')
    beammodel.append('b:Sc/DoseAtFilm2/OutputToConsole = "False"')
    beammodel.append('')
    beammodel.append('s:Sc/DoseAtWaterTank/Quantity = "DoseToMedium"')
    beammodel.append('s:Sc/DoseAtWaterTank/Component = "WaterTank"')
    beammodel.append('s:Sc/DoseAtWaterTank/IfOutputFileAlreadyExists = "Overwrite"')
    beammodel.append('s:Sc/DoseAtWaterTank/OutputFile =  "../Results/DoseAtWaterTank_itt_' + str(iteration) + '"')
    beammodel.append('b:Sc/DoseAtWaterTank/OutputToConsole = "False"')

    return [beammodel], ['beammodel']

if __name__ == "__main__":
    Scripts, ScriptNames = GenerateTopasScripts(".", 1)
    for i, script in enumerate(Scripts):
        filename = ScriptNames[i] + ".tps"
        f = open(filename, "w")
        for line in script:
            f.write(line)
            f.write("\n")
