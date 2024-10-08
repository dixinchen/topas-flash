from TopasOpt.TopasScriptGenerator import generate_topas_script_generator
from pathlib import Path

this_directory = Path(__file__).parent  # figures out where your working directory is located

# nb: the order is important to make sure that a phase space files are correctly classified - they should be entered in the same order they will be run
generate_topas_script_generator(this_directory, ['Topas_data/beam_model.txt'])