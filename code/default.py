import os
import sys

from printer import Console

# The purpose of this file is to run a required default setup.
# It is also possible to check if the requirements are met.

current_env = os.environ.get('CONDA_DEFAULT_ENV')
env_name = "MRSI_simulation"

if current_env != env_name:
    Console.printf("error", f"Activated conda environment: {current_env} \n"
                           f"However, '{env_name}' is required! \n"
                           f"Activate it in the terminal with: 'conda activate {env_name}'")
    sys.exit()
