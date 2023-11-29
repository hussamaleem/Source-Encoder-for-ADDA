
import optuna_class
import os
import pathlib
import yaml

main_path = pathlib.Path(__file__
)
with open(os.path.join(main_path.parents[0], 'config.yaml')) as file:
    config = yaml.safe_load(file)
    optuna_config = config['Optuna']


runner = optuna_class.OptunaOptim(n_trials= optuna_config['n_trials'])
runner.run_objective()
runner.create_summary() 
 