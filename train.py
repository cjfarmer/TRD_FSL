import copy
import os
from typing import List
from trainer import Trainer
import numpy as np 
from config import TASKS, DATASETS_PATH, DATA_SEED

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_all_config(base_config):
    # get the para names that need to grid search
    search_name = [] # need to grid search
    fix_name = [] # fixed paras
    for k, v in base_config.items():
        if isinstance(v, List):
            search_name.append(k)
        else:
            fix_name.append(k)
    
    if not search_name:
        return [copy.deepcopy(base_config)]
        
    # construct a list of parameters setup to be selected
    grid_configs = []
    root_search_name = search_name[0]
    for v in base_config[root_search_name]:
        new_config = copy.deepcopy(base_config)
        new_config[root_search_name] = v
        grid_configs += get_all_config(new_config)

    return grid_configs


if __name__ == '__main__':
    trainer = Trainer()
    
    for task in TASKS:
        results = {'seed':[],'best_para':[], 'final_metric':[]} # collect results

        # grid search for every spilt of the task
        for seed_file in DATA_SEED:
            base_config = copy.deepcopy(TASKS[task])
            base_config['data_path'] = os.path.join(DATASETS_PATH, task, seed_file)

            # grid search
            print(f'{"#"*15} {task}, {seed_file}: searching for parameters {"#"*15}')
            search_configs = get_all_config(base_config)
            if len(search_configs) == 1:
                best_config = search_configs[0]
            else:
                best_eval = -999999
                best_config = None
                for config in search_configs:
                    trainer.set_paras(config)
                    metric = trainer.train_and_eval(dev_as_test=True)

                    if metric > best_eval:
                        best_eval = metric
                        best_config = config

            # train and evaluate based on best parameters
            print(f'{"#"*15} {task}, {seed_file}: final test {"#"*15}')
            trainer.set_paras(best_config)
            final_metric = trainer.train_and_eval()
            results['best_para'].append(best_config)
            results['final_metric'].append(final_metric)
            results['seed'].append(seed_file)

        # write the results to file
        with open('results.txt', 'a') as f_result:
            f_result.write(f'{task}: {best_config["metric"]}\n')
            metrics = np.array(results["final_metric"])
            for i in range(len(DATA_SEED)):
                f_result.write(f'\t{results["seed"][i]}: {metrics[i]} \t{str(results["best_para"][i])} \n')
            f_result.write(f'\t mean +- std: {metrics.mean()*100:.1f} ({metrics.std()*100:.1f})\n')
            