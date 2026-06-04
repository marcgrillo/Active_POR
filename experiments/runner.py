import os
import json
import numpy as np
import time
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from common import utils
from experiments.simulation import process_single_table

def run_batch_experiments(F1, F2, F3, sub_fold, dataset_folds, alg, active_method, overwrite, hm=None, calculate_heuristic=False, generate_scatter_plots=False, mape_threshold=0.05, plot_mape_fit=False, n_samples_mc=2000, use_linear_approx=False, use_is_bald=False, check_passive_algs_completed=False, use_mh_sampler=False, use_hmc_sampler=False, num_cores=1, n_samples_mcmc=2000):
    """
    Orchestrates the experiments across multiple datasets and configurations.
    
    Args:
        hm (int, optional): Number of Human Models (tables) to process. 
                            If None, processes all available in the dataset.
        num_cores (int): Number of CPU cores for parallel table processing.
    """
    for dataset_fold in dataset_folds:
        print(f"\n=== Processing Dataset: {dataset_fold} ===")
        
        # Load Generation Params from datasets/dataset_name/
        params_path = os.path.join("datasets", dataset_fold, "generation_params.json")
        gen_params = {}
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                gen_params = json.load(f)
            print(f"Loaded generation parameters from {params_path}")
        else:
            print("Warning: generation_params.json not found. Using default lambda=inf.")

        samples_root = os.path.join("samples", dataset_fold)
        
        for f1 in F1:
            for f2 in F2:
                # Lookup Lambda
                config_key = f"f1_{f1}_f2_{f2}"
                current_lam = gen_params.get(config_key, np.inf)
                print(f"Configuration: f1={f1}, f2={f2} | Using Lambda={current_lam:.4f}")
                
                for f3 in F3:
                    try:
                        # Read From datasets/dataset_name/
                        tables, rankings, dm_prefs, Us = utils.read_dataset(os.path.join("datasets", dataset_fold), f1, f2, f3)
                    except FileNotFoundError as e:
                        print(f"Skipping {f1}/{f2}/{f3}: {e}")
                        continue
                        
                    num_dm_dec = int(np.round(f3 * (f1 * (f1 - 1) / 200)))
                    
                    config_dir = os.path.join(samples_root, f"f1_{f1}_f2_{f2}_f3_{f3}")
                    method_dir = os.path.join(config_dir, sub_fold)
                    utils.save_path(method_dir)

                    # Determine Shared Passive Directory
                    passive_method_dir = None
                    if check_passive_algs_completed:
                         # Construct shared folder name: e.g. "BAYES-BT_PASSIVE"
                         algo_type, model_type = alg.split('-') # e.g. BAYES, BT
                         passive_sub_fold = f"{algo_type}-{model_type}_PASSIVE"
                         passive_method_dir = os.path.join(config_dir, passive_sub_fold)
                         utils.save_path(passive_method_dir)

                    # Handle Missing Utilities
                    if Us is None or len(Us) == 0:
                        Us_iter = [None] * len(tables)
                    else:
                        Us_iter = Us

                    # --- SLICE DATASET BASED ON HM ---
                    if hm is not None:
                        # Ensure we don't exceed available data
                        limit = min(hm, len(tables))
                        tables = tables[:limit]
                        rankings = rankings[:limit]
                        dm_prefs = dm_prefs[:limit]
                        Us_iter = Us_iter[:limit]
                    
                    # Prepare tasks
                    tasks = []
                    for i, (table, ranking, prefs, true_u) in enumerate(zip(tables, rankings, dm_prefs, Us_iter)):
                        table_dir = os.path.join(method_dir, f"table_{i}")
                        utils.save_path(table_dir)
                        
                        task_args = {
                            'table': table,
                            'ground_truth_prefs': prefs,
                            'true_ranking': ranking,
                            'true_utility': true_u,
                            'num_steps': num_dm_dec,
                            'output_dir': table_dir,
                            'alg': alg,
                            'active_method': active_method,
                            'lam': current_lam,
                            'overwrite': overwrite,
                            'calculate_heuristic': calculate_heuristic,
                            'generate_scatter_plots': generate_scatter_plots,
                            'table_index': i,
                            'sub_fold': sub_fold,
                            'dataset_fold': dataset_fold,
                            'mape_threshold': mape_threshold,
                            'plot_mape_fit': plot_mape_fit,
                            'n_samples_mc': n_samples_mc,
                            'use_linear_approx': use_linear_approx,
                            'use_is_bald': use_is_bald,
                            'check_passive_algs_completed': check_passive_algs_completed,
                            'shared_passive_dir': passive_method_dir,
                            'use_mh_sampler': use_mh_sampler,
                            'use_hmc_sampler': use_hmc_sampler,
                            'n_samples_mcmc': n_samples_mcmc
                        }
                        tasks.append(task_args)

                    # Execute Tasks
                    if num_cores > 1 and len(tasks) > 1:
                        print(f"Running {len(tasks)} tables in parallel using {num_cores} cores...")
                        
                        with Manager() as manager:
                            # Global dictionary to track progress of each table
                            progress_dict = manager.dict({task['table_index']: 0 for task in tasks})
                            for task in tasks:
                                task['disable_tqdm'] = True
                                task['progress_dict'] = progress_dict

                            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                                # Submit all tasks
                                futures = [executor.submit(process_single_table, **task) for task in tasks]
                                
                                # Progress bar that tracks the SLOWEST worker
                                pbar = tqdm(total=num_dm_dec, desc="  Initializing...", leave=True)
                                last_min = 0
                                
                                while any(not f.done() for f in futures):
                                    # Identify slowest task
                                    vals = list(progress_dict.values())
                                    if not vals: break
                                    
                                    min_step = min(vals)
                                    # Identify table index of the bottleneck
                                    slowest_idx = -1
                                    for idx, step in progress_dict.items():
                                        if step == min_step:
                                            slowest_idx = idx
                                            break
                                    
                                    pbar.set_description(f"  Slowest: Table {slowest_idx}")
                                    
                                    if min_step > last_min:
                                        pbar.update(min_step - last_min)
                                        last_min = min_step
                                    
                                    time.sleep(0.5) # Poll every 0.5s
                                
                                # Complete bar to 100%
                                if num_dm_dec > last_min:
                                    pbar.update(num_dm_dec - last_min)
                                pbar.close()
                                
                                # Final check for worker exceptions
                                for f in futures:
                                    try:
                                        f.result()
                                    except Exception as exc:
                                        print(f"A worker generated an exception: {exc}")

                    else:
                        # Sequential execution - keep inner tqdm bars enabled
                        for task in tasks:
                            process_single_table(**task)