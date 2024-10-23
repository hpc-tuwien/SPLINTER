import os

import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

from controller.evaluate import sort_pareto_fronts
from moop_solver import extract_from_pareto, LOCAL_COMPUTE_IDX

# Set the seed for reproducibility
np.random.seed(123456789)


# Function to generate the progress file path based on the network
def get_progress_file(network):
    return f'simulation_progress_{network}.csv'


def get_simulation_file(network):
    """Returns the file path for the simulation pool of a given network."""
    return f'simulation_pool_{network}.csv'


def read_and_filter_csv(csv_path, network):
    """Read CSV and filter for the specific network."""
    df = pd.read_csv(csv_path)
    # Assuming 'Network' column in the CSV corresponds to the network model type
    df_filtered = df[df['Network'] == network]
    return df_filtered


def load_simulation_pool(network):
    """Load the simulation pool CSV and return as a DataFrame."""
    simulation_file = get_simulation_file(network)
    if not os.path.exists(simulation_file):
        raise FileNotFoundError(f"Simulation pool file {simulation_file} not found.")
    return pd.read_csv(simulation_file)


def run_experiment_for_qos(model, params):
    """Simulate a configuration by randomly sampling from previously recorded runs."""
    filtered_pool = load_simulation_pool(model)
    for key, value in params.items():
        filtered_pool = filtered_pool[filtered_pool[key] == value]

    if filtered_pool.empty:
        print(f"No matching configurations found for parameters: {params}")
        return {}

    sampled_row = filtered_pool.sample(n=1).iloc[0]
    return sampled_row.to_dict()


def extract_config_from_series(series):
    """
    Convert a pandas.Series object returned from extract_from_pareto to a dictionary format
    similar to the cloud or edge configurations.
    """
    config = {
        'cpu-freq': series.get('cpu-freq'),
        'layer': series.get('layer'),
        'edge-accelerator': series.get('edge-accelerator'),
        'server-accelerator': series.get('server-accelerator')
    }
    return config


def execute_static_strategy(index, model, qos, strategy_name, params, network):
    """Execute a static strategy and save the result."""
    result = run_experiment_for_qos(model, params)
    result.update({'index': index, 'qos': qos, 'strategy': strategy_name, **params})
    return result


def execute_static_strategies(index, model, qos, energy_config, latency_config, network, pbar):
    """Execute and collect results for the static strategies."""
    results = []
    strategies = {
        'cloud': {'cpu-freq': 1800, 'layer': 0, 'edge-accelerator': 'off', 'server-accelerator': True},
        'edge': {'cpu-freq': 1800, 'layer': LOCAL_COMPUTE_IDX[model],
                 'edge-accelerator': 'max' if model == "vgg16" else "off",
                 'server-accelerator': False},
        'energy': energy_config,
        'latency': latency_config
    }

    for strategy_name, params in strategies.items():
        result = execute_static_strategy(index, model, qos, strategy_name, params, network)
        if result:
            results.append(result)
        pbar.update(1)

    return results


def get_study_name(model, exhaustive_pareto):
    """Infer the study name based on the model and whether exhaustive Pareto is enabled."""
    if exhaustive_pareto:
        return f"{model}_grid_1_pruning_True"
    return f"{model}_nsga_0.2_pruning_True"


def execute_dynamic_strategy_for_qos(index, model, qos, splinter_pareto_front, network, pbar, exhaustive_pareto):
    """Execute SPLINTER latency strategy dynamically and collect the result."""
    strategy_name = 'splinter:latency_exhaustive' if exhaustive_pareto else 'splinter:latency'
    splinter_series = extract_from_pareto('splinter:latency', splinter_pareto_front, qos)
    splinter_config = extract_config_from_series(splinter_series)

    splinter_result = run_experiment_for_qos(model, splinter_config)
    splinter_result.update({'index': index, 'qos': qos, 'strategy': strategy_name, **splinter_config})
    pbar.update(1)

    return splinter_result


def load_pareto_front_from_study(study_name):
    """Load Pareto front from the Optuna study."""
    study = optuna.load_study(study_name=study_name, storage='sqlite:///splinter.db')
    return study.best_trials


def continue_experiment(model, csv_path, exhaustive_pareto):
    """Main function to continue the experiment."""
    # Infer the study name based on the model and exhaustive Pareto flag
    study_name = get_study_name(model, exhaustive_pareto)

    # Load progress if exists for the specific model
    df_progress = []

    # Read and filter the CSV
    df_filtered = read_and_filter_csv(csv_path, model)

    # Load the Pareto front from the Optuna study
    pareto_front = load_pareto_front_from_study(study_name)

    # Sort Pareto front for splinter
    splinter_latency_sorted = sort_pareto_fronts(pareto_front, ['splinter:latency'])

    # Sort Pareto front for static baselines
    energy_sorted, latency_sorted = sort_pareto_fronts(pareto_front, ['energy', 'latency'])

    energy_series = extract_from_pareto('energy', energy_sorted)
    energy_config = extract_config_from_series(energy_series)

    latency_series = extract_from_pareto('latency', latency_sorted)
    latency_config = extract_config_from_series(latency_series)

    # For each QoS request in the CSV, execute all strategies and collect results
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"Processing QoS for {model}",
                         dynamic_ncols=True, ncols=100, leave=False):
        qos = row['Latency']
        index = row['Index']

        strategies_count = 1 if exhaustive_pareto else 5
        with tqdm(total=strategies_count, desc=f"Processing strategies for QoS index {index}", dynamic_ncols=True,
                  ncols=100, leave=False) as pbar:
            static_results = execute_static_strategies(index, model, qos, energy_config, latency_config, model, pbar)
            dynamic_result = execute_dynamic_strategy_for_qos(index, model, qos, splinter_latency_sorted, model, pbar,
                                                              exhaustive_pareto)

            # Combine all results for this QoS request
            df_progress.extend(static_results)
            df_progress.append(dynamic_result)

    # Save all results at the end
    result_file = f'{model}_simulation_results.csv'
    results_df = pd.DataFrame(df_progress)
    results_df.to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")


# In the main function or evaluation entry point
if __name__ == "__main__":
    continue_experiment('vit', 'network_latency_10000_samples.csv', False)
