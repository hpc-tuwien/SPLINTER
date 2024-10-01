import pandas as pd
import os
from tqdm import tqdm
import optuna

from controller.moop_solver import extract_from_pareto, sort_pareto_front, LOCAL_COMPUTE_IDX


# Function to generate the progress file path based on the network
def get_progress_file(network):
    return f'experiment_progress_{network}.csv'


def read_and_filter_csv(csv_path, network):
    """Read CSV and filter for the specific network."""
    df = pd.read_csv(csv_path)
    # Assuming 'Network' column in the CSV corresponds to the network model type
    df_filtered = df[df['Network'] == network]
    return df_filtered


def run_experiment_for_qos(model, qos, strategy, params):
    """Run the configuration for a specific QoS request and return the results."""
    # Simulate running the configuration - this will depend on your objective() function
    latency, energy = 100.0, 10.0  # Replace with actual logic or function call
    return {'latency': latency, 'energy': energy}


def save_run_result(df_progress, result, network):
    """Append the result to the progress DataFrame and save."""
    df_progress = df_progress.append(result, ignore_index=True)
    save_progress(df_progress, network)


def execute_static_strategy(index, model, qos, strategy_name, params, df_progress, network):
    """Execute a static strategy and save the result."""
    result = run_experiment_for_qos(model, qos, strategy_name, params)
    result.update({'index': index, 'qos': qos, 'strategy': strategy_name, **params})
    save_run_result(df_progress, result, network)


def execute_static_strategies(index, model, qos, energy_config, latency_config, df_progress, network,
                              exhaustive_pareto):
    """Execute and save the static strategies. Skip cloud and edge if exhaustive Pareto is True."""
    if not exhaustive_pareto:
        # Cloud config
        cloud_params = {'cpu-freq': 1800, 'layer': 0, 'edge-accelerator': 'off', 'server-accelerator': True}
        execute_static_strategy(index, model, qos, 'cloud', cloud_params, df_progress, network)

        # Edge config
        edge_params = {'cpu-freq': 1800, 'layer': LOCAL_COMPUTE_IDX[model],
                       'edge-accelerator': 'max' if model != 'vit' else 'off', 'server-accelerator': False}
        execute_static_strategy(index, model, qos, 'edge', edge_params, df_progress, network)

    # Energy Pareto front config
    execute_static_strategy(index, model, qos, 'energy', energy_config, df_progress, network)

    # Latency Pareto front config
    execute_static_strategy(index, model, qos, 'latency', latency_config, df_progress, network)


def execute_dynamic_strategy_for_qos(index, model, qos, splinter_pareto_front, df_progress, network):
    """Execute SPLINTER latency strategy dynamically and save the result."""
    splinter_config = extract_from_pareto('splinter:latency', splinter_pareto_front, qos)
    splinter_result = run_experiment_for_qos(model, qos, 'splinter:latency', splinter_config)
    splinter_result.update({'index': index, 'qos': qos, 'strategy': 'splinter:latency', **splinter_config})
    save_run_result(df_progress, splinter_result, network)


def save_progress(df, network):
    """Save current progress to CSV for the given network."""
    progress_file = get_progress_file(network)
    df.to_csv(progress_file, index=False)


def load_progress(network):
    """Load progress from CSV if exists for the given network."""
    progress_file = get_progress_file(network)
    if os.path.exists(progress_file):
        return pd.read_csv(progress_file)
    return pd.DataFrame()


def sort_pareto_fronts(pareto_front):
    """Sort the Pareto front in three ways: energy, latency, splinter:latency."""
    energy_sorted = sort_pareto_front(pareto_front, 'energy')
    latency_sorted = sort_pareto_front(pareto_front, 'latency')
    splinter_latency_sorted = sort_pareto_front(pareto_front, 'splinter:latency')
    return energy_sorted, latency_sorted, splinter_latency_sorted


def load_pareto_front_from_study(study_name):
    """Load Pareto front from the Optuna study."""
    study = optuna.load_study(study_name=study_name, storage='sqlite:///splinter.db')
    return study.best_trials


def get_study_name(model, exhaustive_pareto):
    """Infer the study name based on the model and whether exhaustive Pareto is enabled."""
    if exhaustive_pareto:
        return f"{model}_grid_1_pruning_True"
    return f"{model}_nsga_0.2_pruning_True"


def continue_experiment(model, csv_path, exhaustive_pareto=False):
    """Main function to continue the experiment."""
    # Infer the study name based on the model and exhaustive Pareto flag
    study_name = get_study_name(model, exhaustive_pareto)

    # Load progress if exists for the specific model
    df_progress = load_progress(model)

    # Read and filter the CSV
    df_filtered = read_and_filter_csv(csv_path, model)

    # Load the Pareto front from the Optuna study
    pareto_front = load_pareto_front_from_study(study_name)

    # Sort Pareto front in different ways
    energy_sorted, latency_sorted, splinter_latency_sorted = sort_pareto_fronts(pareto_front)

    # Extract energy and latency configurations (only once)
    energy_config = extract_from_pareto('energy', energy_sorted)
    latency_config = extract_from_pareto('latency', latency_sorted)

    # For each QoS request in the CSV, execute all strategies
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"Processing QoS for {model}"):
        qos = row['Latency']  # Assuming 'Latency' column exists in CSV
        index = row['Index']  # Assuming 'Index' column exists in CSV

        # Check if this specific index and strategy have been processed already
        if not df_progress.empty and (df_progress['index'] == index).any():
            # If the index exists, check which strategies have been processed
            processed_strategies = df_progress[df_progress['index'] == index]['strategy'].unique()
            required_strategies = ['energy', 'latency', 'splinter:latency'] if exhaustive_pareto else ['cloud', 'edge',
                                                                                                       'energy',
                                                                                                       'latency',
                                                                                                       'splinter:latency']
            if all(strategy in processed_strategies for strategy in required_strategies):
                continue

        # Execute static strategies for this QoS request
        execute_static_strategies(index, model, qos, energy_config, latency_config, df_progress, model,
                                  exhaustive_pareto)

        # Execute dynamic SPLINTER strategy for this QoS request
        execute_dynamic_strategy_for_qos(index, model, qos, splinter_latency_sorted, df_progress, model)

    print(f"Experiment for {model} completed or interrupted.")

# Example usage:
# continue_experiment('vgg16', 'network_latency_50_samples.csv', exhaustive_pareto=True)
