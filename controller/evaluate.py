import argparse
import csv
import os
import time

import optuna
import pandas as pd
import psutil
from pymeas.device import GPMDevice
from sklearn.metrics import auc
from tqdm import tqdm

overhead_log_path = 'splinter_overhead_metrics.csv'

from moop_solver import extract_from_pareto, sort_pareto_front, LOCAL_COMPUTE_IDX, get_ssh_client, setup_pi


# Utility function to log overhead metrics to CSV
def log_overhead_metrics(phase, start_time, end_time, cpu_usage, ram_usage):
    duration = (end_time - start_time) / 1e6  # Convert ns to ms
    file_exists = os.path.isfile(overhead_log_path)

    with open(overhead_log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header only if the file does not exist
        if not file_exists:
            writer.writerow(["Phase", "Duration (ms)", "CPU Usage (%)", "Additional RAM Usage (MB)"])
        # Append the data
        writer.writerow([phase, duration, cpu_usage, ram_usage])


# Function to generate the progress file path based on the network
def get_progress_file(network):
    return f'experiment_progress_{network}.csv'


def read_and_filter_csv(csv_path, network):
    """Read CSV and filter for the specific network."""
    df = pd.read_csv(csv_path)
    # Assuming 'Network' column in the CSV corresponds to the network model type
    df_filtered = df[df['Network'] == network]
    return df_filtered


def run_experiment_for_qos(model, params, ip):
    """Run the configuration for a specific QoS request and return the results."""
    # Use the passed configuration if space_configuration is provided, otherwise get it from the trial

    objectives = {}
    user_attributes = {}

    ssh_client = get_ssh_client({'host': '128.131.169.160', 'port': 22, 'username': 'pi', 'password': 'rucon2020'})

    # Measure hardware configuration change latency
    start_time = time.perf_counter_ns()
    _ = psutil.cpu_percent(interval=None)
    ram_before = psutil.virtual_memory().used / (1024 * 1024)

    setup_pi(params['cpu-freq'], params['edge-accelerator'], ssh_client)

    cpu_after = psutil.cpu_percent(interval=None)
    ram_after = psutil.virtual_memory().used / (1024 * 1024)
    end_time = time.perf_counter_ns()

    # Log the hardware configuration change metrics
    log_overhead_metrics("Hardware Configuration Change", start_time, end_time, cpu_after,
                         ram_after - ram_before)

    time.sleep(1.5)

    command = f"cd /home/pi/may_research/communication/ && source /home/pi/.virtualenvs/tensorflow/bin/activate && " \
              f"python split_computing_client.py " \
              f"--cloud_gpu {params['server-accelerator']} " \
              f"--model {model} " \
              f"--n_samples {1000} " \
              f"--splitting_point {params['layer']} " \
              f"--ip {ip} " \
              f"--tpu_mode {params['edge-accelerator']} " \
              f"--port {50051} "

    try:
        print(f"Starting evaluation run for {model} with run configuration {params}.")

        # setup power meter
        power_meter_device = GPMDevice(host="192.168.167.91")
        power_meter_device.connect()
        measurement_thread = None

        # setup SSH
        stdin, stdout, stderr = ssh_client.exec_command(command=command)
        stdout.channel.set_combine_stderr(True)
        stdin.close()

        for line in stdout:
            if "Start Experiment" in line:
                # start power measurement on edge
                measurement_thread = power_meter_device.start_power_capture(0.2)
            elif "End Experiment" in line:
                # Stop power measurement and retrieve the power data (timestamps and power values)
                power_data = power_meter_device.stop_power_capture(measurement_thread)

                # Extract timestamps and power values from the returned data
                timestamps = list(power_data.keys())  # timestamps are in seconds (Unix epoch)
                power_values = list(power_data.values())  # Power values in watts

                # Perform trapezoidal integration using sklearn's AUC
                energy_edge = auc(timestamps, power_values) / 1000  # Total energy in Joules
                user_attributes['avg energy edge (J)'] = energy_edge
            elif "Total Latency" in line:
                # extract and save avg total latency in ms
                latency_total = float(line.strip().split()[2])
                objectives['latency'] = latency_total
            elif "Cloud Latency" in line:
                # extract and save avg cloud latency in ms
                latency_server = float(line.strip().split()[2])
                user_attributes['avg latency cloud (ms)'] = latency_server
            elif "Transfer Latency" in line:
                # extract and save avg transfer latency in ms
                latency_transfer = float(line.strip().split()[2])
                user_attributes['avg latency transfer (ms)'] = latency_transfer
            elif "Edge Latency" in line:
                # extract and save avg edge latency in ms
                latency_edge = float(line.strip().split()[2])
                user_attributes['avg latency edge (ms)'] = latency_edge
            elif "Cloud Energy" in line:
                # extract and save avg cloud energy in J
                energy_cloud = float(line.strip().split()[2])
                user_attributes['avg energy cloud (J)'] = energy_cloud
                energy_total = energy_cloud + energy_edge
                objectives['energy'] = energy_total
            elif "Cloud GPU Energy" in line:
                # extract and save avg cloud gpu energy in J
                energy_cloud_gpu = float(line.strip().split()[3])
                user_attributes['avg energy cloud gpu (J)'] = energy_cloud_gpu
            elif "Tensor Size" in line:
                # extract and save tensor size in KB
                tensor_size = float(line.strip().split()[2])
                user_attributes['tensor size (KB)'] = tensor_size
            elif "Edge CPU Utilization" in line:
                # extract and save edge cpu utilization in %
                edge_cpu_utilization = float(line.strip().split()[3])
                user_attributes['utilization edge cpu (%)'] = edge_cpu_utilization
            elif "Cloud CPU Utilization" in line:
                # extract and save cloud cpu utilization in %
                cloud_cpu_utilization = float(line.strip().split()[3])
                user_attributes['utilization cloud cpu (%)'] = cloud_cpu_utilization
            elif "Cloud GPU Utilization" in line:
                # extract and save cloud gpu utilization in %
                cloud_gpu_utilization = float(line.strip().split()[3])
                user_attributes['utilization cloud gpu (%)'] = cloud_gpu_utilization
            elif "Accuracy" in line:
                # extract and save accuracy
                accuracy = float(line.strip().split()[1])
                objectives['accuracy'] = accuracy
            else:
                print(line.strip())
        # Check the exit status
        exit_status = stdout.channel.recv_exit_status()
        stdout.close()
        ssh_client.close()
        if exit_status == 0:
            print(f"Experiment finished")
            return {**objectives, **user_attributes}
        else:
            ssh_client.close()
            raise Exception(f"Experiment terminated with error: {exit_status}.")
    except Exception as e:
        if measurement_thread is not None:
            try:
                power_meter_device._stop_measurement_in_thread(measurement_thread)
                print("Measurement thread stopped due to an error.")
            except Exception as stop_err:
                print(f"Failed to stop measurement thread: {stop_err}")
        ssh_client.close()
        print(
            f"Failed to execute experiment with run configuration: {params}.")
        print(e)
        return {}
    finally:
        ssh_client.close()
        power_meter_device.disconnect()


def save_run_result(df_progress, result, network):
    """Append the result to the progress DataFrame and save."""
    # Check if the result contains valid data (e.g., check for required keys like 'latency', 'energy', etc.)
    if not result or 'latency' not in result or 'energy' not in result:
        print("Skipping failed experiment; no valid results to save.")
        return  # Skip adding a row if the result is invalid or the experiment failed

    # Load the existing progress file to ensure we are accumulating rows
    progress_file = get_progress_file(network)

    if os.path.exists(progress_file):
        df_progress_existing = pd.read_csv(progress_file)
        # Concatenate the new result with the existing progress DataFrame
        df_progress = pd.concat([df_progress_existing, pd.DataFrame([result])], ignore_index=True)
    else:
        # If no existing progress, just append the new row
        df_progress = pd.concat([df_progress, pd.DataFrame([result])], ignore_index=True)

    # Save the updated DataFrame to the CSV file
    save_progress(df_progress, network)


def extract_config_from_series(series):
    """
    Convert a pandas.Series object returned from extract_from_pareto to a dictionary format
    similar to the cloud or edge configurations.
    """
    # Assuming the series has keys that match the required configuration fields
    config = {
        'cpu-freq': series.get('cpu-freq'),
        'layer': series.get('layer'),
        'edge-accelerator': series.get('edge-accelerator'),
        'server-accelerator': series.get('server-accelerator')
    }
    return config


def execute_static_strategy(index, model, qos, strategy_name, params, df_progress, network, ip_address):
    """Execute a static strategy and save the result."""
    result = run_experiment_for_qos(model, params, ip_address)
    result.update({'index': index, 'qos': qos, 'strategy': strategy_name, **params})
    save_run_result(df_progress, result, network)


def execute_static_strategies(index, model, qos, energy_config, latency_config, df_progress, network, ip_address, pbar):
    """Execute and save the static strategies"""
    # Check if the dataframe is empty
    if df_progress.empty:
        processed_strategies = []
    else:
        # Check which strategies have already been processed
        processed_strategies = df_progress[df_progress['index'] == index]['strategy'].unique()

    required_strategies = ['cloud', 'edge', 'energy', 'latency']

    if 'cloud' in required_strategies and 'cloud' not in processed_strategies:
        cloud_params = {'cpu-freq': 1800, 'layer': 0, 'edge-accelerator': 'off', 'server-accelerator': True}
        execute_static_strategy(index, model, qos, 'cloud', cloud_params, df_progress, network, ip_address)
    pbar.update(1)
    if 'edge' in required_strategies and 'edge' not in processed_strategies:
        edge_params = {'cpu-freq': 1800, 'layer': LOCAL_COMPUTE_IDX[model],
                       'edge-accelerator': 'max' if model != 'vit' else 'off', 'server-accelerator': False}
        execute_static_strategy(index, model, qos, 'edge', edge_params, df_progress, network, ip_address)
    pbar.update(1)
    if 'energy' not in processed_strategies:
        execute_static_strategy(index, model, qos, 'energy', energy_config, df_progress, network, ip_address)
    pbar.update(1)
    if 'latency' not in processed_strategies:
        execute_static_strategy(index, model, qos, 'latency', latency_config, df_progress, network, ip_address)
    pbar.update(1)


def execute_dynamic_strategy_for_qos(index, model, qos, splinter_pareto_front, df_progress, network, ip_address, pbar,
                                     exhaustive_pareto):
    """Execute SPLINTER latency strategy dynamically and save the result."""

    # Determine the strategy name based on whether exhaustive_pareto is set
    strategy_name = 'splinter:latency_exhaustive' if exhaustive_pareto else 'splinter:latency'

    # Check if the dataframe is empty or if the 'index' column exists before filtering
    if df_progress.empty or 'index' not in df_progress.columns:
        processed_strategies = []
    else:
        # Ensure that we're checking if the 'splinter:latency' strategy has been processed
        processed_strategies = df_progress[df_progress['index'] == index]['strategy'].unique()

    if strategy_name not in processed_strategies:
        pbar.set_description(f"Running {strategy_name} strategy for QoS {qos}")

        # Measure configuration retrieval overhead for 'splinter:latency'
        start_time = time.perf_counter_ns()
        _ = psutil.cpu_percent(interval=None)
        ram_before = psutil.virtual_memory().used / (1024 * 1024)

        splinter_series = extract_from_pareto('splinter:latency', splinter_pareto_front, qos)
        splinter_config = extract_config_from_series(splinter_series)

        cpu_after = psutil.cpu_percent(interval=None)
        ram_after = psutil.virtual_memory().used / (1024 * 1024)
        end_time = time.perf_counter_ns()

        # Log the configuration retrieval metrics
        log_overhead_metrics("Configuration Retrieval", start_time, end_time, cpu_after,
                             ram_after - ram_before)

        splinter_result = run_experiment_for_qos(model, splinter_config, ip_address)
        splinter_result.update({'index': index, 'qos': qos, 'strategy': strategy_name, **splinter_config})
        save_run_result(df_progress, splinter_result, network)
    pbar.update(1)


def save_progress(df, network):
    """Save current progress to CSV for the given network."""
    progress_file = get_progress_file(network)
    df.to_csv(progress_file, index=False)


def load_progress(network):
    """Load progress from CSV if exists for the given network."""
    progress_file = get_progress_file(network)
    if os.path.exists(progress_file):
        # Load the existing progress file
        df = pd.read_csv(progress_file)
        return df
    return pd.DataFrame()


def sort_pareto_fronts(pareto_front, strategies: list):
    """
    Sort the Pareto front based on the strategies provided in the list.

    Parameters:
        pareto_front: The Pareto front data to be sorted.
        strategies (list): List of strategies (as strings) to sort by.

    Returns:
        A single sorted list if one strategy is given, otherwise a list of sorted lists.
    """
    sorted_lists = [sort_pareto_front(pareto_front, strategy) for strategy in strategies]

    # Return a single sorted list if only one strategy is provided
    if len(strategies) == 1:
        return sorted_lists[0]
    else:
        return sorted_lists


def load_pareto_front_from_study(study_name):
    """Load Pareto front from the Optuna study."""
    study = optuna.load_study(study_name=study_name, storage='sqlite:///splinter.db')
    return study.best_trials


def get_study_name(model, exhaustive_pareto):
    """Infer the study name based on the model and whether exhaustive Pareto is enabled."""
    if exhaustive_pareto:
        return f"{model}_grid_1_pruning_True"
    return f"{model}_nsga_0.2_pruning_True"


def continue_experiment(model, csv_path, exhaustive_pareto, ip_address):
    """Main function to continue the experiment."""
    # Infer the study name based on the model and exhaustive Pareto flag
    study_name = get_study_name(model, exhaustive_pareto)

    # Load progress if exists for the specific model
    df_progress = load_progress(model)

    # Read and filter the CSV
    df_filtered = read_and_filter_csv(csv_path, model)

    # Measure Pareto front loading and sorting overhead
    start_time = time.perf_counter_ns()
    _ = psutil.cpu_percent(interval=None)
    ram_before = psutil.virtual_memory().used / (1024 * 1024)  # Convert bytes to MB

    # Load the Pareto front from the Optuna study
    pareto_front = load_pareto_front_from_study(study_name)

    # Sort Pareto front for splinter
    splinter_latency_sorted = sort_pareto_fronts(pareto_front, ['splinter:latency'])

    cpu_after = psutil.cpu_percent(interval=None)
    ram_after = psutil.virtual_memory().used / (1024 * 1024)  # Convert bytes to MB
    end_time = time.perf_counter_ns()

    # Log the Pareto front loading and sorting metrics
    log_overhead_metrics("Pareto Load & Sort", start_time, end_time, cpu_after, ram_after - ram_before)

    # Sort Pareto front for static baselines
    energy_sorted, latency_sorted = sort_pareto_fronts(pareto_front, ['energy', 'latency'])

    energy_series = extract_from_pareto('energy', energy_sorted)
    energy_config = extract_config_from_series(energy_series)

    latency_series = extract_from_pareto('latency', latency_sorted)
    latency_config = extract_config_from_series(latency_series)

    # For each QoS request in the CSV, execute all missing strategies
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"Processing QoS for {model}"):
        qos = row['Latency']  # Assuming 'Latency' column exists in CSV
        index = row['Index']  # Assuming 'Index' column exists in CSV

        # Create a progress bar for each strategy within a QoS request
        strategies_count = 1 if exhaustive_pareto else 5
        with tqdm(total=strategies_count, desc=f"Processing strategies for QoS index {index}", leave=False) as pbar:
            # Check which strategies are missing for this QoS request and run them
            execute_static_strategies(index, model, qos, energy_config, latency_config, df_progress, model, ip_address,
                                      pbar)
            execute_dynamic_strategy_for_qos(index, model, qos, splinter_latency_sorted, df_progress, model, ip_address,
                                             pbar, exhaustive_pareto)

    print(f"Experiment for {model} completed or interrupted.")


def evaluate_additional_trials(ip):
    # Define the configurations with their respective number of repetitions
    configurations = [
        {'repeats': 8, 'cpu-freq': 1000, 'layer': 22, 'edge-accelerator': 'max', 'server-accelerator': False},
        {'repeats': 8, 'cpu-freq': 1400, 'layer': 22, 'edge-accelerator': 'max', 'server-accelerator': False},
        {'repeats': 9, 'cpu-freq': 1000, 'layer': 22, 'edge-accelerator': 'std', 'server-accelerator': False},
        {'repeats': 9, 'cpu-freq': 1400, 'layer': 10, 'edge-accelerator': 'max', 'server-accelerator': True},
        {'repeats': 9, 'cpu-freq': 1800, 'layer': 18, 'edge-accelerator': 'max', 'server-accelerator': True}
    ]

    # Load the progress file
    progress_file = 'additional_trials_vgg16_exhaustive.csv'
    if os.path.exists(progress_file):
        df_progress = pd.read_csv(progress_file)
    else:
        df_progress = pd.DataFrame(
            columns=['cpu-freq', 'layer', 'edge-accelerator', 'server-accelerator', 'repeat', 'results'])

    # Iterate over each configuration
    for config in tqdm(configurations, desc="Evaluating additional trials"):
        for repeat in range(1, config['repeats'] + 1):
            # Check if the specific configuration and repeat is already in the progress file
            if not df_progress[(df_progress['cpu-freq'] == config['cpu-freq']) &
                               (df_progress['layer'] == config['layer']) &
                               (df_progress['edge-accelerator'] == config['edge-accelerator']) &
                               (df_progress['server-accelerator'] == config['server-accelerator']) &
                               (df_progress['repeat'] == repeat)].empty:
                print(f"Configuration {config} with repeat {repeat} already evaluated, skipping...")
                continue

            # Run the experiment for the current configuration and repeat
            result = run_experiment_for_qos('vgg16', config, ip)  # Update IP as needed
            config_with_repeat = {**config, 'repeat': repeat, 'results': result}

            # Append result to progress DataFrame
            df_progress = pd.concat([df_progress, pd.DataFrame([config_with_repeat])], ignore_index=True)

            # Save the updated progress to file after each run
            df_progress.to_csv(progress_file, index=False)
            print(f"Progress saved for configuration: {config}, repeat: {repeat}")

    print("Evaluation of additional trials completed.")


# Add this block for command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation experiment")
    parser.add_argument('-i', '--ip', type=str, required=True, help="IP address for the cloud device")
    return parser.parse_args()


# In the main function or evaluation entry point
if __name__ == "__main__":
    args = parse_args()
    ip_address = args.ip
    #evaluate_additional_trials(args.ip)
    continue_experiment('vit', 'network_latency_50_samples.csv', False, ip_address)
