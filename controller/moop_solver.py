import argparse
import random
import sys
import time
from functools import partial
from itertools import product
from math import ceil

import optuna
import pandas as pd
from optuna import Trial
from optuna.samplers import TPESampler, NSGAIIISampler, GridSampler
from optuna.trial import TrialState
from pandas import DataFrame
from paramiko import AuthenticationException
from paramiko.client import SSHClient
from pymeas.device import GPMDevice
from sklearn.metrics import auc

LOCAL_COMPUTE_IDX = {
    "vgg16": 22,
    "resnet50": 40,
    "mobilenetv2": 75,
    "vit": 19
}

LOGGER = optuna.logging.get_logger("optuna")
LOGGER.setLevel(optuna.logging.INFO)


def calculate_search_space_size(model):
    cpu_freq_values = list(range(600, 1801, 200))
    layer_values = list(range(0, LOCAL_COMPUTE_IDX[model] + 1))

    size = 0

    for layer in layer_values:
        if model == 'vit' or layer == 0:
            edge_choices = ['off']
        else:
            edge_choices = ['off', 'std', 'max']

        if layer == LOCAL_COMPUTE_IDX[model]:
            server_choices = [False]
        else:
            server_choices = [True, False]

        # Calculate the number of unique combinations for this layer
        size += len(cpu_freq_values) * len(edge_choices) * len(server_choices)

    return size


def get_space_configuration(model: str, trial: Trial):
    cpu_freq = trial.suggest_int('cpu-freq', low=600, high=1800, step=200)

    # Suggest from the full space
    edge_accelerator = trial.suggest_categorical('edge-accelerator', ['off', 'std', 'max'])

    layer = trial.suggest_int('layer', low=0, high=LOCAL_COMPUTE_IDX[model], step=1)

    server_accelerator = trial.suggest_categorical('server-accelerator', [True, False])

    # Raise an exception to prune the trial if the configuration is invalid
    if (model == 'vit' or layer == 0) and edge_accelerator != 'off':
        raise optuna.TrialPruned()

    if layer == LOCAL_COMPUTE_IDX[model] and server_accelerator:
        raise optuna.TrialPruned()

    return {
        'cpu-freq': cpu_freq,
        'edge-accelerator': edge_accelerator,
        'layer': layer,
        'server-accelerator': server_accelerator
    }


# Function to extract trial info and parse into DataFrame
def extract_trial_info(trial):
    trial_info = {
        "trial_id": trial.number,
        "latency": trial.values[0],
        "params": trial.params,
        "attributes": trial.user_attrs
    }
    if len(trial.values) == 3:
        trial_info.update({
            "accuracy": trial.values[1],
            "energy": trial.values[2]
        })
    elif len(trial.values) == 2:
        trial_info.update({
            "energy": trial.values[1]
        })
    return trial_info


# Pre-sort the Pareto front once
def sort_pareto_front(pareto_front: list, strategy: str) -> pd.DataFrame:
    trial_data = []

    for trial in pareto_front:
        trial_data.append(extract_trial_info(trial))

    df = pd.DataFrame(trial_data)

    # Expand 'params' and 'attributes' into separate columns
    params_df = pd.json_normalize(df['params'])
    user_attr_df = pd.json_normalize(df['attributes'])
    df = df.drop(columns=['params', 'attributes']).join(params_df).join(user_attr_df)

    # Sort based on the strategy (once)
    if 'splinter' in strategy:
        if 'accuracy' in df.columns:
            df_sorted = df.sort_values(by=['energy', 'accuracy'], ascending=[True, False])
        else:
            df_sorted = df.sort_values(by='energy', ascending=True)
    elif strategy in ['energy', 'latency']:
        df_sorted = df.sort_values(by=strategy, ascending=True)

    return df_sorted


def extract_from_pareto(strategy: str, pareto_front: pd.DataFrame, qos: float = 0.0):
    """
    Expects a sorted dataframe.
    :param strategy:
    :param pareto_front:
    :param qos:
    :return:
    """
    if strategy in ['energy', 'latency']:
        return pareto_front.iloc[0]
    elif 'splinter' in strategy:
        fastest = pareto_front.iloc[0]
        most_energy_saving = pareto_front.iloc[0]
        for index, row in pareto_front.iterrows():
            if row['latency'] <= qos:
                return row
            if row['latency'] < fastest['latency']:
                fastest = row  # Update fastest
            if row['energy'] < most_energy_saving['energy']:
                most_energy_saving = row  # Update most energy saving
        if 'latency' in strategy:
            return fastest
        elif 'energy' in strategy:
            return most_energy_saving


def run_evaluation(model: str, latencies: DataFrame(), port: int, ip: str, strategy: str, pareto_front: list):
    if strategy == "edge":
        params = {'cpu-freq': 1800,
                  'layer': LOCAL_COMPUTE_IDX[model],
                  'edge-accelerator': 'max',
                  'server-accelerator': False}
    elif strategy == "cloud":
        params = {'cpu-freq': 1800,
                  'layer': 0,
                  'edge-accelerator': 'off',
                  'server-accelerator': True}
    else:
        considered_trials = extract_from_pareto(strategy, pareto_front)
        if strategy in ["energy", "latency"]:
            params = {'cpu-freq': considered_trials['cpu-freq'],
                      'layer': considered_trials['layer'],
                      'edge-accelerator': considered_trials['edge-accelerator'],
                      'server-accelerator': considered_trials['server-accelerator']}


def objective(model: str, port: int, ip: str, n_samples: int, pruning: bool, trial: Trial = None) -> tuple:
    # Use the passed configuration if space_configuration is provided, otherwise get it from the trial

    space_configuration = get_space_configuration(model, trial)

    if pruning:
        completed_trials = trial.study.get_trials(states=[optuna.trial.TrialState.COMPLETE])

        for completed_trial in completed_trials:
            if completed_trial.params == trial.params:
                raise optuna.TrialPruned()

    ssh_client = get_ssh_client({'host': '128.131.169.160', 'port': 22, 'username': 'pi', 'password': 'rucon2020'})
    setup_pi(space_configuration['cpu-freq'], space_configuration['edge-accelerator'], ssh_client)
    time.sleep(0.8)

    command = f"cd /home/pi/may_research/communication/ && source /home/pi/.virtualenvs/tensorflow/bin/activate && " \
              f"python split_computing_client.py " \
              f"--cloud_gpu {space_configuration['server-accelerator']} " \
              f"--model {model} " \
              f"--n_samples {n_samples} " \
              f"--splitting_point {space_configuration['layer']} " \
              f"--ip {ip} " \
              f"--tpu_mode {space_configuration['edge-accelerator']} " \
              f"--port {port} "

    try:
        LOGGER.info(f"Starting experiment for {model} with run configuration {space_configuration}.")

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
                energy_edge = auc(timestamps, power_values) / n_samples  # Total energy in Joules
                trial.set_user_attr('avg energy edge (J)', energy_edge)
            elif "Total Latency" in line:
                # extract and save avg total latency in ms
                latency_total = float(line.strip().split()[2])
            elif "Cloud Latency" in line:
                # extract and save avg cloud latency in ms
                latency_server = float(line.strip().split()[2])
                trial.set_user_attr('avg latency cloud (ms)', latency_server)
            elif "Transfer Latency" in line:
                # extract and save avg transfer latency in ms
                latency_transfer = float(line.strip().split()[2])
                trial.set_user_attr('avg latency transfer (ms)', latency_transfer)
            elif "Edge Latency" in line:
                # extract and save avg edge latency in ms
                latency_edge = float(line.strip().split()[2])
                trial.set_user_attr('avg latency edge (ms)', latency_edge)
            elif "Cloud Energy" in line:
                # extract and save avg cloud energy in J
                energy_cloud = float(line.strip().split()[2])
                trial.set_user_attr('avg energy cloud (J)', energy_cloud)
                energy_total = energy_cloud + energy_edge
            elif "Cloud GPU Energy" in line:
                # extract and save avg cloud gpu energy in J
                energy_cloud_gpu = float(line.strip().split()[3])
                trial.set_user_attr('avg energy cloud gpu (J)', energy_cloud_gpu)
            elif "Tensor Size" in line:
                # extract and save tensor size in KB
                tensor_size = float(line.strip().split()[2])
                trial.set_user_attr('tensor size (KB)', tensor_size)
            elif "Edge CPU Utilization" in line:
                # extract and save edge cpu utilization in %
                edge_cpu_utilization = float(line.strip().split()[3])
                trial.set_user_attr('utilization edge cpu (%)', edge_cpu_utilization)
            elif "Cloud CPU Utilization" in line:
                # extract and save cloud cpu utilization in %
                cloud_cpu_utilization = float(line.strip().split()[3])
                trial.set_user_attr('utilization cloud cpu (%)', cloud_cpu_utilization)
            elif "Cloud GPU Utilization" in line:
                # extract and save cloud gpu utilization in %
                cloud_gpu_utilization = float(line.strip().split()[3])
                trial.set_user_attr('utilization cloud gpu (%)', cloud_gpu_utilization)
            elif "Accuracy" in line:
                # extract and save accuracy
                accuracy = float(line.strip().split()[1])
            else:
                LOGGER.info(line.strip())
        # Check the exit status
        exit_status = stdout.channel.recv_exit_status()
        stdout.close()
        if exit_status == 0:
            LOGGER.info(f"Experiment finished")
            if model == 'vit':
                return latency_total, energy_total
            else:
                return latency_total, accuracy, energy_total
        else:
            ssh_client.close()
            raise Exception(f"Experiment terminated with error: {exit_status}.")
    except Exception as e:
        LOGGER.error(
            f"Failed to execute experiment with run configuration: {space_configuration}.")
        LOGGER.exception(e)
        if measurement_thread is not None:
            try:
                power_meter_device._stop_measurement_in_thread(measurement_thread)
                LOGGER.info("Measurement thread stopped due to an error.")
            except Exception as stop_err:
                LOGGER.error(f"Failed to stop measurement thread: {stop_err}")

        if model == 'vit':
            return float('nan'), float('nan')
        else:
            return float('nan'), float('nan'), float('nan')
    finally:
        ssh_client.close()
        power_meter_device.disconnect()


def setup_pi(cpu_freq: int, tpu_mode: str, ssh_client: SSHClient):
    command = f"cd /home/pi/may_research/ && source /home/pi/.virtualenvs/tensorflow/bin/activate && " \
              f"python communication/hardware.py " \
              f"--cpu_frequency {cpu_freq} " \
              f"--tpu_mode {tpu_mode}"
    try:
        LOGGER.info(f"Setting up Pi with cpu frequency {cpu_freq} and tpu mode {tpu_mode}.")
        stdin, stdout, stderr = ssh_client.exec_command(command=command)
        stdout.channel.set_combine_stderr(True)
        stdin.close()
        if stdout.channel.exit_status_ready():
            exit_status = stdout.channel.exit_status
            if exit_status != 0:
                LOGGER.error(f"Setting up Pi terminated with error: {exit_status}.")
                stdout.close()
    except Exception as e:
        LOGGER.error(
            f"Failed setting up Pi with cpu frequency {cpu_freq} and tpu mode {tpu_mode}.")
        LOGGER.exception(e)


def get_ssh_client(device: dict) -> SSHClient:
    ssh_client = SSHClient()
    ssh_client.load_system_host_keys()
    try:
        ssh_client.connect(hostname=device['host'], port=device['port'], username=device['username'],
                           password=device['password'], look_for_keys=False, allow_agent=False)
        return ssh_client
    except AuthenticationException as e:
        LOGGER.error(f"AuthenticationException occurred: {e}.")
        LOGGER.exception(e)
        ssh_client.close()
        raise e
    except Exception as e:
        LOGGER.error(f"Unexpected error occurred while connecting to host: {e}.")
        LOGGER.exception(e)
        ssh_client.close()
        raise e


def run_optimization_multi(model: str, port: int, ip: str, n_samples: int, fraction_trails: float, algorithm: str,
                           pruning: bool):
    LOGGER.info(
        f"Starting study for model {model} with fraction {fraction_trails} of configuration space using {algorithm} pruning {pruning}.")

    if algorithm == "tpe":
        sampler = TPESampler(seed=123456789)
    elif algorithm == "nsga":
        sampler = NSGAIIISampler(seed=123456789)
    elif algorithm == 'grid':
        sampler = GridSampler({'cpu-freq': list(range(600, 1801, 200)),
                               'edge-accelerator': ['off', 'std', 'max'],
                               'layer': list(range(0, LOCAL_COMPUTE_IDX[model] + 1)),
                               'server-accelerator': [True, False]})

    n_trials = ceil(calculate_search_space_size(model) * fraction_trails)

    if model == "vit":
        directions = ["minimize", "minimize"]
    else:
        directions = ["minimize", "maximize", "minimize"]

    study = optuna.create_study(study_name=f"{model}_{algorithm}_{fraction_trails}_pruning_{pruning}", sampler=sampler,
                                directions=directions,
                                storage=f"sqlite:///splinter.db",
                                load_if_exists=True)

    if model == "vit":
        study.set_metric_names(["latency (ms)", "energy (J)"])
    else:
        study.set_metric_names(["latency (ms)", "accuracy", "energy (J)"])

    while True:
        LOGGER.info(f"Number of trials executed: {len(study.trials)}")
        LOGGER.info(f"Number of complete trials: {len(study.get_trials(states=[TrialState.COMPLETE]))}.")
        LOGGER.info(f"Number of trials on the Pareto front: {len(study.best_trials)}.")

        if len(study.get_trials(states=[TrialState.COMPLETE])) < n_trials:
            missing_trials = n_trials - len(study.get_trials(states=[TrialState.COMPLETE]))
            LOGGER.info(f"Number of missing trials to reach {n_trials}: {missing_trials}.")
            study.optimize(partial(objective, model, port, ip, n_samples, pruning), show_progress_bar=True,
                           n_trials=missing_trials)
        else:
            LOGGER.info(f"{n_trials} trials threshold reached.")
            break


def get_completed_trials(study_name):
    study = optuna.load_study(study_name=study_name, storage="sqlite:///splinter.db")
    completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    return completed_trials


def generate_grid_trials(model):
    try:
        # Define the grid search space
        search_space = {
            'cpu-freq': list(range(600, 1801, 200)),
            'edge-accelerator': ['off', 'std', 'max'],
            'layer': list(range(0, LOCAL_COMPUTE_IDX[model] + 1)),
            'server-accelerator': [True, False]
        }

        # Generate all possible combinations of the parameters
        keys, values = zip(*search_space.items())
        total_trials = [dict(zip(keys, v)) for v in product(*values)]

        return total_trials
    except Exception as e:
        print(f"Error generating grid trials: {e}")
        raise


def subtract_completed_from_grid(model, completed_trials):
    try:
        # Generate all possible grid trials
        total_trials = generate_grid_trials(model)

        # Remove completed trials by comparing their parameters
        missing_trials = [trial for trial in total_trials if trial not in [t.params for t in completed_trials]]

        return missing_trials
    except Exception as e:
        print(f"Error subtracting completed trials from grid: {e}")
        raise


def dummy_objective(trial):
    # Objective function that always fails the trial
    raise optuna.TrialPruned()  # Alternatively, you can raise an exception to fail


def mark_all_waiting_as_failed(study):
    # Get all waiting trials and run the dummy objective
    waiting_trials = study.get_trials(states=[optuna.trial.TrialState.WAITING])
    print(f"Marking {len(waiting_trials)} waiting trials as failed.")

    # Run a dummy objective for each waiting trial, marking them as failed
    study.optimize(dummy_objective, n_trials=len(waiting_trials), catch=(Exception,))
    print(f"Marked all waiting trials as failed.")


def enqueue_shuffled_trials(model, study, completed_trials):
    # Subtract completed trials and shuffle the remaining ones
    missing_trials = subtract_completed_from_grid(model, completed_trials)
    random.shuffle(missing_trials)

    # Enqueue the missing shuffled trials
    for trial in missing_trials:
        study.enqueue_trial(trial)
    print(f"Enqueued {len(missing_trials)} shuffled trials.")


def clear_and_enqueue_missing_trials(model, port, ip, n_samples, pruning):
    try:
        study_name = f"{model}_grid_1_pruning_True"
        study = optuna.load_study(study_name=study_name, storage="sqlite:///splinter.db")

        # Mark all waiting trials as failed
        mark_all_waiting_as_failed(study)

        # Get completed trials
        completed_trials = get_completed_trials(study_name)

        # Enqueue shuffled missing trials
        enqueue_shuffled_trials(model, study, completed_trials)

        # Start optimization with the new queue
        # study.optimize(partial(objective, model, port, ip, n_samples, pruning), show_progress_bar=True)
    except Exception as e:
        print(f"Error enqueuing missing trials: {e}")
        raise


def enqueue_custom_trials_with_checks(model='vgg16'):
    study = optuna.load_study(study_name=f"{model}_grid_1_pruning_True", storage="sqlite:///splinter.db")
    # Step 1: Mark all currently queued trials as failed
    mark_all_waiting_as_failed(study)

    # Step 2: Get completed trials
    completed_trials = get_completed_trials(study.study_name)
    completed_params = [trial.params for trial in completed_trials]

    # Define specific trials
    specific_trials = (
        # CPU frequency evaluation
            [{'cpu-freq': f, 'edge-accelerator': 'off', 'layer': 22, 'server-accelerator': False}
             for f in range(600, 1801, 200)]
            +
            # Layer evaluation
            [{'cpu-freq': 1800, 'edge-accelerator': 'off', 'layer': 0, 'server-accelerator': True}] +
            [{'cpu-freq': 1800, 'edge-accelerator': 'max', 'layer': l, 'server-accelerator': True}
             for l in range(1, 22)]
            +
            # TPU evaluation
            [{'cpu-freq': 1800, 'edge-accelerator': e, 'layer': 22, 'server-accelerator': False}
             for e in ['off', 'std', 'max']]
            +
            # Accuracy evaluation
            [{'cpu-freq': 1800, 'edge-accelerator': 'off', 'layer': l, 'server-accelerator': True}
             for l in range(1, 22)]
            +
            # Server accelerator evaluation
            [{'cpu-freq': 1800, 'edge-accelerator': 'off', 'layer': 0, 'server-accelerator': b}
             for b in [False, True]]
    )

    # Step 3: Enqueue specific trials if not completed yet
    enqueued_trials = 0
    for trial in specific_trials:
        if trial not in completed_params:
            for _ in range(2):  # Adjust to 3 if you need three runs
                study.enqueue_trial(trial)
            enqueued_trials += 1

    print(f"{enqueued_trials} specific trials enqueued.")

    # Step 4: Enqueue remaining missing trials
    remaining_trials = subtract_completed_from_grid(model, completed_trials)
    # Remove trials that are already queued or completed
    remaining_trials = [trial for trial in remaining_trials if trial not in completed_params]
    random.shuffle(remaining_trials)

    for trial in remaining_trials:
        study.enqueue_trial(trial)

    print(f"{len(remaining_trials)} additional trials enqueued randomly.")


def main(args: argparse.Namespace):
    #run_optimization_multi('vgg16', args.port, args.ip, 1000, 1, 'grid', True)
    run_optimization_multi('resnet50', args.port, args.ip, 1000, 0.2, 'nsga', True)
    run_optimization_multi('mobilenetv2', args.port, args.ip, 1000, 0.2, 'nsga', True)

    return 0


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', type=str, choices=['vgg16', 'resnet50', 'mobilenetv2', 'vit'],
                        default='vgg16',
                        help='The neural network model to be used.')
    parser.add_argument('-n', '--n_samples', type=int, default=1000, help='The number of samples to average over.')
    parser.add_argument('-t', '--n_trials', type=int, default=100, help='The number of trials.')
    parser.add_argument('-p', '--port', type=int, default=50051, help='The port to connect to.')
    parser.add_argument('-i', '--ip', type=str, default='192.168.167.81', help='The server IP address to connect to.')
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(read_args()))
