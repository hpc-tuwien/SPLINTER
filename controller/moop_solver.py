import argparse
import sys
import time
from functools import partial
from math import ceil

import numpy
import optuna
import pandas as pd
from optuna import Trial
from optuna.samplers import TPESampler, NSGAIIISampler, BruteForceSampler
from optuna.trial import TrialState
from paramiko import AuthenticationException
from paramiko.client import SSHClient
from pymeas.device import GPMDevice

LOCAL_COMPUTE_IDX = {
    "vgg16": 22,
    "resnet50": 40,
    "mobilenetv2": 75,
    "vit": 19
}

LOGGER = optuna.logging.get_logger("optuna")
LOGGER.setLevel(optuna.logging.INFO)


def get_space_configuration(model: str, trial: Trial):
    if model == 'vit':
        edge_choices = ['off']
    else:
        edge_choices = ['off', 'std', 'max']
    return {'cpu-freq': trial.suggest_int('cpu-freq', low=600, high=1800, step=200),
            'layer': trial.suggest_int('layer', low=0, high=LOCAL_COMPUTE_IDX[model], step=1),
            'edge-accelerator': trial.suggest_categorical('edge-accelerator', edge_choices),
            'server-accelerator': trial.suggest_categorical('server-accelerator', [True, False])}


def alternative_objective(model: str, port: int, ip: str, n_samples: int, weights: list, trial: Trial):
    lat_min = 17.695055719
    lat_max = 251.948823873
    acc_min = 59
    acc_max = 60
    e_min = 87.9558916019154
    e_max = 961.8902197823395

    latency, accuracy, energy = objective(model, port, ip, n_samples, trial)
    trial.set_user_attr('latency', latency)
    trial.set_user_attr('accuracy', accuracy)
    trial.set_user_attr('energy', energy)

    # min max scaling
    latency = (latency - lat_min) / (lat_max - lat_min)
    accuracy = (accuracy - acc_min) / (acc_max - acc_min)
    energy = (energy - e_min) / (e_max - e_min)
    # one minimization measure with main focus on latency
    return weights[0] * latency - weights[1] * accuracy + weights[2] * energy


def objective(model: str, port: int, ip: str, n_samples: int, pruning: bool, trial: Trial) -> tuple:
    space_configuration = get_space_configuration(model, trial)
    if pruning:
        completed_trials = trial.study.get_trials(states=[optuna.trial.TrialState.COMPLETE])

        for completed_trial in completed_trials:
            if completed_trial.params == trial.params:
                raise optuna.TrialPruned()

    ssh_client = get_ssh_client({'host': '128.131.169.160', 'port': 22, 'username': 'pi', 'password': 'rucon2020'})
    setup_pi(space_configuration['cpu-freq'], space_configuration['edge-accelerator'], ssh_client)
    time.sleep(0.3)

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

        # setup SSH
        stdin, stdout, stderr = ssh_client.exec_command(command=command)
        stdout.channel.set_combine_stderr(True)
        stdin.close()

        for line in stdout:
            if "Init done" in line:
                # start power measurement
                measurement_thread = power_meter_device.start_power_capture(0.005)
            elif "latency" in line:
                # stop power measurement
                power = power_meter_device.stop_power_capture(measurement_thread)
                power_meter_device.disconnect()
                # extract and save avg latency in ms
                latency = float(line.strip().split()[1]) / 1_000_000
                # calculate energy in mJ by using the median power consumption
                df = pd.DataFrame(power.items(), columns=['timestamp', 'value'])
                energy = df['value'].quantile(0.5) * latency
            elif "accuracy" in line:
                # extract and save accuracy
                accuracy = float(line.strip().split()[1])
            else:
                LOGGER.info(line.strip())
        # Check the exit status
        exit_status = stdout.channel.recv_exit_status()
        stdout.close()
        ssh_client.close()
        if exit_status == 0:
            LOGGER.info(f"Experiment finished")
            if model == 'vit':
                return latency, energy
            else:
                return latency, accuracy, energy
        else:
            ssh_client.close()
            raise Exception(f"Experiment terminated with error: {exit_status}.")
    except Exception as e:
        ssh_client.close()
        LOGGER.error(f"Failed to execute energy consumption experiment with run configuration: {space_configuration}.")
        LOGGER.exception(e)
        if model == 'vit':
            return float('nan'), float('nan')
        else:
            return float('nan'), float('nan'), float('nan')


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


def setup_grid5k():
    pass


def run_optimization_single(model: str, port: int, ip: str, n_samples: int, n_trials: int, weights: list):
    LOGGER.info(f"Starting study for model {model}.")
    objectives = ['lat', 'acc', 'eng']
    study = optuna.create_study(study_name=f"{model}_{objectives[numpy.argmax(weights)]}_{n_trials}",
                                sampler=TPESampler(seed=11809922),
                                directions=["minimize"], storage="sqlite:///bigstudy.db",
                                load_if_exists=True)
    while True:
        LOGGER.info(f"Number of trials executed: {len(study.trials)}")
        LOGGER.info(f"Number of complete trials: {len(study.get_trials(states=[TrialState.COMPLETE]))}.")
        LOGGER.info(f"Number of trials on the Pareto front: {len(study.best_trials)}.")

        if len(study.get_trials(states=[TrialState.COMPLETE])) < n_trials:
            missing_trials = n_trials - len(study.get_trials(states=[TrialState.COMPLETE]))
            LOGGER.info(f"Number of missing trials to reach {n_trials}: {missing_trials}.")
            study.optimize(partial(alternative_objective, model, port, ip, n_samples, weights), n_trials=missing_trials,
                           show_progress_bar=True)
        else:
            LOGGER.info(f"{n_trials} trials threshold reached.")
            break


def run_optimization_multi(model: str, port: int, ip: str, n_samples: int, fraction_trails: float, algorithm: str,
                           pruning: bool):
    LOGGER.info(
        f"Starting study for model {model} with fraction {fraction_trails} of configuration space using {algorithm} pruning {pruning}.")

    if algorithm == "tpe":
        sampler = TPESampler(seed=123456789)
    elif algorithm == "nsga":
        sampler = NSGAIIISampler(seed=123456789)
    elif algorithm == 'grid':
        sampler = BruteForceSampler()

    if model == 'vit':
        n_trials = ceil((7 * 2 * (LOCAL_COMPUTE_IDX[model] + 1)) * fraction_trails)
    else:
        n_trials = ceil((7 * 3 * 2 * (LOCAL_COMPUTE_IDX[model] + 1)) * fraction_trails)

    if model == "vit":
        directions = ["minimize", "minimize"]
    else:
        directions = ["minimize", "maximize", "minimize"]

    study = optuna.create_study(study_name=f"{model}_{algorithm}_{fraction_trails}_pruning_{pruning}", sampler=sampler,
                                directions=directions,
                                storage=f"sqlite:///exhaustive.db",
                                load_if_exists=True)

    if model == "vit":
        study.set_metric_names(["latency in ms", "energy in mJ"])
    else:
        study.set_metric_names(["latency in ms", "accuracy", "energy in mJ"])

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


def main(args: argparse.Namespace):
    run_optimization_multi('vgg16', args.port, args.ip, 1000, 0.2, 'nsga', True)
    run_optimization_multi('resnet50', args.port, args.ip, 1000, 0.2, 'nsga', True)
    run_optimization_multi('mobilenetv2', args.port, args.ip, 1000, 0.2, 'nsga', True)
    run_optimization_multi('vit', args.port, args.ip, 1000, 0.2, 'nsga', True)
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
