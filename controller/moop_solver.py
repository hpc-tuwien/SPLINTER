import random
import sys
from functools import partial

import optuna
import pandas as pd
from optuna import Trial
from optuna.samplers import TPESampler
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
    param = {
        'cpu-freq': trial.suggest_int('cpu-freq', low=600, high=1800, step=200),
        'layer': trial.suggest_int('layer', low=0, high=LOCAL_COMPUTE_IDX[model], step=1)
    }
    # TPU is only available for non ViT models
    # always unplug edge TPU for cloud computing
    if model == 'vit' or param['layer'] == 0:
        param['edge-accelerator'] = 'off'
    else:
        param['edge-accelerator'] = trial.suggest_categorical('edge-accelerator', ['off', 'std', 'max'])

    # don't consider cloud acceleration in edge computing
    if param['layer'] != LOCAL_COMPUTE_IDX[model]:
        param['server-accelerator'] = trial.suggest_categorical('server-accelerator', [True, False])
    else:
        param['server-accelerator'] = False

    return param


def run_configuration_test(model: str, space_configuration: dict) -> tuple:
    ssh_client = get_ssh_client({'host': '192.168.167.140', 'port': 22, 'username': 'pi', 'password': 'rucon2020'})
    setup_pi(space_configuration['cpu-freq'], space_configuration['edge-accelerator'], ssh_client)

    command = f"cd /home/pi/may_research/communication/ && source /home/pi/.virtualenvs/tensorflow/bin/activate && " \
              f"python split_computing_client.py " \
              f"--cloud_gpu {space_configuration['server-accelerator']} " \
              f"--model {model} " \
              f"--n_samples {100} " \
              f"--splitting_point {space_configuration['layer']} " \
              f"--ip {'chifflot-5.lille.grid5000.fr '} " \
              f"--tpu_mode {space_configuration['edge-accelerator']} "

    try:
        LOGGER.info(f"Starting experiment with run configuration {space_configuration}.")

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
                # extract and save latency in s
                latency = int(line.strip().split()[1]) / 1_000_000_000
                # calculate energy in joule by using the median power consumption
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
        if exit_status == 0:
            LOGGER.info(f"Experiment finished")
        else:
            raise Exception(f"Experiment terminated with error: {exit_status}.")
    except Exception as e:
        LOGGER.error(f"Failed to execute energy consumption experiment with run configuration: {space_configuration}.")
        LOGGER.exception(e)

    if model == 'vit':
        return latency, energy
    else:
        return latency, accuracy, energy


def run_configuration(model: str, trial: Trial) -> tuple:
    ssh_client = get_ssh_client({'host': '192.168.167.140', 'port': 22, 'username': 'pi', 'password': 'rucon2020'})
    space_configuration = get_space_configuration(model, trial)

    setup_pi(space_configuration['cpu-freq'], space_configuration['edge-accelerator'], ssh_client)

    latency = random.randrange(100)
    accuracy = random.randrange(100)
    energy = random.randrange(100)
    if model == 'vit':
        return latency, energy
    else:
        return latency, accuracy, energy


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
    ssh_client.set_log_channel('optuna')
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


def objective(model: str, trial: Trial):
    latency, accuracy, energy = run_configuration(model, trial)
    return latency, accuracy, energy


def run_optimization(model: str):
    study = optuna.create_study(study_name=model, sampler=TPESampler(seed=11809922),
                                directions=["minimize", "maximize", "minimize"])
    study.optimize(partial(objective, model), n_trials=30, show_progress_bar=True)

    print("Number of finished trials: ", len(study.trials))


def main():
    latency, accuracy, energy = run_configuration_test('vgg16',
                                                       {'cpu-freq': 1800, 'layer': 5, 'edge-accelerator': 'off',
                                                        'server-accelerator': True})
    print(f"Latency: {latency}s")
    print(f"Accuracy: {accuracy}")
    print(f"Energy: {energy}J")
    return 0


if __name__ == '__main__':
    sys.exit(main())
