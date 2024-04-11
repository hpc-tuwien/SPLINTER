import random
from functools import partial

import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from paramiko import SSHClient, AuthenticationException

LOCAL_COMPUTE_IDX = {
    "vgg16": 22,
    "resnet50": 40,
    "mobilenetv2": 75,
    "vit": 19
}

LOGGER = optuna.logging.get_logger("optuna")
LOGGER.setLevel(optuna.logging.INFO)


class MoopSolver:

    def __init__(self):
        self.client = SSHClient()

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

    def run_configuration(model: str, trial: Trial):
        latency = random.randrange(100)
        accuracy = random.randrange(100)
        energy = random.randrange(100)

        # TODO if not local compute index then run inference of 1 image to load model on server
        # TODO call script on Pi
        if model == 'vit':
            return latency, energy
        else:
            return latency, accuracy, energy

    def setup_pi(self, cpu_freq: int, tpu_mode: str):
        command = f"cd /home/pi/may_research/ && source /home/pi/.virtualenvs/tensorflow/bin/activate && " \
                  f"python communication/hardware.py " \
                  f"--cpu_frequency {cpu_freq} " \
                  f"--tpu_mode {tpu_mode}"
        try:
            LOGGER.info(f"Setting up Pi with cpu frequency {cpu_freq} and tpu mode {tpu_mode}.")
            stdin, stdout, stderr = self.client.exec_command(command=command)
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

    def init_ssh_client(self, device: dict) -> None:
        self.client.load_system_host_keys()
        self.client.set_log_channel('optuna')
        try:
            self.client.connect(hostname=device['host'], port=device['port'], username=device['username'],
                                password=device['password'], look_for_keys=False, allow_agent=False)
        except AuthenticationException as e:
            LOGGER.error(f"AuthenticationException occurred: {e}.")
            LOGGER.exception(e)
            self.client.close()
            raise e
        except Exception as e:
            LOGGER.error(f"Unexpected error occurred while connecting to host: {e}.")
            LOGGER.exception(e)
            self.client.close()
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


if __name__ == '__main__':
    solver = MoopSolver()
    solver.init_ssh_client({'host': '192.168.167.140', 'port': 22, 'username': 'pi', 'password': 'rucon2020'})
    solver.setup_pi(600, 'off')
