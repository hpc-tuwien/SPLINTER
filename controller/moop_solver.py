from optuna import Trial


def get_space_configuration(trial: Trial):
    param = {
        'cpu-freq': trial.suggest_int('cpu-freq', low=600, high=1800, step=200),
        'layer': trial.suggest_int('layer', low=0, high=22, step=1),
    }
    # don't activate edge TPU for cloud computing
    if param['layer'] == 0:
        param['edge-accelerator'] = 'off'
    else:
        param['edge-accelerator'] = trial.suggest_categorical('edge-accelerator', ['off', 'std', 'max'])
    # don't consider cloud processor in edge computing
    if param['layer'] != 22:
        param['server-accelerator']: trial.suggest_categorical('gpu', [True, False])
    return param
