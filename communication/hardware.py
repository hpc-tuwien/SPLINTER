import argparse
import subprocess
import sys
import time


def plug_out_tpu():
    console_output = subprocess.run(['sudo', 'uhubctl'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    # disconnect TPU if connected
    if 'U0 enable connect' in console_output:
        subprocess.run(['sudo', 'uhubctl', '-l', '2', '-a', '0'], stdout=subprocess.PIPE)
        console_output = subprocess.run(['sudo', 'uhubctl'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        if 'U0 enable connect' in console_output:
            raise Exception("Error: Could not unplug TPU.")


def plug_in_tpu():
    console_output = subprocess.run(['sudo', 'uhubctl'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    # disconnect TPU if connected
    if 'power 5gbps U0 enable connect [1a6e:089a]' not in console_output:
        subprocess.run(['sudo', 'uhubctl', '-l', '2', '-a', '1'], stdout=subprocess.PIPE)
        time.sleep(0.2)
        console_output = subprocess.run(['sudo', 'uhubctl'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        if 'U0 enable connect' not in console_output:
            raise Exception("Error: Could not plug in TPU.")


def setup_hardware(tpu_mode: str, cpu_frequency: str):
    # CPU settings
    console_output = subprocess.run(['cpufreq-info', '-p'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    # set userspace governor to allow direct setting of CPU frequency
    if 'userspace' not in console_output:
        subprocess.run(['sudo', 'cpufreq-set', '-g', 'userspace'], stdout=subprocess.PIPE)

    console_output = subprocess.run(['cat', '/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'],
                                    stdout=subprocess.PIPE).stdout.decode('utf-8')
    if cpu_frequency not in console_output:
        subprocess.run(['sudo', 'cpufreq-set', '-f', cpu_frequency + '000'], stdout=subprocess.PIPE)
        console_output = subprocess.run(['cat', '/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'],
                                        stdout=subprocess.PIPE).stdout.decode('utf-8')
        if cpu_frequency not in console_output:
            raise Exception("Error: Could not set CPU frequency.")

    # TPU settings
    if tpu_mode == 'off':
        plug_out_tpu()
    else:
        console_output = subprocess.run(['apt', 'list', '--installed', 'libedgetpu*'],
                                        stdout=subprocess.PIPE).stdout.decode('utf-8')
        # install correct apt package if not already there
        if tpu_mode not in console_output:
            subprocess.run(['sudo', 'apt', '-y', 'install', 'libedgetpu1-' + tpu_mode],
                           stdout=subprocess.PIPE)
            console_output = subprocess.run(['apt', 'list', '--installed', 'libedgetpu*'],
                                            stdout=subprocess.PIPE).stdout.decode('utf-8')
            if tpu_mode not in console_output:
                raise Exception("Error: Could not install libedgetpu1-" + tpu_mode + ".")
            # eject device if already connected
            plug_out_tpu()
            # plug device in
            plug_in_tpu()
        else:
            plug_in_tpu()


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--tpu_mode', type=str, choices=['off', 'std', 'max'], default='std',
                        help='The TPU mode to be used.')
    # from 600 MHz to 1800 MHz in 200 MHz steps
    parser.add_argument('-c', '--cpu_frequency', type=str, choices=[str(x) for x in range(600, 2000, 200)],
                        default='1800', help='The CPU frequency in MHz to be used.')
    return parser.parse_args()


def main(args: argparse.Namespace):
    print("Setting up hardware ...")
    setup_hardware(args.tpu_mode, args.cpu_frequency)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(read_args()))
