import argparse
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tflite_runtime.interpreter as tflite
from keras.applications import mobilenet_v2
from keras.applications import resnet50
from keras.applications import vgg16
from pymeas.device import GPMDevice
from pymeas.output import CsvOutput
from tqdm import tqdm

from communication import service_pb2

PATH_PREFIX = {service_pb2.VGG16: "VGG16",
               service_pb2.RESNET50: "resnet50",
               service_pb2.MOBILENETV2: "mobilenetv2"}

LOCAL_COMPUTE_IDX = {
    service_pb2.VGG16: 22,
    service_pb2.RESNET50: 40,
    service_pb2.MOBILENETV2: 75
}


def normalize_img_vgg16(img, lbl):
    img = tf.image.resize_with_pad(img, 224, 224)
    img = vgg16.preprocess_input(img)
    return img, lbl


def normalize_img_resnet50(img, lbl):
    img = tf.image.resize_with_pad(img, 224, 224)
    img = resnet50.preprocess_input(img)
    return img, lbl


def normalize_img_mobilenetv2(img, lbl):
    img = tf.image.resize_with_pad(img, 224, 224)
    img = mobilenet_v2.preprocess_input(img)
    return img, lbl


class SplitComputeClient:

    def __init__(self):
        self.head = None
        self.network = None
        self.partition_index = None
        self.input_details = None
        self.output_details = None
        self.input_scale = None
        self.input_zero_point = None
        self.accelerator = None

    def split_compute(self, image, network, accelerator):
        if (network == self.network) and (self.accelerator == accelerator):
            pass
        else:
            self.network = network

            self.accelerator = accelerator
            if self.accelerator:
                self.head = tflite.Interpreter(
                    model_path="../" + PATH_PREFIX[network] + "/models/head/" + str(
                        LOCAL_COMPUTE_IDX[network]) + "_edgetpu.tflite",
                    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
            else:
                self.head = tf.lite.Interpreter(
                    model_path="../" + PATH_PREFIX[network] + "/models/head/" + str(
                        LOCAL_COMPUTE_IDX[network]) + ".tflite")
            self.head.allocate_tensors()

            # Get input and output tensors from head network.
            self.input_details = self.head.get_input_details()[0]
            self.output_details = self.head.get_output_details()[0]

            self.input_scale, self.input_zero_point = self.input_details["quantization"]

        # convert image
        scaled_image = image / self.input_scale + self.input_zero_point
        batch_int = np.expand_dims(scaled_image, axis=0).astype(self.input_details["dtype"])

        # invoke head network
        self.head.set_tensor(self.input_details["index"], batch_int)
        self.head.invoke()
        intermediate = self.head.get_tensor(self.output_details['index'])

        # decode predictions if local only computation
        if self.network == service_pb2.VGG16:
            classes = [label_id for label_id, _, _ in vgg16.decode_predictions(intermediate, top=5)[0]]
        elif self.network == service_pb2.RESNET50:
            classes = [label_id for label_id, _, _ in resnet50.decode_predictions(intermediate, top=5)[0]]
        elif self.network == service_pb2.MOBILENETV2:
            classes = [label_id for label_id, _, _ in mobilenet_v2.decode_predictions(intermediate, top=5)[0]]

        return classes


def run(num_images, network_arg, accelerator):
    if network_arg == 'vgg16':
        network = service_pb2.VGG16
    elif network_arg == 'resnet50':
        network = service_pb2.RESNET50
    elif network_arg == 'mobilenetv2':
        network = service_pb2.MOBILENETV2

    data_dir = '/home/pi/tensorflow_datasets/downloads'
    write_dir = '/home/pi/tensorflow_datasets/extracted'
    download_config = tfds.download.DownloadConfig(
        extract_dir=os.path.join(write_dir, 'extracted'),
        manual_dir=data_dir
    )
    download_and_prepare_kwargs = {
        'download_dir': os.path.join(write_dir, 'downloaded'),
        'download_config': download_config,
    }

    validation_ds, metadata = tfds.load(
        'imagenet2012',
        split='validation',
        with_info=True,
        as_supervised=True,
        data_dir=os.path.join(write_dir, 'data'),
        download=True,
        download_and_prepare_kwargs=download_and_prepare_kwargs)

    get_label_name = metadata.features['label'].int2str

    # preprocess according to network; and skip images used for quantization
    if network == service_pb2.VGG16:
        validation_ds = validation_ds.map(normalize_img_vgg16, num_parallel_calls=tf.data.AUTOTUNE)
    elif network == service_pb2.RESNET50:
        validation_ds = validation_ds.map(normalize_img_resnet50, num_parallel_calls=tf.data.AUTOTUNE)
    elif network == service_pb2.MOBILENETV2:
        validation_ds = validation_ds.map(normalize_img_mobilenetv2, num_parallel_calls=tf.data.AUTOTUNE)
    it = iter(validation_ds)
    client = SplitComputeClient()
    power_meter_device = GPMDevice(host="192.168.167.91")
    power_meter_device.connect()
    measurement_thread = power_meter_device.start_power_capture()
    start = time.perf_counter_ns()
    # do something here
    for img_num in tqdm(range(num_images)):
        image, label = next(it)
        client.split_compute(image, network, accelerator)
    end = time.perf_counter_ns()
    power = power_meter_device.stop_power_capture(measurement_thread)
    power_meter_device.disconnect()

    if accelerator:
        pu = "tpu"
    else:
        pu = "cpu"
    data = [{'timestamp': key, 'value': value, 'total_time': (end - start) / 1000000000} for key, value in
            power.items()]

    CsvOutput.save("power_measurement" + "_" + network_arg + "_" + str(num_images) + "img_" + pu + ".csv",
                   field_names=['timestamp', 'value', 'total_time'],
                   data=data)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--accelerator', action=argparse.BooleanOptionalAction, help='Use TPU instead of CPU.',
                        default=False)
    parser.add_argument('-n', '--network', type=str, choices=['vgg16', 'resnet50', 'mobilenetv2'], default='vgg16',
                        help='The network to be used.')
    parser.add_argument('-s', '--n_samples', type=int, default=100, help='The number of samples to average over.')

    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    run(args.n_samples, args.network, args.accelerator)
