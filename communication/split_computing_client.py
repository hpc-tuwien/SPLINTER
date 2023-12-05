import argparse
import os
import time

import grpc
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tflite_runtime.interpreter as tflite
from keras.applications import mobilenet_v2
from keras.applications import resnet50
from keras.applications import vgg16
from tqdm import tqdm

from communication import service_pb2_grpc, service_pb2
from communication.hardware import setup_hardware

PATH_PREFIX = {service_pb2.VGG16: "VGG16",
               service_pb2.RESNET50: "resnet50",
               service_pb2.MOBILENETV2: "mobilenetv2"}
LOCAL_COMPUTE_IDX = {
    service_pb2.VGG16: 22,
    service_pb2.RESNET50: 40,
    service_pb2.MOBILENETV2: 75
}
MAX_MESSAGE_LENGTH = 3211308


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
        self.accelerator = None

    def load_head_network(self):
        # cloud computing
        if self.partition_index == 0:
            self.head = None
        else:
            if self.accelerator:
                self.head = tflite.Interpreter(
                    model_path="../" + PATH_PREFIX[self.network] + "/models/head/" + str(
                        self.partition_index) + "_edgetpu.tflite",
                    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
            else:
                self.head = tflite.Interpreter(
                    model_path="../" + PATH_PREFIX[self.network] + "/models/head/" + str(
                        self.partition_index) + ".tflite")
            self.head.allocate_tensors()

    def split_compute(self, stub, image, network, partition_index, accelerator, cloud_accelerator):
        start_client_time = time.perf_counter_ns()
        # load head network
        if (network != self.network) or (partition_index != self.partition_index) or (accelerator != self.accelerator):
            self.network = network
            self.partition_index = partition_index
            self.accelerator = accelerator
            self.load_head_network()

        # cloud only computing
        if partition_index == 0:
            intermediate = np.expand_dims(image, axis=0)
        else:
            # scale input for head network
            input_details = self.head.get_input_details()[0]
            input_scale = input_details["quantization_parameters"]["scales"][0]
            input_zero_point = input_details["quantization_parameters"]["zero_points"][0]
            scaled_image = image / input_scale + input_zero_point
            batch_int = np.expand_dims(scaled_image, axis=0).astype(input_details["dtype"])

            # invoke head network
            self.head.set_tensor(input_details["index"], batch_int)
            output_details = self.head.get_output_details()[0]
            self.head.invoke()
            intermediate = self.head.get_tensor(output_details['index'])

            # edge only computation
            if self.partition_index == LOCAL_COMPUTE_IDX[self.network]:
                if self.network == service_pb2.VGG16:
                    classes = [label_id for label_id, _, _ in vgg16.decode_predictions(intermediate, top=5)[0]]
                elif self.network == service_pb2.RESNET50:
                    classes = [label_id for label_id, _, _ in resnet50.decode_predictions(intermediate, top=5)[0]]
                elif self.network == service_pb2.MOBILENETV2:
                    classes = [label_id for label_id, _, _ in mobilenet_v2.decode_predictions(intermediate, top=5)[0]]
                end_client_time = time.perf_counter_ns()
                client_time = end_client_time - start_client_time
                network_time = 0
                server_time = 0
                total_time = client_time + server_time + network_time
                return classes, client_time / 1000000, server_time / 1000000, network_time / 1000000, total_time / 1000000

        # send to cloud
        serialized_tensor = tf.io.serialize_tensor(intermediate).numpy()
        req = service_pb2.SplitRequest(network=self.network,
                                       partition_index=self.partition_index,
                                       tensor=serialized_tensor,
                                       accelerator=cloud_accelerator)
        # set scaling if there was a head network
        if partition_index != 0:
            output_details = self.head.get_output_details()[0]
            req.scale = output_details["quantization_parameters"]["scales"][0]
            req.zero_point = output_details["quantization_parameters"]["zero_points"][0]
        end_client_time = time.perf_counter_ns()
        start_network_time = time.perf_counter_ns()
        resp = stub.SplitCompute(req)
        end_network_time = time.perf_counter_ns()
        network_time = end_network_time - start_network_time
        server_time = resp.server_time
        network_time = network_time - server_time
        client_time = end_client_time - start_client_time
        total_time = client_time + server_time + network_time
        return resp.classes, client_time / 1000000, server_time / 1000000, network_time / 1000000, total_time / 1000000


def run(num_images: int, network_arg, accelerator: bool, cloud_accelerator: bool, host: str, port: int):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel(host + ":" + str(port), [
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]) as channel:

        stub = service_pb2_grpc.SplitServiceStub(channel)
        client = SplitComputeClient()
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
            validation_ds = validation_ds.map(normalize_img_vgg16, num_parallel_calls=tf.data.AUTOTUNE).skip(100)
        elif network == service_pb2.RESNET50:
            validation_ds = validation_ds.map(normalize_img_resnet50, num_parallel_calls=tf.data.AUTOTUNE).skip(100)
        elif network == service_pb2.MOBILENETV2:
            validation_ds = validation_ds.map(normalize_img_mobilenetv2, num_parallel_calls=tf.data.AUTOTUNE).skip(100)
        accuracies = pd.DataFrame(range(LOCAL_COMPUTE_IDX[network] + 1), columns=['layer'])
        latencies = pd.DataFrame()
        top1 = list()
        top5 = list()
        layer = list()
        img = list()
        client_timings = list()
        server_timings = list()
        network_timings = list()
        total_timings = list()

        for partition_index in tqdm(range(LOCAL_COMPUTE_IDX[network] + 1)):
            it = iter(validation_ds)
            top1_cnt = 0
            top5_cnt = 0

            for img_num in range(num_images):
                image, label = next(it)
                pred_classes, client_time, server_time, network_time, total_time = client.split_compute(stub, image,
                                                                                                        network,
                                                                                                        partition_index,
                                                                                                        accelerator,
                                                                                                        cloud_accelerator)
                layer.append(partition_index)
                img.append(img_num)
                client_timings.append(client_time)
                server_timings.append(server_time)
                network_timings.append(network_time)
                total_timings.append(total_time)
                if get_label_name(label) in pred_classes:
                    top5_cnt += 1
                if get_label_name(label) == pred_classes[0]:
                    top1_cnt += 1
            top1.append(top1_cnt / num_images)
            top5.append(top5_cnt / num_images)
        accuracies['top1'] = top1
        accuracies['top5'] = top5
        latencies['layer'] = layer
        latencies['img'] = img
        latencies['client'] = client_timings
        latencies['server'] = server_timings
        latencies['network'] = network_timings
        latencies['total'] = total_timings

        if accelerator:
            pu = "tpu"
        else:
            pu = "cpu"
        accuracies.to_csv("../" + network_arg + "_" + pu + "_accuracies.csv", index=False)
        latencies.to_csv("../" + network_arg + "_" + pu + "_latencies.csv", index=False)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--cloud_gpu', action=argparse.BooleanOptionalAction, help='Use cloud accelerator (GPU).',
                        default=False)
    parser.add_argument('-n', '--network', type=str, choices=['vgg16', 'resnet50', 'mobilenetv2'], default='vgg16',
                        help='The network to be used.')
    parser.add_argument('-s', '--n_samples', type=int, default=100, help='The number of samples to average over.')
    parser.add_argument('-t', '--tpu_mode', type=str, choices=['off', 'std', 'max'], default='std',
                        help='The TPU mode to be used.')
    parser.add_argument('-p', '--port', type=int, default=50051, help='The port to connect to.')
    parser.add_argument('-i', '--ip', type=str, default='192.168.167.81', help='The server IP address to connect to.')
    # from 600 MHz to 1800 MHz in 200 MHz steps
    parser.add_argument('-c', '--cpu_frequency', type=str, choices=[str(x) for x in range(600, 2000, 200)],
                        default='1500', help='The CPU frequency in MHz to be used.')

    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    print("Setting up hardware ...")
    setup_hardware(args.tpu_mode, args.cpu_frequency)
    print("Starting experiment.")
    run(args.n_samples, args.network, args.tpu_mode != 'off', args.cloud_gpu, args.ip, args.port)
