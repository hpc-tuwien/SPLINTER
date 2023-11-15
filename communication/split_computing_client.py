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

PATH_PREFIX = {service_pb2.VGG16: "VGG16",
               service_pb2.RESNET50: "resnet50",
               service_pb2.MOBILENETV2: "mobilenetv2"}
LOCAL_COMPUTE_IDX = {
    service_pb2.VGG16: 22,
    service_pb2.RESNET50: 40,
    service_pb2.MOBILENETV2: 75
}
MAX_MESSAGE_LENGTH = 12845090


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
        self.output_scale = None
        self.output_zero_point = None
        self.accelerator = None

    def split_compute(self, stub, image, network, partition_index, accelerator):
        start_counter_ns = time.perf_counter_ns()

        # check if there is any local computation
        if partition_index == 0:
            self.head = None
            self.network = network
            self.partition_index = partition_index
            self.input_details = None
            self.output_details = None
            self.input_scale = None
            self.input_zero_point = None
            self.output_scale = None
            self.output_zero_point = None
            self.accelerator = accelerator
            intermediate = image
        else:
            if (network == self.network) and (partition_index == self.partition_index) and (
                    self.accelerator == accelerator):
                pass
            else:
                self.network = network
                self.partition_index = partition_index
                self.accelerator = accelerator
                if self.accelerator:
                    self.head = tflite.Interpreter(
                        model_path="../" + PATH_PREFIX[network] + "/models/head/" + str(
                            partition_index) + "_edgetpu.tflite",
                        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
                else:
                    self.head = tf.lite.Interpreter(
                        model_path="../" + PATH_PREFIX[network] + "/models/head/" + str(partition_index) + ".tflite")
                self.head.allocate_tensors()

                # Get input and output tensors from head network.
                self.input_details = self.head.get_input_details()[0]
                self.output_details = self.head.get_output_details()[0]

                self.input_scale, self.input_zero_point = self.input_details["quantization"]
                self.output_scale, self.output_zero_point = self.output_details["quantization"]

            # convert image
            scaled_image = image / self.input_scale + self.input_zero_point
            batch_int = np.expand_dims(scaled_image, axis=0).astype(self.input_details["dtype"])

            # invoke head network
            self.head.set_tensor(self.input_details["index"], batch_int)
            self.head.invoke()
            intermediate = self.head.get_tensor(self.output_details['index'])

        # decode predictions if local only computation
        if partition_index == LOCAL_COMPUTE_IDX[self.network]:
            if self.network == service_pb2.VGG16:
                classes = [label_id for label_id, _, _ in vgg16.decode_predictions(intermediate, top=5)[0]]
            elif self.network == service_pb2.RESNET50:
                classes = [label_id for label_id, _, _ in resnet50.decode_predictions(intermediate, top=5)[0]]
            elif self.network == service_pb2.MOBILENETV2:
                classes = [label_id for label_id, _, _ in mobilenet_v2.decode_predictions(intermediate, top=5)[0]]
            end_counter_ns = time.perf_counter_ns()
            network_time = 0
            server_time = 0
        else:
            if self.partition_index == 0:
                intermediate_scaled = np.expand_dims(intermediate, axis=0)
            else:
                # rescale tensor
                intermediate_scaled = ((intermediate - self.output_zero_point) * self.output_scale).astype("float32")
            serialized_tensor = tf.io.serialize_tensor(intermediate_scaled).numpy()
            end_counter_ns = time.perf_counter_ns()
            start_network_time = time.perf_counter_ns()
            resp = stub.SplitCompute(service_pb2.SplitRequest(network=self.network,
                                                              partition_index=self.partition_index,
                                                              tensor=serialized_tensor))
            end_network_time = time.perf_counter_ns()
            classes = resp.classes
            server_time = resp.server_time
            network_time = end_network_time - start_network_time
            network_time = network_time - server_time
        client_time = end_counter_ns - start_counter_ns
        total_time = client_time + server_time + network_time
        return classes, client_time / 1000000, server_time / 1000000, network_time / 1000000, total_time / 1000000


def run(num_images, network_arg, accelerator):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel("192.168.167.81:50051", [
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
                                                                                                        accelerator)
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
    parser.add_argument('-a', '--accelerator', action=argparse.BooleanOptionalAction, help='Use TPU instead of CPU.',
                        default=False)
    parser.add_argument('-n', '--network', type=str, choices=['vgg16', 'resnet50', 'mobilenetv2'], default='vgg16',
                        help='The network to be used.')
    parser.add_argument('-s', '--n_samples', type=int, default=100, help='The number of samples to average over.')

    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    run(args.n_samples, args.network, args.accelerator)
