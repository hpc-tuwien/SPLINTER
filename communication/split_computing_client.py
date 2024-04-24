import argparse
import os
import sys
import time

import grpc
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tflite_runtime.interpreter as tflite
from keras.src.applications import imagenet_utils

import service_pb2
import service_pb2_grpc

PATH_PREFIX = {service_pb2.VGG16: "VGG16",
               service_pb2.RESNET50: "resnet50",
               service_pb2.MOBILENETV2: "mobilenetv2",
               service_pb2.VISIONTRANSFORMER: "ViT"}
LOCAL_COMPUTE_IDX = {
    service_pb2.VGG16: 22,
    service_pb2.RESNET50: 40,
    service_pb2.MOBILENETV2: 75,
    service_pb2.VISIONTRANSFORMER: 19
}
MAX_MESSAGE_SEND_LENGTH = 3211308
MAX_MESSAGE_RECEIVE_LENGTH = 11


def normalize_img(img, lbl, mode):
    """Normalizes images: `uint8` -> `float32`."""
    img = tf.image.resize_with_pad(img, 224, 224)
    img = imagenet_utils.preprocess_input(img, data_format=None, mode=mode)
    return img, lbl


def normalize_img_caffe(img, lbl):
    # VGG16, ResNet50
    return normalize_img(img, lbl, "caffe")


def normalize_img_tf(img, lbl):
    # MobileNetV2, ViT
    return normalize_img(img, lbl, "tf")


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
            if self.network != service_pb2.VISIONTRANSFORMER:
                input_scale = input_details["quantization_parameters"]["scales"][0]
                input_zero_point = input_details["quantization_parameters"]["zero_points"][0]
                scaled_image = image / input_scale + input_zero_point
                batch_int = np.expand_dims(scaled_image, axis=0).astype(input_details["dtype"])
            else:
                batch_int = np.expand_dims(image, axis=0)

            # invoke head network
            self.head.set_tensor(input_details["index"], batch_int)
            output_details = self.head.get_output_details()[0]
            self.head.invoke()
            intermediate = self.head.get_tensor(output_details['index'])

            # edge only computation
            if self.partition_index == LOCAL_COMPUTE_IDX[self.network]:
                label = imagenet_utils.decode_predictions(intermediate, top=1)[0][0][0]
                return label

        # send to cloud
        serialized_tensor = tf.io.serialize_tensor(intermediate).numpy()
        req = service_pb2.SplitRequest(network=self.network,
                                       partition_index=self.partition_index,
                                       tensor=serialized_tensor,
                                       accelerator=cloud_accelerator)
        # set scaling if there was a head network
        if partition_index != 0 and self.network != service_pb2.VISIONTRANSFORMER:
            output_details = self.head.get_output_details()[0]
            req.scale = output_details["quantization_parameters"]["scales"][0]
            req.zero_point = output_details["quantization_parameters"]["zero_points"][0]
        resp = stub.SplitCompute(req)
        return resp.label


def run(num_images: int, network_arg, accelerator: bool, cloud_accelerator: bool, host: str, port: int,
        partition_index: int) -> None:
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel(host + ":" + str(port), [
        ('grpc.max_send_message_length', MAX_MESSAGE_SEND_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_RECEIVE_LENGTH),
    ]) as channel:

        stub = service_pb2_grpc.SplitServiceStub(channel)
        client = SplitComputeClient()
        if network_arg == 'vgg16':
            network = service_pb2.VGG16
        elif network_arg == 'resnet50':
            network = service_pb2.RESNET50
        elif network_arg == 'mobilenetv2':
            network = service_pb2.MOBILENETV2
        elif network_arg == 'vit':
            network = service_pb2.VISIONTRANSFORMER

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
        if network == service_pb2.VGG16 or network == service_pb2.RESNET50:
            validation_ds = validation_ds.map(normalize_img_caffe, num_parallel_calls=tf.data.AUTOTUNE).skip(100)
        elif network == service_pb2.MOBILENETV2 or network == service_pb2.VISIONTRANSFORMER:
            validation_ds = validation_ds.map(normalize_img_tf, num_parallel_calls=tf.data.AUTOTUNE).skip(100)

        it = iter(validation_ds)
        top1 = 0
        # do one prediction extra for loading the model on client and server
        image, label = next(it)
        client.split_compute(stub, image, network, partition_index, accelerator, cloud_accelerator)
        # notify controller over stdout
        print("<>Init done")
        sys.stdout.flush()
        start_time = time.perf_counter_ns()
        for img_num in range(num_images):
            image, label = next(it)
            pred_label = client.split_compute(stub, image, network, partition_index, accelerator, cloud_accelerator)
            if get_label_name(label) == pred_label:
                top1 += 1
        end_time = time.perf_counter_ns()
        print(f"<>latency: {end_time - start_time}")
        sys.stdout.flush()
        if network != service_pb2.VISIONTRANSFORMER:
            print(f"<>accuracy: {top1 / num_images * 100}")
            sys.stdout.flush()


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--cloud_gpu', type=bool, default=False, help='Use cloud accelerator (GPU).', )
    parser.add_argument('-m', '--model', type=str, choices=['vgg16', 'resnet50', 'mobilenetv2', 'vit'],
                        default='vgg16',
                        help='The neural network model to be used.')
    parser.add_argument('-n', '--n_samples', type=int, default=100, help='The number of samples to average over.')
    parser.add_argument('-s', '--splitting_point', type=int, default=0, help='Index of the splitting point.')
    parser.add_argument('-p', '--port', type=int, default=50051, help='The port to connect to.')
    parser.add_argument('-i', '--ip', type=str, default='192.168.167.81', help='The server IP address to connect to.')
    parser.add_argument('-t', '--tpu_mode', type=str, choices=['off', 'std', 'max'], default='std',
                        help='The TPU mode to be used.')
    return parser.parse_args()


def main(args: argparse.Namespace):
    run(args.n_samples, args.model, args.tpu_mode != 'off', args.cloud_gpu, args.ip, args.port, args.splitting_point)
    return 0


if __name__ == "__main__":
    sys.exit(main(read_args()))
