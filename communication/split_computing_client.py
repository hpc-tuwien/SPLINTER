import time

import grpc
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
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

    def split_compute(self, stub, image, network, partition_index):
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
            intermediate = image
        else:
            if (network == self.network) and (partition_index == self.partition_index):
                pass
            else:
                self.network = network
                self.partition_index = partition_index
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
        return classes, client_time, server_time, network_time


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel("localhost:50051", [
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]) as channel:

        stub = service_pb2_grpc.SplitServiceStub(channel)
        client = SplitComputeClient()
        network = service_pb2.VGG16
        num_images = 10

        validation_ds, metadata = tfds.load(
            'imagenet2012',
            split='validation',
            with_info=True,
            as_supervised=True,
        )
        # TODO maybe better sampling?
        get_label_name = metadata.features['label'].int2str

        # preprocess according to network; and skip images used for quantization
        if network == service_pb2.VGG16:
            validation_ds = validation_ds.map(normalize_img_vgg16, num_parallel_calls=tf.data.AUTOTUNE).skip(100)
        elif network == service_pb2.RESNET50:
            validation_ds = validation_ds.map(normalize_img_resnet50, num_parallel_calls=tf.data.AUTOTUNE).skip(100)
        elif network == service_pb2.MOBILENETV2:
            validation_ds = validation_ds.map(normalize_img_mobilenetv2, num_parallel_calls=tf.data.AUTOTUNE).skip(100)
        accuracies = pd.DataFrame(range(LOCAL_COMPUTE_IDX[network] + 1), columns=['layer'])
        top1 = list()
        top5 = list()
        for partition_index in tqdm(range(LOCAL_COMPUTE_IDX[network] + 1)):
            it = iter(validation_ds)
            top1_cnt = 0
            top5_cnt = 0

            for _ in range(num_images):
                image, label = next(it)
                pred_classes, client_time, server_time, network_time = client.split_compute(stub, image,
                                                                                            service_pb2.VGG16,
                                                                                            partition_index)
                if get_label_name(label) in pred_classes:
                    top5_cnt += 1
                if get_label_name(label) == pred_classes[0]:
                    top1_cnt += 1
            top1.append(top1_cnt / num_images)
            top5.append(top5_cnt / num_images)
        accuracies['top1'] = top1
        accuracies['top5'] = top5
        accuracies.to_csv("VGG16_CPU_accuracies.csv", index=False)
        # TODO write timings into file!


if __name__ == "__main__":
    run()
