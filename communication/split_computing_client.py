import logging
import time

import grpc
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.applications import resnet50

from communication import service_pb2_grpc, service_pb2

path_prefix = {service_pb2.VGG16: "VGG16",
               service_pb2.RESNET50: "resnet50",
               service_pb2.MOBILENETV2: "mobilenetv2"}


def normalize_img(img, lbl):
    """Normalizes images: `uint8` -> `float32`."""
    img = tf.image.resize_with_pad(img, 224, 224)
    # todo change!!
    img = resnet50.preprocess_input(img)
    return img, lbl


def split_compute(stub, image, network, partition_index, quantized_tail):
    start_counter_ns = time.perf_counter_ns()
    head = tf.lite.Interpreter(
        model_path="../" + path_prefix[network] + "/models/head/" + str(partition_index) + ".tflite")
    head.allocate_tensors()

    # Get input and output tensors from head network.
    input_details = head.get_input_details()[0]
    output_details = head.get_output_details()[0]

    # convert image
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]
    scaled_image = image / input_scale + input_zero_point
    batch_int = np.expand_dims(scaled_image, axis=0).astype(input_details["dtype"])

    # invoke head network
    head.set_tensor(input_details["index"], batch_int)
    head.invoke()
    intermediate = head.get_tensor(output_details['index'])

    # rescale tensor
    intermediate_scaled = ((intermediate - output_zero_point) * output_scale).astype("float32")
    serialized_tensor = tf.io.serialize_tensor(intermediate_scaled).numpy()
    end_counter_ns = time.perf_counter_ns()
    start_network_time = time.perf_counter_ns()
    resp = stub.SplitCompute(service_pb2.SplitRequest(network=network,
                                                      partition_index=partition_index,
                                                      tensor=serialized_tensor,
                                                      quantized_tail=quantized_tail))
    end_network_time = time.perf_counter_ns()
    network_time = end_network_time - start_network_time
    print("Classes:", resp.classes, "calculated in")
    print("Split at", partition_index)
    print("Client time", (end_counter_ns - start_counter_ns) / 1000000, "ms")
    print("Server time:", resp.server_time / 1000000, "ms")
    print("Network time:", (network_time - resp.server_time) / 1000000, "ms")
    return resp.classes, resp.server_time


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = service_pb2_grpc.SplitServiceStub(channel)
        validation_ds, metadata = tfds.load(
            'imagenet2012',
            split='validation',
            with_info=True,
            as_supervised=True,
        )
        validation_ds = validation_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        get_label_name = metadata.features['label'].int2str
        it = iter(validation_ds)
        for _ in range(8):
            image, label = next(it)
        print("Predicting label", get_label_name(label))
        split_compute(stub, image, service_pb2.RESNET50, 6, True)


if __name__ == "__main__":
    logging.basicConfig()
    run()
