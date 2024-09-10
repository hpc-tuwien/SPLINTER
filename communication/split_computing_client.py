import argparse
import asyncio
import os
import time

import grpc
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tflite_runtime.interpreter as tflite

import service_pb2
import service_pb2_grpc

# Constants
PATH_PREFIX = {
    'vgg16': "VGG16",
    'resnet50': "resnet50",
    'mobilenetv2': "mobilenetv2",
    'vit': "ViT"
}

LOCAL_COMPUTE_IDX = {
    'vgg16': 22,
    'resnet50': 40,
    'mobilenetv2': 75,
    'vit': 19
}

NETWORK_ENUM_MAP = {
    'vgg16': service_pb2.Network.VGG_16,
    'resnet50': service_pb2.Network.RES_NET_50,
    'mobilenetv2': service_pb2.Network.MOBILE_NET_V2,
    'vit': service_pb2.Network.VISION_TRANSFORMER
}

WARMUP_INFERENCES = 5
MAX_MESSAGE_SEND_LENGTH = -1
MAX_MESSAGE_RECEIVE_LENGTH = -1

DATA_DIR = '/home/pi/tensorflow_datasets/downloads'
WRITE_DIR = '/home/pi/tensorflow_datasets/extracted'


def normalize_img(img, lbl, mode):
    """Normalizes images: `uint8` -> `float32`."""
    img = tf.image.resize_with_pad(img, 224, 224)
    img = tf.keras.applications.imagenet_utils.preprocess_input(img, data_format=None, mode=mode)
    return img, lbl


def preprocess_dataset(network, validation_ds, skip):
    """Preprocess the dataset according to the network type."""
    mode = 'caffe' if network in ('vgg16', 'resnet50') else 'tf'
    return validation_ds.map(lambda img, lbl: normalize_img(img, lbl, mode), num_parallel_calls=tf.data.AUTOTUNE).skip(
        100 + skip)


def load_and_prepare_dataset(skip, network):
    """Load and preprocess the dataset."""
    download_config = tfds.download.DownloadConfig(
        extract_dir=os.path.join(WRITE_DIR, 'extracted'),
        manual_dir=DATA_DIR
    )

    validation_ds, metadata = tfds.load(
        'imagenet2012',
        split='validation',
        with_info=True,
        as_supervised=True,
        data_dir=os.path.join(WRITE_DIR, 'data'),
        download=True,
        download_and_prepare_kwargs={'download_dir': os.path.join(WRITE_DIR, 'downloaded'),
                                     'download_config': download_config}
    )

    return preprocess_dataset(network, validation_ds, skip), metadata.features['label'].int2str


def prepare_image(network, image, input_details=None, partition_index=0):
    """Prepare image for inference based on network type and partition index."""
    if partition_index == 0 or network == 'vit':
        return np.expand_dims(image, axis=0)

    input_scale = input_details["quantization_parameters"]["scales"][0]
    input_zero_point = input_details["quantization_parameters"]["zero_points"][0]
    scaled_image = image / input_scale + input_zero_point
    return np.expand_dims(scaled_image, axis=0).astype(input_details["dtype"])


def load_model(network, partition_index, edge_accelerator):
    """Load the TFLite model based on the network type and partition index."""
    model_suffix = "_edgetpu.tflite" if edge_accelerator else ".tflite"
    model_path = f"../{PATH_PREFIX[network]}/models/head/{partition_index}{model_suffix}"
    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')] if edge_accelerator else None
    )
    interpreter.allocate_tensors()
    return interpreter


def perform_warmup_inferences(interpreter, network, input_details, partition_index):
    """Perform warmup inferences to prime the model."""
    input_shape = input_details['shape'][1:]
    output_details = interpreter.get_output_details()[0]
    for _ in range(WARMUP_INFERENCES):
        warmup_input = tf.random.uniform(shape=input_shape, minval=0, maxval=1, dtype=tf.float32)
        batch_int = prepare_image(network, warmup_input, input_details, partition_index=partition_index)
        interpreter.set_tensor(input_details["index"], batch_int)
        interpreter.invoke()
        result = interpreter.get_tensor(output_details['index'])
        if args.splitting_point == LOCAL_COMPUTE_IDX[args.model]:
            _ = tf.keras.applications.imagenet_utils.decode_predictions(result, top=1)[0][0][0]


async def initialize_session(stub, args, head=None):
    """Initialize session on the server."""
    network_enum_value = NETWORK_ENUM_MAP[args.model]
    init_request = service_pb2.SessionInitRequest(
        num_requests=args.n_samples,
        network=network_enum_value,
        partition_index=args.splitting_point,
        accelerator=args.cloud_gpu
    )

    if args.splitting_point != 0 and network_enum_value != service_pb2.Network.VISION_TRANSFORMER:
        output_details = head.get_output_details()[0]
        init_request.scale = output_details["quantization_parameters"]["scales"][0]
        init_request.zero_point = output_details["quantization_parameters"]["zero_points"][0]

    response = await stub.InitializeSession(init_request)
    if response.status != service_pb2.Status.READY:
        raise RuntimeError("Failed to initialize session on the server")


async def compute_head_inferences(validation_ds, n_samples, int2str, args, head=None, input_details=None,
                                  output_details=None):
    """Compute all head inferences and return the intermediate tensors."""
    head_results = []
    real_labels = {}
    it = iter(validation_ds)

    for i in range(n_samples):
        image, label = next(it)
        real_labels[i] = int2str(label)

        print(f"[Client] Processing head inference for sample ID: {i}")

        if args.splitting_point == 0:
            batch_int = prepare_image(args.model, image)
        else:
            batch_int = prepare_image(args.model, image, input_details, args.splitting_point)
            head.set_tensor(input_details["index"], batch_int)
            head.invoke()
            intermediate = head.get_tensor(output_details['index'])
            batch_int = intermediate

        if args.splitting_point == LOCAL_COMPUTE_IDX[args.model]:
            head_result = tf.keras.applications.imagenet_utils.decode_predictions(batch_int, top=1)[0][0][0]
        else:
            head_result = tf.io.serialize_tensor(batch_int).numpy()
        head_results.append((i, head_result))

        print(f"[Client] Head inference for sample ID: {i} complete.")

    return head_results, real_labels


async def stream_inferences(intermediate_tensors, stub, n_samples):
    """Stream intermediate tensors to the server and receive results."""
    predicted_labels = []

    async def request_stream():
        for i, serialized_tensor in intermediate_tensors:
            yield service_pb2.SplitRequest(id=i, tensor=serialized_tensor)
        print("[Client] Finished streaming intermediate tensors to server.")

    # Stream all intermediate tensors and wait for all results
    print("[Client] Streaming intermediate tensors to server.")
    async for response in stub.SplitCompute(request_stream()):
        predicted_labels.append((response.id, response.label))
    print("[Client] Results received.")
    return predicted_labels


def calculate_accuracy(real_labels, predicted_labels, n_samples):
    """Calculate accuracy of predictions."""
    matches = sum(1 for pred_id, pred_label in predicted_labels if real_labels[pred_id] == pred_label)
    accuracy = matches / n_samples
    print(f"<>Accuracy: {accuracy:.2%}")
    return accuracy


async def main(args):
    validation_ds, int2str = load_and_prepare_dataset(args.skip, args.model)

    async with grpc.aio.insecure_channel(f'{args.ip}:{args.port}', [
        ('grpc.max_send_message_length', MAX_MESSAGE_SEND_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_RECEIVE_LENGTH),
    ]) as channel:
        stub = service_pb2_grpc.SplitServiceStub(channel)

        head, input_details, output_details = None, None, None

        if args.splitting_point != 0:
            head = load_model(args.model, args.splitting_point, args.tpu_mode != 'off')
            input_details = head.get_input_details()[0]
            output_details = head.get_output_details()[0]
            perform_warmup_inferences(head, args.model, input_details, args.splitting_point)

        if args.splitting_point != LOCAL_COMPUTE_IDX[args.model]:
            print("[Client] Initializing session.")
            await initialize_session(stub, args, head)
            print("[Client] Session initialized.")

        # Start measuring the time for the overall inference process
        start_time = time.perf_counter_ns()

        # First, compute all head inferences
        head_results, real_labels = await compute_head_inferences(validation_ds, args.n_samples, int2str, args,
                                                                  head, input_details, output_details)
        edge_completed = time.perf_counter_ns()

        if args.splitting_point != LOCAL_COMPUTE_IDX[args.model]:
            # After all head inferences are computed, stream them and receive results
            predicted_labels = await stream_inferences(head_results, stub, args.n_samples)
        else:
            predicted_labels = head_results

        # Measure the total time for the inference process
        end_time = time.perf_counter_ns()
        total_time_ns = end_time - start_time
        edge_time_ns = edge_completed - start_time
        edge_avg_inference_time_ms = edge_time_ns / args.n_samples / 1_000_000  # Convert to ms
        total_avg_inference_time_ms = total_time_ns / args.n_samples / 1_000_000  # Convert to ms
        print(f"<>Total average inference time: {total_avg_inference_time_ms:.2f} ms")
        print(f"<>Edge average inference time: {edge_avg_inference_time_ms:.2f} ms")

        # Calculate accuracy
        accuracy = calculate_accuracy(real_labels, predicted_labels, args.n_samples)
        if args.splitting_point != LOCAL_COMPUTE_IDX[args.model]:
            # Call GetMetrics to retrieve server-side metrics
            print("[Client] Retrieving metrics from server.")
            get_metrics_request = service_pb2.SessionInitRequest(
                num_requests=args.n_samples,
                network=NETWORK_ENUM_MAP[args.model],
                partition_index=args.splitting_point,
                accelerator=args.cloud_gpu
            )

            # Make the gRPC call to GetMetrics
            metrics_response = await stub.GetMetrics(get_metrics_request)

            # Print the retrieved metrics
            print(f"CPU Utilization: {metrics_response.cpu_utilization:.2f}")
            print(f"GPU Utilization: {metrics_response.gpu_utilization:.2f}")
            print(f"GPU Energy Consumption: {metrics_response.gpu_energy:.2f} Joules")
            print(f"Node Power Measurements: {metrics_response.node_power_measurements}")
            print("Server Latencies:")
            for sample_id, latency in metrics_response.server_latencies.items():
                print(f"Sample ID {sample_id}: {latency.seconds} s, {latency.nanos} ns")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--cloud_gpu', action='store_true', help='Use cloud accelerator (GPU).')
    parser.add_argument('-m', '--model', type=str, choices=['vgg16', 'resnet50', 'mobilenetv2', 'vit'], default='vgg16',
                        help='The neural network model to be used.')
    parser.add_argument('-n', '--n_samples', type=int, default=100, help='The number of samples to average over.')
    parser.add_argument('-k', '--skip', type=int, default=0, help='The number of images to skip.')
    parser.add_argument('-s', '--splitting_point', type=int, default=0, help='Index of the splitting point.')
    parser.add_argument('-p', '--port', type=int, default=50051, help='The port to connect to.')
    parser.add_argument('-i', '--ip', type=str, default='localhost', help='The server IP address to connect to.')
    parser.add_argument('-t', '--tpu_mode', type=str, choices=['off', 'std', 'max'], default='std',
                        help='The TPU mode to be used.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
