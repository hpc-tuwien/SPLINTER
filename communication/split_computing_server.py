import argparse
import asyncio
import configparser
import datetime
import math
import time
import warnings

import grpc
import numpy as np
import pandas as pd
import psutil
import pynvml
import requests
import tensorflow as tf
from google.protobuf.duration_pb2 import Duration
from requests.auth import HTTPBasicAuth
from sklearn.metrics import auc

import service_pb2
import service_pb2_grpc

# Constants
PATH_NETWORK = {
    service_pb2.Network.VGG_16: "VGG16",
    service_pb2.Network.RES_NET_50: "resnet50",
    service_pb2.Network.MOBILE_NET_V2: "mobilenetv2",
    service_pb2.Network.VISION_TRANSFORMER: "ViT"
}

ACCELERATOR = {
    True: "/GPU:0",
    False: "/CPU:0"
}

WARMUP_INFERENCES = 5
MAX_MESSAGE_SEND_LENGTH = -1
MAX_MESSAGE_RECEIVE_LENGTH = -1


# Load credentials from config.ini
def load_credentials():
    config = configparser.ConfigParser(interpolation=None)  # Disable interpolation
    config.read('config.ini')
    return config['power_api']['username'], config['power_api']['password']


# Helper function to generate warmup input
def generate_warmup_input(input_shape):
    input_shape = [dim if dim is not None else 1 for dim in input_shape]  # Replace None with 1 for batch size
    return tf.random.uniform(shape=input_shape, minval=0, maxval=1, dtype=tf.float32)


def get_node_energy(host: str, start_time_ns, end_time_ns, num_requests):
    """
        Request power data from the API and compute node energy using the trapezoidal rule.

        Args:
            host (str): Hostname of the server.
            start_time_ns (int): Start time in nanoseconds.
            end_time_ns (int): End time in nanoseconds.
            num_requests (int): Number of inferences to divide the total energy.

        Returns:
            float: Joules per inference.
        """
    # Load credentials from the INI file
    username, password = load_credentials()

    # Extract node and site from the host
    node = host.split('-ipv6')[0]
    site = host.split('.')[1]

    # Convert the start and end times to seconds and then to UTC
    start_time_s = math.floor(start_time_ns / 1_000_000_000)
    end_time_s = math.ceil(end_time_ns / 1_000_000_000)

    # Convert the start and end times to UTC before making the request
    start_time_iso = datetime.datetime.utcfromtimestamp(start_time_s).isoformat() + 'Z'
    end_time_iso = datetime.datetime.utcfromtimestamp(end_time_s).isoformat() + 'Z'

    url = f"https://api.grid5000.fr/stable/sites/{site}/metrics?nodes={node}&metrics=wattmetre_power_watt&start_time={start_time_iso}&end_time={end_time_iso}"

    response = requests.get(url, auth=HTTPBasicAuth(username, password), verify=False)

    if response.status_code != 200:
        raise Exception(f"Could not get power measurements.")

    # Parse the JSON response
    json_data = response.json()

    # Create a dataframe from the JSON data
    df = pd.DataFrame(json_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Convert the start and end timestamps to pandas datetime objects
    start_time = pd.to_datetime(start_time_ns, unit='ns').tz_localize('UTC').tz_convert('Europe/Paris')
    end_time = pd.to_datetime(end_time_ns, unit='ns').tz_localize('UTC').tz_convert('Europe/Paris')

    # Filter the dataframe for the relevant time period
    df_filtered = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

    if df_filtered.empty:
        print("No data in the filtered range.")
        return 0.0

    # Extract time in seconds and power values
    time_values = (df_filtered['timestamp'].astype(int) / 1_000_000_000).values  # Convert timestamp to seconds
    power_values = df_filtered['value'].values  # Power in watts

    # Compute energy (Joules) using AUC (trapezoidal rule)
    total_energy_joules = auc(time_values, power_values)

    # Return energy per inference (joules per inference)
    return total_energy_joules / num_requests


# gRPC service implementation
class SplitServiceServicer(service_pb2_grpc.SplitServiceServicer):

    def __init__(self):
        self.network = None
        self.partition_index = None
        self.tail = None
        self.zero_point = None
        self.scale = None
        self.accelerator = None
        self.num_requests = 0
        self.received_requests = []
        self.latest_metrics = None
        self.metrics_ready = asyncio.Event()  # Event to signal when metrics are ready
        # Initialize NVML and GPU handle once when the server starts
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        if not tf.config.list_physical_devices('GPU'):
            warnings.warn("No GPU detected.")

    def get_gpu_utilization(self):
        # Get GPU utilization rates (returns percentage)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        return utilization.gpu  # Percent of GPU utilization

    async def InitializeSession(self, request, context):
        print("[Server] Initializing session.")
        self._initialize_session_variables(request)
        self._load_and_warmup_model()
        print("[Server] Session initialized.")
        return service_pb2.SessionInitResponse(status=service_pb2.Status.READY)

    def _initialize_session_variables(self, request):
        self.num_requests = request.num_requests
        print(f"Requests to receive: {self.num_requests}")
        self.network = request.network
        self.partition_index = request.partition_index
        self.accelerator = request.accelerator
        if self.partition_index != 0 and self.network != service_pb2.Network.VISION_TRANSFORMER:
            self.zero_point = request.zero_point
            self.scale = request.scale
        self.received_requests.clear()
        # Reset the metrics ready event at the start of the session
        self.metrics_ready.clear()

    def _load_and_warmup_model(self):
        print("[Server] Loading and warming up model.")
        with tf.device(ACCELERATOR[self.accelerator]):
            model_path = f"../{PATH_NETWORK[self.network]}/models/tail/{self.partition_index}"
            self.tail = tf.keras.models.load_model(model_path, compile=False)
            for _ in range(WARMUP_INFERENCES):
                warmup_input = generate_warmup_input(self.tail.input_shape)
                _ = tf.keras.applications.imagenet_utils.decode_predictions(self.tail.predict(warmup_input, verbose=0),
                                                                            top=1)[0][0][0]
        print("[Server] Model loaded and warmed up.")

    async def SplitCompute(self, request_iterator, context):
        print("[Server] Starting to receive requests.")

        # Accumulate all requests
        async for request in request_iterator:
            self.received_requests.append(request)
            if len(self.received_requests) == self.num_requests:
                print("[Server] All requests received. Starting processing.")
                break

        results = []
        utilization_values = []

        # start of measurements
        loop_start_time_ns = time.perf_counter_ns()
        start_time = time.time_ns()
        _ = psutil.cpu_percent()
        energy_start = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle)

        # Process all accumulated requests at once

        with tf.device(ACCELERATOR[self.accelerator]):
            for req in sorted(self.received_requests, key=lambda r: r.id):
                intermediate_tensor = self._deserialize_tensor(req)
                predictions = self.tail.predict(intermediate_tensor, verbose=0)
                utilization_values.append(self.get_gpu_utilization())
                label = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=1)[0][0][0]
                results.append(service_pb2.SplitResponse(id=req.id, label=label))

        energy_end = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle)
        cpu_utilization = psutil.cpu_percent()
        end_time = time.time_ns()
        loop_end_time_ns = time.perf_counter_ns()

        # Stream the results back once all processing is done
        print("[Server] Finished processing. Streaming results back to client.")
        for result in results:
            yield result

        print("[Server] Finished streaming results. Now retrieving node energy.")

        # Get power data
        # TODO change os.environ["HOSTNAME"]
        node_energy = get_node_energy("gemini-2-ipv6.lyon.grid5000.fr", start_time, end_time, self.num_requests)
        print("[Server] Finished getting node energy. Now saving metrics.")

        duration = Duration()
        duration.FromNanoseconds(round((loop_end_time_ns - loop_start_time_ns) / self.num_requests))

        # Create metrics to be saved
        self.latest_metrics = {
            "cpu_utilization": cpu_utilization,
            "gpu_utilization": np.mean(utilization_values),
            "gpu_energy": ((energy_end - energy_start) / 1000) / self.num_requests,  # J per inference
            "node_energy": node_energy,  # J per inference
            "server_latency": duration
        }

        # Set the event to signal that the metrics are ready
        self.metrics_ready.set()

    def _deserialize_tensor(self, request):
        if self.partition_index == 0 or self.network == service_pb2.Network.VISION_TRANSFORMER:
            return tf.io.parse_tensor(request.tensor, out_type=tf.float32)
        tensor = tf.cast(tf.io.parse_tensor(request.tensor, out_type=tf.int8), tf.float32)
        return (tensor - self.zero_point) * self.scale

    async def GetMetrics(self, request, context):
        # Wait until the metrics are ready
        print("[Server] Waiting for metrics to be available.")
        await self.metrics_ready.wait()

        if self.latest_metrics is None:
            raise Exception("No metrics available. Please run SplitCompute first.")
        print("[Server] Sending metrics.")
        return service_pb2.Metrics(
            server_latency=self.latest_metrics['server_latency'],
            cpu_utilization=self.latest_metrics['cpu_utilization'],
            gpu_utilization=self.latest_metrics['gpu_utilization'],
            gpu_energy=self.latest_metrics['gpu_energy'],
            node_energy=self.latest_metrics['node_energy'],
        )


async def serve(port: int):
    server = grpc.aio.server(options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_SEND_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_RECEIVE_LENGTH)
    ])
    service_pb2_grpc.add_SplitServiceServicer_to_server(SplitServiceServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    await server.wait_for_termination()


# Argument parsing
def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--port', type=int, default=50051, help='The port to listen on.')
    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    try:
        asyncio.run(serve(args.port))
    finally:
        pynvml.nvmlShutdown()
