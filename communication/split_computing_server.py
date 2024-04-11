import argparse
import warnings
from concurrent import futures

import grpc
import tensorflow as tf
from keras.src.applications import imagenet_utils

import service_pb2
import service_pb2_grpc

PATH_NETWORK = {service_pb2.VGG16: "VGG16",
                service_pb2.RESNET50: "resnet50",
                service_pb2.MOBILENETV2: "mobilenetv2",
                service_pb2.VISIONTRANSFORMER: "ViT"}

ACCELERATOR = {True: "/GPU:0",
               False: "/CPU:0"}

MAX_MESSAGE_SEND_LENGTH = 11
MAX_MESSAGE_RECEIVE_LENGTH = 3211308


class SplitServiceServicer(service_pb2_grpc.SplitServiceServicer):

    def __init__(self):
        self.network = None
        self.partition_index = None
        self.tail = None
        if not tf.config.list_physical_devices('GPU'):
            warnings.warn("No GPU detected.")

    def SplitCompute(self, request, context):
        with tf.device(ACCELERATOR[request.accelerator]):
            # load tail network
            if (request.network != self.network) or (request.partition_index != self.partition_index):
                self.network = request.network
                self.partition_index = request.partition_index
                self.tail = tf.keras.models.load_model(
                    "../" + PATH_NETWORK[self.network] + "/models/tail/" + str(self.partition_index), compile=False)

            if self.partition_index == 0 or self.network == service_pb2.VISIONTRANSFORMER:
                intermediate_float = tf.io.parse_tensor(request.tensor, out_type=tf.float32)
            # rescale to 32bit float if head network was involved
            else:
                tensor = tf.cast(tf.io.parse_tensor(request.tensor, out_type=tf.int8), tf.float32)
                intermediate_float = ((tensor - request.zero_point) * request.scale)

            # invoke tail network
            preds = self.tail.predict(intermediate_float, verbose=0)
            label = imagenet_utils.decode_predictions(preds, top=1)[0][0][0]
            return service_pb2.SplitResponse(label=label)


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1),
                         options=[('grpc.max_send_message_length', MAX_MESSAGE_SEND_LENGTH),
                                  ('grpc.max_receive_message_length', MAX_MESSAGE_RECEIVE_LENGTH)])
    service_pb2_grpc.add_SplitServiceServicer_to_server(SplitServiceServicer(), server)
    server.add_insecure_port("[::]:" + str(port))
    server.start()
    server.wait_for_termination()


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--port', type=int, default=50051, help='The port to listen on.')
    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    serve(args.port)
