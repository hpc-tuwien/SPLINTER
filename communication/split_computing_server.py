import time
from concurrent import futures

import grpc
import tensorflow as tf
from keras.applications import mobilenet_v2
from keras.applications import resnet50
from keras.applications import vgg16

import service_pb2
import service_pb2_grpc

PATH_NETWORK = {service_pb2.VGG16: "VGG16",
                service_pb2.RESNET50: "resnet50",
                service_pb2.MOBILENETV2: "mobilenetv2"}

MAX_MESSAGE_LENGTH = 12845090


class SplitServiceServicer(service_pb2_grpc.SplitServiceServicer):

    def __init__(self):
        self.network = None
        self.partition_index = None
        self.tail = None
        self.input_details = None
        self.output_details = None
        self.input_scale = None
        self.input_zero_point = None

    def SplitCompute(self, request, context):
        start_counter_ns = time.perf_counter_ns()
        if (request.network == self.network) and (request.partition_index == self.partition_index):
            pass
        else:
            self.network = request.network
            self.partition_index = request.partition_index
            self.tail = tf.lite.Interpreter(
                model_path="../" + PATH_NETWORK[self.network] + "/models/tail/" + str(
                    self.partition_index) + ".tflite")
            self.tail.allocate_tensors()
            # Get input and output tensors from tail network and convert tensor.
            self.input_details = self.tail.get_input_details()[0]
            self.output_details = self.tail.get_output_details()[0]
            self.input_scale, self.input_zero_point = self.input_details["quantization"]

        intermediate_float = tf.io.parse_tensor(request.tensor, out_type=tf.float32).numpy()
        intermediate_int = (intermediate_float / self.input_scale + self.input_zero_point).astype(
            self.input_details["dtype"])

        # invoke tail network
        self.tail.set_tensor(self.input_details["index"], intermediate_int)
        self.tail.invoke()
        output_data = self.tail.get_tensor(self.output_details['index'])

        if self.network == service_pb2.VGG16:
            classes = [label_id for label_id, _, _ in vgg16.decode_predictions(output_data, top=5)[0]]
        elif self.network == service_pb2.RESNET50:
            classes = [label_id for label_id, _, _ in resnet50.decode_predictions(output_data, top=5)[0]]
        elif self.network == service_pb2.MOBILENETV2:
            classes = [label_id for label_id, _, _ in mobilenet_v2.decode_predictions(output_data, top=5)[0]]

        end_counter_ns = time.perf_counter_ns()

        return service_pb2.SplitResponse(
            classes=classes,
            server_time=end_counter_ns - start_counter_ns
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1),
                         options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                  ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    service_pb2_grpc.add_SplitServiceServicer_to_server(SplitServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
