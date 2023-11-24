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

MAX_MESSAGE_LENGTH = 100


class SplitServiceServicer(service_pb2_grpc.SplitServiceServicer):

    def __init__(self):
        self.network = None
        self.partition_index = None
        self.tail = None

    def SplitCompute(self, request, context):
        start_server_time = time.perf_counter_ns()
        # load tail network
        if (request.network != self.network) or (request.partition_index != self.partition_index):
            self.network = request.network
            self.partition_index = request.partition_index
            self.tail = tf.lite.Interpreter(
                model_path="../" + PATH_NETWORK[self.network] + "/models/tail/" + str(
                    self.partition_index) + ".tflite")
            self.tail.allocate_tensors()

        # rescale to 32bit float if head network was involved
        if self.partition_index != 0:
            # TODO check if numpy call neccesary
            tensor = tf.io.parse_tensor(request.tensor, out_type=tf.int8).numpy()
            intermediate_float = ((tensor - request.zero_point) * request.scale).astype("float32")
        else:
            # TODO check if numpy call neccesary
            intermediate_float = tf.io.parse_tensor(request.tensor, out_type=tf.float32).numpy()

        # scale tensor for tail network
        input_details = self.tail.get_input_details()[0]
        input_scale = input_details["quantization_parameters"]["scales"][0]
        input_zero_point = input_details["quantization_parameters"]["zero_points"][0]

        intermediate_int = (intermediate_float / input_scale + input_zero_point).astype(input_details["dtype"])
        # invoke tail network
        self.tail.set_tensor(input_details["index"], intermediate_int)
        self.tail.invoke()
        output_details = self.tail.get_output_details()[0]
        output_data = self.tail.get_tensor(output_details['index'])

        if self.network == service_pb2.VGG16:
            classes = [label_id for label_id, _, _ in vgg16.decode_predictions(output_data, top=5)[0]]
        elif self.network == service_pb2.RESNET50:
            classes = [label_id for label_id, _, _ in resnet50.decode_predictions(output_data, top=5)[0]]
        elif self.network == service_pb2.MOBILENETV2:
            classes = [label_id for label_id, _, _ in mobilenet_v2.decode_predictions(output_data, top=5)[0]]

        end_server_time = time.perf_counter_ns()

        return service_pb2.SplitResponse(
            classes=classes,
            server_time=end_server_time - start_server_time
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
