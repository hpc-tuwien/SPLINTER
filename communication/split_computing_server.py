import logging
import time
from concurrent import futures

import grpc
import tensorflow as tf
from keras.src.applications import resnet

import service_pb2
import service_pb2_grpc

path_prefix = {service_pb2.VGG16: "VGG16",
               service_pb2.RESNET50: "resnet50",
               service_pb2.MOBILENETV2: "mobilenetv2"}


class SplitServiceServicer(service_pb2_grpc.SplitServiceServicer):
    def SplitCompute(self, request, context):
        start_counter_ns = time.perf_counter_ns()
        start_temp = time.perf_counter_ns()
        tail = tf.lite.Interpreter(
            model_path="../" + path_prefix[request.network] + "/models/tail/" + str(
                request.partition_index) + ".tflite")
        tail.allocate_tensors()
        # Get input and output tensors from tail network and convert tensor.
        input_details = tail.get_input_details()[0]
        output_details = tail.get_output_details()[0]
        input_scale, input_zero_point = input_details["quantization"]
        stop_temp = time.perf_counter_ns()
        print("Model loaded and tensors allocated in", (stop_temp - start_temp) / 1000000, "ms")
        start_temp = time.perf_counter_ns()
        intermediate_float = tf.io.parse_tensor(request.tensor, out_type=tf.float32).numpy()
        intermediate_int = (intermediate_float / input_scale + input_zero_point).astype(input_details["dtype"])

        # invoke tail network
        tail.set_tensor(input_details["index"], intermediate_int)
        tail.invoke()
        output_data = tail.get_tensor(output_details['index'])
        # todo change!
        classes = [label_id for label_id, _, _ in resnet.decode_predictions(output_data, top=5)[0]]
        stop_temp = time.perf_counter_ns()
        print("Inference done in", (stop_temp - start_temp) / 1000000, "ms")
        end_counter_ns = time.perf_counter_ns()

        return service_pb2.SplitResponse(
            classes=classes,
            server_time=end_counter_ns - start_counter_ns
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_SplitServiceServicer_to_server(SplitServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
