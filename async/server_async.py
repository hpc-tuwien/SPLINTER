import grpc
import asyncio
import example_pb2
import example_pb2_grpc
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np
import time
import threading

# Load the VGG16 model and define the tail model starting after the 'block5_pool' layer
base_model = VGG16(weights='imagenet', include_top=True)

# Start after the 'block5_pool' layer (this is the tail model)
tail_input = Input(shape=(7, 7, 512))
x = base_model.get_layer('flatten')(tail_input)
x = base_model.get_layer('fc1')(x)
x = base_model.get_layer('fc2')(x)
tail_output = base_model.get_layer('predictions')(x)

tail_model = Model(inputs=tail_input, outputs=tail_output)


class InferenceServiceServicer(example_pb2_grpc.InferenceServiceServicer):
    def __init__(self):
        self.num_requests = 0
        self.received_requests = []
        self.processed_results = {}
        self.events = []  # List to store events for chaining responses

    async def InitializeSession(self, request, context):
        self.num_requests = request.num_requests
        self.received_requests = []
        self.processed_results = {}
        self.events = [asyncio.Event() for _ in range(self.num_requests)]
        print(f"[{time.strftime('%H:%M:%S')}] Initialized session for {self.num_requests} requests.")
        return example_pb2.SessionInitResponse(message=f"Ready to receive {self.num_requests} requests.")

    async def SendInference(self, request, context):
        print(f"[{time.strftime('%H:%M:%S')}] Received request ID {request.id}.")
        self.received_requests.append(request)

        if len(self.received_requests) == self.num_requests:
            print(f"[{time.strftime('%H:%M:%S')}] Last request received. Starting processing...")
            start_time = time.time()

            # Process all requests
            for req in sorted(self.received_requests, key=lambda r: r.id):
                # Deserialize the incoming tensor
                tensor = tf.io.parse_tensor(req.tensor, out_type=tf.float32)
                tensor = tf.reshape(tensor, (7, 7, 512))

                # Expand dimensions to match the expected input shape for the tail model
                tensor = tf.expand_dims(tensor, axis=0)  # Now tensor has shape (1, 7, 7, 512)

                # Perform inference on the tail model
                inference_start_time = time.time()
                predictions = tail_model(tensor)
                inference_end_time = time.time()

                # Decode predictions to get human-readable labels
                decoded_predictions = decode_predictions(predictions.numpy(), top=1)
                class_label = decoded_predictions[0][0][1]  # Get the class label

                # Store the result corresponding to the request ID
                self.processed_results[req.id] = class_label
                print(
                    f"[{time.strftime('%H:%M:%S')}] Processed request ID {req.id}. Inference time: {inference_end_time - inference_start_time:.4f} seconds.")

            total_processing_time = time.time() - start_time
            print(
                f"[{time.strftime('%H:%M:%S')}] Processing complete. Total processing time: {total_processing_time:.4f} seconds.")
            print(f"[{time.strftime('%H:%M:%S')}] Active threads: {threading.active_count()}")

            # Set the event for the first request to start sending responses
            self.events[0].set()

        # The first request waits for the event that is set after processing is complete
        await self.events[request.id].wait()

        # Send the response for this request
        result = self.processed_results[request.id]
        print(f"[{time.strftime('%H:%M:%S')}] Sending response for request ID {request.id} with result: {result}")

        # Signal the next request can proceed
        if request.id < self.num_requests - 1:
            self.events[request.id + 1].set()

        return example_pb2.InferenceResponse(id=request.id, result=result.encode('utf-8'))


async def serve():
    print(f"[{time.strftime('%H:%M:%S')}] Starting gRPC server...")
    server = grpc.aio.server()
    example_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    print(f"[{time.strftime('%H:%M:%S')}] gRPC server started. Listening on port 50051.")
    await server.wait_for_termination()


if __name__ == '__main__':
    print(f"[{time.strftime('%H:%M:%S')}] Starting server process...")
    asyncio.run(serve())
