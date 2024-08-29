import grpc
import asyncio
import example_pb2
import example_pb2_grpc
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import numpy as np
import time
import multiprocessing
import threading

# Load the VGG16 model and extract the head part
base_model = VGG16(weights='imagenet')
head_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)


async def send_inference(stub, inference_id, tensor):
    # Serialize the tensor
    serialized_tensor = tf.io.serialize_tensor(tensor).numpy()

    # Create the gRPC request
    request = example_pb2.InferenceRequest(id=inference_id, tensor=serialized_tensor)

    # Perform the asynchronous gRPC call
    response_start_time = time.time()
    response = await stub.SendInference(request)
    response_end_time = time.time()

    print(
        f"[{time.strftime('%H:%M:%S')}] Received response for inference ID {response.id}. Response time: {response_end_time - response_start_time:.4f} seconds.")

    # The result is already a string, no need to decode
    result = response.result  # No decoding needed
    return response.id, result


async def main():
    print(f"[{time.strftime('%H:%M:%S')}] Starting client...")

    # Connect to the gRPC server
    async with grpc.aio.insecure_channel('gemini-2-ipv6.lyon.grid5000.fr:50051') as channel:
        stub = example_pb2_grpc.InferenceServiceStub(channel)

        # Load and preprocess images
        images = [tf.keras.preprocessing.image.load_img(f'../turtle.jpg', target_size=(224, 224)) for i in range(10)]
        images = [tf.keras.preprocessing.image.img_to_array(image) for image in images]
        images = [np.expand_dims(image, axis=0) for image in images]
        images = [tf.keras.applications.vgg16.preprocess_input(image) for image in images]

        # Initialize the session
        num_requests = len(images)
        init_response = await stub.InitializeSession(example_pb2.SessionInitRequest(num_requests=num_requests))
        print(init_response.message)

        # Perform head network inferences consecutively
        head_outputs = []
        for i, image in enumerate(images):
            inference_start_time = time.time()
            head_output = head_model(image)
            inference_end_time = time.time()
            head_outputs.append(head_output[0])
            print(
                f"[{time.strftime('%H:%M:%S')}] Performed head inference for image {i}. Inference time: {inference_end_time - inference_start_time:.4f} seconds.")

        # Send all inferences asynchronously to the server
        send_tasks = [send_inference(stub, i, head_outputs[i]) for i in range(len(head_outputs))]
        results = await asyncio.gather(*send_tasks)

        print(f"[{time.strftime('%H:%M:%S')}] All inferences completed. Active threads: {threading.active_count()}")
        print(f"[{time.strftime('%H:%M:%S')}] Active processes: {len(multiprocessing.active_children())}")

        # Process the results
        for inf_id, result in results:
            print(f"Received result for inference id {inf_id} with class label: {result}")


if __name__ == '__main__':
    start_time = time.time()
    asyncio.run(main())
    total_time = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}] Client process completed. Total execution time: {total_time:.4f} seconds.")
