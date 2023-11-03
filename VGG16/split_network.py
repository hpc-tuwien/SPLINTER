from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.applications import vgg16
from keras.src.applications import VGG16
from tqdm import tqdm

from VGG16.utils import split_sequential_model


def normalize_img(img, lbl):
    """Normalizes images: `uint8` -> `float32`."""
    img = tf.image.resize_with_pad(img, 224, 224)
    img = vgg16.preprocess_input(img)
    return img, lbl


# load caltech101 dataset
test_ds, metadata = tfds.load(
    'caltech101',
    split='test',
    with_info=True,
    as_supervised=True,
)
test_ds = test_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
get_label_name = metadata.features['label'].int2str


def representative_dataset(head_network=None):
    # should be 100 to 500 according to documentation
    number_of_samples = 100
    if head_network is None:
        for data in test_ds.batch(1).take(number_of_samples):
            yield [data[0]]
    else:
        interpreter = tf.lite.Interpreter(model_content=head_network)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_scale, input_zero_point = input_details["quantization"]
        output_scale, output_zero_point = output_details["quantization"]
        for data in test_ds.batch(1).take(number_of_samples):
            test_image = data[0][0] / input_scale + input_zero_point
            test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
            interpreter.set_tensor(input_details["index"], test_image)
            interpreter.invoke()
            yield [((interpreter.get_tensor(output_details['index']) - output_zero_point) * output_scale).astype(
                "float32")]


def quantize_and_save_model(model, name, head_network=None):
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = partial(representative_dataset, head_network)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_model = converter.convert()

    # Save the model.
    with open("models/" + name + '.tflite', 'wb') as f:
        f.write(tflite_model)
    return tflite_model


vgg16_model = VGG16(weights='imagenet')

print("Save full model")
quantize_and_save_model(vgg16_model, "full")

print("Save partial models")
# skip full model with first and last index
for i in tqdm(range(1, len(vgg16_model.layers) - 1)):
    head, tail = split_sequential_model(vgg16_model, i)
    head_quantized = quantize_and_save_model(head, "head/" + str(i))
    quantize_and_save_model(tail, "tail/" + str(i), head_quantized)

# prediction on the full 32 bit floating point model
it = iter(test_ds)
for _ in range(4):
    image, label = next(it)
preds = vgg16_model.predict(tf.stack([image]))
print('32 bit fp prediction:', vgg16.decode_predictions(preds, top=5)[0])

# prediction on the TensorFlow Lite 8 bit quantized model
interpreter = tf.lite.Interpreter(model_path="models/full.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# convert image to 8 bit
input_scale, input_zero_point = input_details["quantization"]
test_image = image / input_scale + input_zero_point
test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
interpreter.set_tensor(input_details["index"], test_image)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details['index'])
print('8 bit quantized prediction:', vgg16.decode_predictions(output_data, top=5)[0])


def show_results(split_at):
    # prediction on the TensorFlow Lite 8 bit split quantized model
    head = tf.lite.Interpreter(model_path="models/head/" + str(split_at) + ".tflite")
    tail = tf.lite.Interpreter(model_path="models/tail/" + str(split_at) + ".tflite")
    head.allocate_tensors()
    tail.allocate_tensors()

    # Get input and output tensors from head network.
    input_details = head.get_input_details()[0]
    output_details = head.get_output_details()[0]

    # convert image
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]
    test_image = image / input_scale + input_zero_point
    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])

    # invoke head network
    head.set_tensor(input_details["index"], test_image)
    head.invoke()
    intermediate = head.get_tensor(output_details['index'])

    # rescale tensor
    intermediate_float = ((intermediate - output_zero_point) * output_scale).astype("float32")

    # Get input and output tensors from tail network and convert tensor.
    input_details = tail.get_input_details()[0]
    output_details = tail.get_output_details()[0]
    input_scale, input_zero_point = input_details["quantization"]
    intermediate_int = (intermediate_float / input_scale + input_zero_point).astype(input_details["dtype"])

    # invoke tail network
    tail.set_tensor(input_details["index"], intermediate_int)
    tail.invoke()
    output_data = tail.get_tensor(output_details['index'])
    print('8 bit split at ' + str(split_at) + ' prediction:', vgg16.decode_predictions(output_data, top=5)[0])


for i in range(1, len(vgg16_model.layers) - 1):
    show_results(i)
