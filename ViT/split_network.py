# %%
import tensorflow as tf
from keras import Sequential
from tqdm import tqdm
from vit_keras import vit

from utils import split_functional_model


# %%
def save_tflite(model, name):
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open("models/head/" + str(name) + '.tflite', 'wb') as f:
        f.write(tflite_model)


# %%
def igelu(x):
    import math
    a = -0.2888
    b = -1.769
    return 0.5 * x * (1 + (tf.math.tanh(1000 * (x / math.sqrt(2))) * (
            a * (tf.math.minimum((x / math.sqrt(2)) * tf.math.tanh(1000 * (x / math.sqrt(2))), -b) + b) ** 2 + 1)))


# %%
model = vit.vit_b16()


# %%
def replace_gelu(model):
    new_model = Sequential()
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Lambda):
            new_layer = tf.keras.layers.Lambda(igelu)
        else:
            new_layer = layer

        new_model.add(new_layer)

    return new_model


# %%
# replace Dense layer and lambda gelu layer with dense layered with a fused polynomial approximation of gelu activation
for layer in model.layers:
    if "encoderblock_" in layer.name:
        layer.mlpblock = replace_gelu(layer.mlpblock)
# %%
# forward pass is needed for weight initialization (also sets batch size to 1)
dummy_input = tf.zeros((1,) + model.input_shape[1:])
_ = model(dummy_input)
# %%
print("Save full models")
# save_tflite(model, "19")
model.save("models/tail/0")
# %%
print("Save partial models")
# skip full model with first and last index
for i in tqdm(range(1, 19)):
    head, tail = split_functional_model(model, i)
    # save_tflite(head, i)
    tail.save("models/tail/" + str(i))
