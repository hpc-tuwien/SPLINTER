import tensorflow as tf


def split_functional_model(model, partition_idx):
    """
    Split the model MobileNetV2 before layer/block with index partition_idx [1,74].
    """
    if partition_idx <= 17:
        head = tf.keras.Model(model.layers[0].output, model.layers[partition_idx].output)
        tail = tf.keras.Model(model.layers[partition_idx].output, model.layers[len(model.layers) - 1].output)
    elif 18 <= partition_idx <= 27:
        head = tf.keras.Model(model.layers[0].output, model.layers[partition_idx + 8].output)
        tail = tf.keras.Model(model.layers[partition_idx + 8].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 28:
        head = tf.keras.Model(model.layers[0].output, model.layers[44].output)
        tail = tf.keras.Model(model.layers[44].output, model.layers[len(model.layers) - 1].output)
    elif 29 <= partition_idx <= 38:
        head = tf.keras.Model(model.layers[0].output, model.layers[partition_idx + 24].output)
        tail = tf.keras.Model(model.layers[partition_idx + 24].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 39:
        head = tf.keras.Model(model.layers[0].output, model.layers[71].output)
        tail = tf.keras.Model(model.layers[71].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 40:
        head = tf.keras.Model(model.layers[0].output, model.layers[80].output)
        tail = tf.keras.Model(model.layers[80].output, model.layers[len(model.layers) - 1].output)
    elif 41 <= partition_idx <= 49:
        head = tf.keras.Model(model.layers[0].output, model.layers[partition_idx + 48].output)
        tail = tf.keras.Model(model.layers[partition_idx + 48].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 50:
        head = tf.keras.Model(model.layers[0].output, model.layers[106].output)
        tail = tf.keras.Model(model.layers[106].output, model.layers[len(model.layers) - 1].output)
    elif 51 <= partition_idx <= 60:
        head = tf.keras.Model(model.layers[0].output, model.layers[partition_idx + 64].output)
        tail = tf.keras.Model(model.layers[partition_idx + 64].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 61:
        head = tf.keras.Model(model.layers[0].output, model.layers[133].output)
        tail = tf.keras.Model(model.layers[133].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx >= 62:
        head = tf.keras.Model(model.layers[0].output, model.layers[partition_idx + 80].output)
        tail = tf.keras.Model(model.layers[partition_idx + 80].output, model.layers[len(model.layers) - 1].output)
    return head, tail
