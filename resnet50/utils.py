import tensorflow as tf


def split_functional_model(model, partition_idx):
    """
    Split the model ResNet50 before layer/block with index partition_idx [1,39].
    """
    if partition_idx <= 6:
        head = tf.keras.Model(model.layers[0].output, model.layers[partition_idx].output)
        tail = tf.keras.Model(model.layers[partition_idx].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 7:
        head = tf.keras.Model(model.layers[0].output, model.layers[17].output)
        tail = tf.keras.Model(model.layers[17].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 8:
        head = tf.keras.Model(model.layers[0].output, model.layers[18].output)
        tail = tf.keras.Model(model.layers[18].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 9:
        head = tf.keras.Model(model.layers[0].output, model.layers[27].output)
        tail = tf.keras.Model(model.layers[27].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 10:
        head = tf.keras.Model(model.layers[0].output, model.layers[28].output)
        tail = tf.keras.Model(model.layers[28].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 11:
        head = tf.keras.Model(model.layers[0].output, model.layers[37].output)
        tail = tf.keras.Model(model.layers[37].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 12:
        head = tf.keras.Model(model.layers[0].output, model.layers[38].output)
        tail = tf.keras.Model(model.layers[38].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 13:
        head = tf.keras.Model(model.layers[0].output, model.layers[49].output)
        tail = tf.keras.Model(model.layers[49].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 14:
        head = tf.keras.Model(model.layers[0].output, model.layers[50].output)
        tail = tf.keras.Model(model.layers[50].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 15:
        head = tf.keras.Model(model.layers[0].output, model.layers[59].output)
        tail = tf.keras.Model(model.layers[59].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 16:
        head = tf.keras.Model(model.layers[0].output, model.layers[60].output)
        tail = tf.keras.Model(model.layers[60].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 17:
        head = tf.keras.Model(model.layers[0].output, model.layers[69].output)
        tail = tf.keras.Model(model.layers[69].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 18:
        head = tf.keras.Model(model.layers[0].output, model.layers[70].output)
        tail = tf.keras.Model(model.layers[70].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 19:
        head = tf.keras.Model(model.layers[0].output, model.layers[79].output)
        tail = tf.keras.Model(model.layers[79].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 20:
        head = tf.keras.Model(model.layers[0].output, model.layers[80].output)
        tail = tf.keras.Model(model.layers[80].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 21:
        head = tf.keras.Model(model.layers[0].output, model.layers[91].output)
        tail = tf.keras.Model(model.layers[91].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 22:
        head = tf.keras.Model(model.layers[0].output, model.layers[92].output)
        tail = tf.keras.Model(model.layers[92].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 23:
        head = tf.keras.Model(model.layers[0].output, model.layers[101].output)
        tail = tf.keras.Model(model.layers[101].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 24:
        head = tf.keras.Model(model.layers[0].output, model.layers[102].output)
        tail = tf.keras.Model(model.layers[102].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 25:
        head = tf.keras.Model(model.layers[0].output, model.layers[111].output)
        tail = tf.keras.Model(model.layers[111].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 26:
        head = tf.keras.Model(model.layers[0].output, model.layers[112].output)
        tail = tf.keras.Model(model.layers[112].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 27:
        head = tf.keras.Model(model.layers[0].output, model.layers[121].output)
        tail = tf.keras.Model(model.layers[121].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 28:
        head = tf.keras.Model(model.layers[0].output, model.layers[122].output)
        tail = tf.keras.Model(model.layers[122].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 29:
        head = tf.keras.Model(model.layers[0].output, model.layers[131].output)
        tail = tf.keras.Model(model.layers[131].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 30:
        head = tf.keras.Model(model.layers[0].output, model.layers[132].output)
        tail = tf.keras.Model(model.layers[132].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 31:
        head = tf.keras.Model(model.layers[0].output, model.layers[141].output)
        tail = tf.keras.Model(model.layers[141].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 32:
        head = tf.keras.Model(model.layers[0].output, model.layers[142].output)
        tail = tf.keras.Model(model.layers[142].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 33:
        head = tf.keras.Model(model.layers[0].output, model.layers[153].output)
        tail = tf.keras.Model(model.layers[153].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 34:
        head = tf.keras.Model(model.layers[0].output, model.layers[154].output)
        tail = tf.keras.Model(model.layers[154].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 35:
        head = tf.keras.Model(model.layers[0].output, model.layers[163].output)
        tail = tf.keras.Model(model.layers[163].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx == 36:
        head = tf.keras.Model(model.layers[0].output, model.layers[164].output)
        tail = tf.keras.Model(model.layers[164].output, model.layers[len(model.layers) - 1].output)
    elif partition_idx >= 37:
        head = tf.keras.Model(model.layers[0].output, model.layers[136 + partition_idx].output)
        tail = tf.keras.Model(model.layers[136 + partition_idx].output, model.layers[len(model.layers) - 1].output)
    return head, tail
