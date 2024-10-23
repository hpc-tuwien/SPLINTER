import tensorflow as tf


def split_sequential_model(model, partition_idx):
    """
    Split the model VGG-16 before layer with index partition_idx.
    """
    partition_idx += 1
    head_module_list = list()
    tail_module_list = list()
    head_module_list.extend(model.layers[:partition_idx])
    tail_module_list.extend(model.layers[partition_idx:])

    head_network = tf.keras.Sequential(head_module_list)
    # manually set input size of tail network
    tail_module_list.insert(0, tf.keras.Input(shape=head_network.layers[-1].output_shape[1:]))
    tail_network = tf.keras.Sequential(tail_module_list)
    return head_network, tail_network


def split_functional_model(model, partition_idx):
    head = tf.keras.Model(model.layers[0].output, model.layers[partition_idx].output)
    tail = tf.keras.Model(model.layers[partition_idx].output, model.layers[len(model.layers) - 1].output)
    return head, tail
