import keras


def split_sequential_model(model, partition_idx):
    """
    Split the model VGG-16 before layer with index partition_idx.
    """
    partition_idx += 1
    head_module_list = list()
    tail_module_list = list()
    head_module_list.extend(model.layers[:partition_idx])
    tail_module_list.extend(model.layers[partition_idx:])

    head_network = keras.Sequential(head_module_list)
    # manually set input size of tail network
    tail_module_list.insert(0, keras.Input(shape=head_network.layers[-1].output_shape[1:]))
    tail_network = keras.Sequential(tail_module_list)
    return head_network, tail_network
