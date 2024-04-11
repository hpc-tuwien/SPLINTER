import keras


def split_functional_model(model, partition_idx):
    """
    Split the model ViT before layer/block with index partition_idx [1,18].
    """
    if 5 <= partition_idx <= 16:
        head = keras.Model(model.layers[0].output, model.layers[partition_idx].output[0])
        tail = keras.Model(model.layers[partition_idx].output[0], model.layers[len(model.layers) - 1].output)
    else:
        head = keras.Model(model.layers[0].output, model.layers[partition_idx].output)
        tail = keras.Model(model.layers[partition_idx].output, model.layers[len(model.layers) - 1].output)
    return head, tail
