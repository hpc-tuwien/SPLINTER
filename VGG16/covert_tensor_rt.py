import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tqdm import tqdm

for partition_index in tqdm(range(22)):
    # load keras model
    model = keras.models.load_model("models/tail/" + str(partition_index) + ".keras")
    # temporary save as saved model
    model.save("models/tmp/" + str(partition_index))

    # Instantiate the TF-TRT converter
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir="models/tmp/" + str(partition_index),
        precision_mode=trt.TrtPrecisionMode.FP32
    )

    # Convert the model into TRT compatible segments
    converter.convert()

    # Save the model to the disk
    converter.save("models/tensorrt/" + str(partition_index))
