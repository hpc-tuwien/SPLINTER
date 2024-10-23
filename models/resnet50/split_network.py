# %%

import tensorflow as tf
from tqdm import tqdm

from utils import split_functional_model

# %%
model = tf.keras.applications.ResNet50(weights='imagenet')
# %%
print("Save full models")
model.save("models/tail/0")
# %%
print("Save partial models")
# skip full model with first and last index
for i in tqdm(range(1, 40)):
    head, tail = split_functional_model(model, i)
    tail.save("models/tail/" + str(i))
