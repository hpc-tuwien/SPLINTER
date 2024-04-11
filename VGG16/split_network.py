# %%

from keras.src.applications import VGG16
from tqdm import tqdm

from utils import split_sequential_model

# %%
model = VGG16(weights='imagenet')
# %%
print("Save full models")
model.save("models/tail/0")
# %%
print("Save partial models")
# skip full model with first and last index
for i in tqdm(range(1, len(model.layers) - 1)):
    head, tail = split_sequential_model(model, i)
    tail.save("models/tail/" + str(i))
