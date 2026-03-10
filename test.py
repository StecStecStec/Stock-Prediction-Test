import pandas as pd

df = pd.read_csv("normalized_with_true_full_dataset.csv")
print(df.columns)

import tensorflow as tf
import keras
import tensorflow_addons as tfa

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("Addons:", tfa.__version__)
