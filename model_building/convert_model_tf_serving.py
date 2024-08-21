import tensorflow as tf
import numpy as np
from PIL import Image
import base64

MODEL_INPUT_SHAPE = (256,128)
MODEL_INPUTS_DTYPE = 'uint8'

# Load model that operates on 256x128 images
#my_custom_model = tf.keras.models.load_model(r'C:\Users\BKONG\OneDrive - Xylem Inc\Documents\ultra-ai\model_building\ultra-ai-image-model-2021-12-15.hdf5')
my_custom_model = tf.keras.models.load_model(r'C:\Users\BKONG\OneDrive - Xylem Inc\Documents\ultra-ai\attempt 13 weights\weights.32-0.22.hdf5')
print(my_custom_model._name)
my_custom_model._name="ultra-ai-64"

# Preprocess functions before 
def preprocess_input(base64_input_bytes): 
    def decode_bytes(img_bytes):
        img = tf.image.decode_png(img_bytes, channels=1)

        # Break up big transducer image into small strided 256x128 patches
        #img = tf.image.resize(img, MODEL_INPUT_SHAPE)  # Make inputs 256x128
        img = tf.image.convert_image_dtype(img, 'float32') # Automatically sets pixels between 0 and 1
        return img

    base64_input_bytes = tf.reshape(base64_input_bytes, (-1,))

    return tf.map_fn(lambda img_bytes:
                     decode_bytes(img_bytes),
                     elems=base64_input_bytes,                     
                     fn_output_signature='float32')


def patcher(x):
    # Break into patches with stride of 64
    patches = tf.image.extract_patches(images=x, sizes=[1, 256, 128, 1], strides=[1, 256, 64, 1], rates=[1, 1, 1, 1], padding='VALID')
    patches = tf.reshape(patches, (-1, 256, 128,1))
    return patches


inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name='b64_input_bytes')

# Lambda layer inputs
x = tf.keras.layers.Lambda(preprocess_input, name='decode_image_bytes')(inputs)
x = tf.keras.layers.Lambda(patcher)(x)

# Existing model
x = my_custom_model(x)

# Stack model
serving_model = tf.keras.Model(inputs, x)
serving_model.summary()

# Convert to TF PB format for serving
tf.saved_model.save(serving_model, './my_serving_model')
