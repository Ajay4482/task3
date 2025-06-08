# task3
#intenship

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

def load_image(path):
    img = PIL.Image.open(path).resize((256, 256))
    img = np.array(img)/255.0
    img = img[np.newaxis, ...]
    return img

def show_image(image):
    image = image[0]
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def style_transfer(content_path, style_path):
    content_image = load_image(content_path)
    style_image = load_image(style_path)
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    show_image(stylized_image)

