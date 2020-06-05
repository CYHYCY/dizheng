import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2


def base64_PIL(base64_str):
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image)
    return image


def base64_PIL_BGR(base64_str):
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image)
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image
