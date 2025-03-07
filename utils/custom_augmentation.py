import numpy as np
import cv2
from PIL import Image

# Custom HSV Transform for PyTorch
class HSVTransform:
    def __init__(self, h_gain=0.015, s_gain=0.7, v_gain=0.4):
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def __call__(self, img):
        # Convert PIL Image to NumPy
        img = np.array(img, dtype=np.uint8)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Apply random HSV augmentation
        h_shift = np.random.uniform(-self.h_gain, self.h_gain) * 180
        s_scale = np.random.uniform(1 - self.s_gain, 1 + self.s_gain)
        v_scale = np.random.uniform(1 - self.v_gain, 1 + self.v_gain)

        img_hsv[..., 0] = (img_hsv[..., 0] + h_shift) % 180  # Hue shift
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] * s_scale, 0, 255)  # Saturation scale
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] * v_scale, 0, 255)  # Brightness scale

        # Convert back to RGB and then to PIL Image
        img_hsv = img_hsv.astype(np.uint8)
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(img_rgb)

