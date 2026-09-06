import numpy as np
from PIL import Image, ImageOps


def prepare_image(image):
    image = ImageOps.exif_transpose(image)  # Respect camera orientation.
    image = image.convert("RGB")
    image = ImageOps.pad(
        image, (224, 224), method=Image.Resampling.BILINEAR,
        color=(0, 0, 0),
    )  # Preserve aspect ratio and add padding.
    return np.asarray(image, dtype=np.float32) / 255.0


# A synthetic input keeps the example runnable without downloading a photo.
example = Image.new("RGB", (320, 180), color=(80, 160, 240))
pixels = prepare_image(example)
print("Shape:", pixels.shape)  # (224, 224, 3): height, width, channels
print("Type:", pixels.dtype)  # float32
print("Range:", pixels.min(), pixels.max())  # Within [0, 1]
# For your photo:
# with Image.open("photo.jpg") as photo:
#     pixels = prepare_image(photo)
