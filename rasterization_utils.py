from typing import Tuple
import numpy as np
from PIL import Image
import cairosvg
import io

from torch import Tensor

from torch_utils import array_image_to_tensor_image


def rasterize(svg_str: str) -> Tuple[np.ndarray, Tensor]:
    """
    Rasterizes an SVG string into a numpy array and a tensor image.
    Args:
        svg_str (str): The input SVG string.
    Returns:
        Tuple[np.ndarray, Tensor]: A tuple containing the rasterized image as a numpy array and as a tensor.
    """
    
    png_bytes = cairosvg.svg2png(bytestring=svg_str)
    image = Image.open(io.BytesIO(png_bytes))

    array_image = np.array(image, dtype=np.float32)
    # take only the last channel (the conversion with PIL results in an array with 3 channels)
    array_image = array_image[:, :, 3]
    # normalize
    array_image /= 255.0

    tensor_image = array_image_to_tensor_image(array_image=array_image)

    return array_image, tensor_image
