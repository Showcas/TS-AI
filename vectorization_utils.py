import xml.etree.ElementTree as ET
import potrace
import numpy as np
from torch import Tensor
import io
from svgelements import SVG

from torch_utils import tensor_image_to_array_image


def _preprocess(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses the input image for vectorization. 
    Args:
        image (np.ndarray): The input image as a numpy array.
    Returns:
        np.ndarray: The preprocessed binary image.
    """
    bw = np.asarray(image).copy()
    bw[bw < 0.5] = 0  # Black
    bw[bw >= 0.5] = 1  # White
    return bw


def _createSVGpath(path) -> str:
    """
    Creates an SVG path description from a Potrace path object.
    Args:
        path: Potrace path object.
    Returns:
        str: SVG path description.
    """
    path_desc = ""
    # Iterate over path curves
    for curve in path:
        path_desc = (
            path_desc
            + " M "
            + str(curve.start_point[0])
            + ","
            + str(curve.start_point[1])
        )
        for segment in curve:
            if segment.is_corner:
                path_desc = (
                    path_desc
                    + " L "
                    + str(segment.c[0])
                    + ","
                    + str(segment.c[1])
                    + " L "
                    + str(segment.end_point[0])
                    + ","
                    + str(segment.end_point[1])
                )
            else:
                path_desc = (
                    path_desc
                    + " C "
                    + str(segment.c1[0])
                    + ","
                    + str(segment.c1[1])
                    + " "
                    + str(segment.c2[0])
                    + ","
                    + str(segment.c2[1])
                    + " "
                    + str(segment.end_point[0])
                    + ","
                    + str(segment.end_point[1])
                )
    return path_desc + " Z"


def _create_svg_xml(desc: str) -> str:
    """
    Creates an SVG XML string from a path description.
    Args:
        desc (str): The SVG path description.
    Returns:
        str: The SVG XML string.
    """
    root = ET.Element("svg")
    root.set("version", "1.0")
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("height", str(28))
    root.set("width", str(28))
    path = ET.SubElement(root, "path")
    path.set("d", desc)
    tree = ET.ElementTree(root)
    tree = tree.getroot()
    xml_str = ET.tostring(tree, encoding="unicode", method="xml")
    return xml_str


def vectorize(tensor_image: Tensor) -> str:
    """
    Vectorizes a tensor image into an SVG object.
    Args:
        tensor_image (Tensor): The input tensor image to be vectorized.
    Returns:
        SVG: The vectorized SVG object.
    """
    
    array_image = tensor_image_to_array_image(tensor_image=tensor_image)

    array = _preprocess(image=array_image)
    # use Potrace lib to obtain a SVG path from a Bitmap
    # Create a bitmap from the array
    bmp = potrace.Bitmap(array)
    # Trace the bitmap to a path
    path = bmp.trace()
    desc = _createSVGpath(path)

    svg_str = _create_svg_xml(desc)
    svg_object = SVG.parse(io.StringIO(svg_str))
    return svg_object
