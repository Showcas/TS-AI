from typing import Tuple
import numpy as np
from svgelements import SVG, Path, Line, QuadraticBezier, CubicBezier, Arc
import matplotlib.pyplot as plt
from torch import Tensor
from copy import deepcopy
import random
import math

from rasterization_utils import rasterize
from torch_utils import tensor_image_to_array_image
from vectorization_utils import vectorize


class Individual:

    def __init__(
        self,
        tensor_image: Tensor,
        svg_object: SVG = None,
    ) -> None:
        if svg_object is None:
            self.svg_object = vectorize(tensor_image=tensor_image)
        else:
            self.svg_object = svg_object
        self.tensor_image = tensor_image

    # docs for the library: https://github.com/meerk40t/svgelements/
    # svg editor: https://yqnn.github.io/svg-path-editor/https://yqnn.github.io/svg-path-editor/
    def mutate(self, extent: float = 1.0) -> "Individual":
        """
        Mutate the individual by applying random changes to its SVG representation.
        Args:
            extent (float): The extent of the mutation.
        Returns:
            Individual: The mutated individual.
        """
        mutated_svg = deepcopy(self.svg_object)

        for element in mutated_svg.elements():
            if isinstance(element, Path):
                for segment in element:
                    if isinstance(segment, Line):
                        segment.start += complex(
                            random.uniform(-extent, extent),
                            random.uniform(-extent, extent),
                        )
                        segment.end += complex(
                            random.uniform(-extent, extent),
                            random.uniform(-extent, extent),
                        )
                    elif isinstance(segment, CubicBezier):
                        for p in [segment.start, segment.end,
                                  segment.control1, segment.control2]:
                            p += complex(
                                random.uniform(-extent, extent),
                                random.uniform(-extent, extent),
                            )
                    elif isinstance(segment, QuadraticBezier):
                        for p in [segment.start, segment.end,
                                  segment.control]:
                            p += complex(
                                random.uniform(-extent, extent),
                                random.uniform(-extent, extent),
                            )
                    elif isinstance(segment, Arc):
                        # small mutation of center, radius, rotation
                        segment.center += complex(
                            random.uniform(-extent, extent),
                            random.uniform(-extent, extent),
                        )
                        segment.radius = (
                            segment.radius[0] + random.uniform(-extent, extent),
                            segment.radius[1] + random.uniform(-extent, extent),
                        )
                        segment.rotation += random.uniform(-extent, extent)

        # Return a new Individual with the mutated SVG
        return Individual(tensor_image=self.tensor_image, svg_object=mutated_svg) 

    def get_str_representation(self) -> str:
        """
        Get the SVG string representation of the individual.
        Returns:
            str: The SVG string representation.
        """
        return self.svg_object.string_xml()

    def get_image_array_and_tensor_representation(self) -> Tuple[np.ndarray, Tensor]:
        """
        Get the rasterized image representations (array and tensor) of the individual.
        Returns:
            Tuple[np.ndarray, Tensor]: A tuple containing the rasterized image as a numpy array and as a tensor.
        """
        array_image, tensor_image = rasterize(svg_str=self.get_str_representation())
        return array_image, tensor_image

    def plot_svg_representation(self, filename: str) -> None:
        """
        Save the SVG representation of the individual to a file.
        Args:
            filename (str): The filename to save the SVG representation.
        Returns:
            None
        """
        with open(f"{filename}.svg", "w") as f:
            f.write(self.get_str_representation())

    def plot_current_image(self, filename: str) -> None:
        """
        Save the current rasterized image representation of the individual to a file.
        Args:
            filename (str): The filename to save the rasterized image.
        Returns:
            None
        """
        array_image, _ = self.get_image_array_and_tensor_representation()
        self._plot_image(array_image=array_image, filename=filename)

    @staticmethod
    def _plot_image(array_image: np.ndarray, filename: str) -> None:
        """
        Save the given array image to a file.
        Args:
            array_image (np.ndarray): The image as a numpy array.
            filename (str): The filename to save the image.
        Returns:
            None
        """
        plt.imshow(array_image, cmap="grey")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{filename}.png")
        plt.close()

    def plot_original_image(self, filename: str) -> None:
        """
        Save the original tensor image representation of the individual to a file.
        Args:
            filename (str): The filename to save the original image.
        Returns:
            None
        """
        array_image = tensor_image_to_array_image(tensor_image=self.tensor_image)
        self._plot_image(array_image=array_image, filename=filename)

    def get_boldness_feature_value(self, threshold: float = 0.5) -> int:
        """
        Get the boldness feature value, defined as the number of pixels
        in the array image that are above a certain threshold.
        Args:
            threshold (float): The threshold to consider a pixel as "bold".
        Returns:
            int: The boldness feature value.
        """

        array_image, _ = self.get_image_array_and_tensor_representation()

        return int((array_image > threshold).sum())

    def get_discontinuity_feature_value(self) -> int:
        """
        Get the discontinuity feature value, defined as the sum of Euclidean distances
        between pairs of consecutive sections of the digit (see Move).
        Returns:
            int: The discontinuity feature value.
        """
        total_distance = 0.0
        delta_threshold = 1e-6

        # get first Path object
        path = None
        for element in self.svg_object.elements():
            if isinstance(element, Path):
                path = element
                break

        if path is None:
            return 0.0

        previous_point = None

        for segment in path:
            # start point of this segment
            current_point = segment.start

            if previous_point is not None:
                dx = current_point.real - previous_point.real
                dy = current_point.imag - previous_point.imag
                distance = math.hypot(dx, dy)

                if distance > delta_threshold:
                    total_distance += distance

            # end point becomes the "previous_point" for next loop iteration
            previous_point = segment.end

        return total_distance
