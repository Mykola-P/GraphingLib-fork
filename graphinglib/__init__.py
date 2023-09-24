"""
# GraphingLib

An object oriented wrapper combining the functionalities of Matplotlib and Scipy.

Provides one-line commands for:
    1. creating 1D and 2D plottable objects like curve plots, scatter plots, etc.
    2. creating curve fits using any function (builtin or not)
    3. getting the derivative, integral or tangent (and many other) of a curve

Also provides the ability to create multiple visual styles for your plots and save
them to be used anytime you want.
"""

from .data_plotting_1d import Curve, Histogram, Scatter
from .data_plotting_2d import Heatmap, Contour, VectorField
from .figure import Figure
from .multifigure import Multifigure, Subfigure
from .fits import (
    FitFromExponential,
    FitFromGaussian,
    FitFromLog,
    FitFromPolynomial,
    FitFromSine,
    FitFromSquareRoot,
    FitFromFunction,
)
from .graph_elements import GraphingException, Hlines, Point, Text, Vlines
