from __future__ import annotations

from copy import deepcopy
from typing import Callable, Iterable, Literal, Optional, Self

import matplotlib.ticker as ticker
import numpy as np
from graph_elements import GraphingException
from matplotlib import colorbar
from matplotlib import colormaps as color_maps_list
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.contour import ContourSet
from matplotlib.figure import Figure
from matplotlib.image import AxesImage


class ColorBar:
    """
    This class implements a color bar object for plotting color bars associated with
    :class:`~graphinglib.graph_elements.Plottable` elements. It allows the creation
    of a color bar for a simple :class:`~graphinglib.smart_figure.SmartFigure` objects
    as well as a general color bar for more complex figures.

    Attributes
    ----------
    label : str, optional
        Label for the color bar.
    position : Literal["left", "right", "top", "bottom"]
        Position of the color bar relative to the parent SmartFigure.
        Default depends on the ``figure_style`` configuration.
    arrowed_ends : Literal["neither", "both", "lower_end", "upper_end"]
        Whether to draw pointed ends of the color bar.
        Defaults to ``"neither"``.
    arrowed_ends_length : float | tuple[float, float] | Literal["auto"], optional
        Length of the color bar's pointed ends. If a scalar, it sets the same length for
        both ends. When a two-element sequence is provided, it sets the lengths of the
        low-value and high-value ends respectively.
        Defaults to ``"auto"``.
    aspect : float | Literal["default"]
        Ratio of long to short dimensions.
        Default depends on the ``figure_style`` configuration.
    color_map : str | list[str] | Literal["default"]
        The color map used for :class:`~graphinglib.data_plotting_1d.Scatter`,
        :class:`~graphinglib.data_plotting_2d.Heatmap` and :class:`~graphinglib.data_plotting_2d.Contour`.
        It can be specified either as a string (a Matplotlib color or color map name) or as a list of strings
        (colors) to create a discrete color map.
        Default depends on the ``figure_style`` configuration.
    min_value, max_value : float, optional
        The data range that the color map will cover. By default, the color map spans the full range of the
        supplied data.
    fraction : float
        Fraction by which to multiply the size of the color bar.
        Defaults to ``1``.
    show : bool
        Whether to display the color bar.
        Defaults to ``True``.
    """

    def __init__(
        self,
        label: Optional[str] = None,
        position: Literal["left", "right", "top", "bottom", "default"] = "default",
        arrowed_ends: Literal["neither", "both", "lower_end", "upper_end"] = "neither",
        arrowed_ends_length: Optional[float | tuple[float, float] | Literal["auto"]] = None,
        aspect: float | Literal["default"] = "default",
        color_map: str | list[str] | Literal["default"] = "default",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        fraction: float = 1,
        show: bool = True,
    ) -> None:
        self.label = label
        self.position = position
        self.arrowed_ends = arrowed_ends
        self.arrowed_ends_length = arrowed_ends_length
        self.aspect = aspect
        self.min_value = min_value
        self.max_value = max_value
        self.fraction = fraction
        self.show = show

        self.color_map = (
            color_map if isinstance(color_map, str) and color_map in color_maps_list else ListedColormap(color_map)
        )

        self._ticks = {}
        self._tick_params = {"major": {}, "minor": {}}

        self._edge_color = None
        self._edge_width = None
        self._hidden_spines = None
        self._user_rc_dict = {}

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    @property
    def position(self) -> str:
        return self._position

    @position.setter
    def position(self, value: str) -> None:
        self._position = value

    @property
    def arrowed_ends(self) -> str:
        return self._arrowed_ends

    @arrowed_ends.setter
    def arrowed_ends(self, value: str) -> None:
        self._arrowed_ends = value

    @property
    def arrowed_ends_length(self) -> str | float | tuple[float, float]:
        return self._arrowed_ends_length

    @arrowed_ends_length.setter
    def arrowed_ends_length(self, value: float | tuple[float, float] | Literal["auto"]) -> None:
        self._arrowed_ends_length = value

    @property
    def aspect(self) -> float:
        return self._aspect

    @aspect.setter
    def aspect(self, value: float) -> None:
        self._aspect = value

    @property
    def color_map(self) -> str | ListedColormap:
        return self._color_map

    @color_map.setter
    def color_map(self, value: str | list[str]) -> None:
        self._color_map = value if isinstance(value, str) and value in color_maps_list else ListedColormap(value)

    @property
    def min_value(self) -> float:
        return self._min_value

    @min_value.setter
    def min_value(self, value: float) -> None:
        self._min_value = value

    @property
    def max_value(self) -> float:
        return self._max_value

    @max_value.setter
    def max_value(self, value: float) -> None:
        self._max_value = value

    @property
    def fraction(self) -> float:
        return self._fraction

    @fraction.setter
    def fraction(self, value: float) -> None:
        self._fraction = value

    @property
    def show(self) -> bool:
        return self._show

    @show.setter
    def show(self, value: bool) -> None:
        self._show = value

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.color_bar.ColorBar` object.
        """
        return deepcopy(self)

    def copy_with(self, **kwargs) -> Self:
        """
        TODO: Docstring.
        """
        # TODO: Implement the method.
        raise (NotImplemented)

    def set_visual_params(
        self,
        reset: bool = False,
        edge_color: Optional[str] = None,
        edge_width: Optional[float] = None,
        label_color: Optional[str] = None,
        label_pad: Optional[float] = None,
        font_family: Optional[str] = None,
        font_size: Optional[float] = None,
        font_weight: Optional[str] = None,
        use_latex: Optional[bool] = None,
        hidden_spines: Optional[Iterable[Literal["right", "left", "top", "bottom"]]] = None,
    ) -> Self:
        """
        Customize the visual style of the :class:`~graphinglib.color_bar.ColorBar`.

        Any parameter that is not specified (None) will be set to the default value for the specified ``figure_style``
        from the parent :class:`~graphinglib.smart_figure.SmartFigure`.

        Parameters
        ----------
        reset : bool
            Whether or not to reset the rc parameters to the default values for the specified ``figure_style``.
            Defaults to ``False``.
        edge_color : str, optional
            The color of the color bar's edge.
        edge_width : float, optional
            The width of the color bar's edge.
        label_color : str, optional
            The color of the color bar's label.
        label_pad : float, optional
            The padding between the color bar the its label.
        font_family : str, optional
            The font family to use.
        font_size : float, optional
            The font size to use.
        font_weight : str, optional
            The font weight to use.
        use_latex : bool, optional
            Whether or not to use latex.
        hidden_spines : Iterable[Literal["right", "left", "top", "bottom"]],
           The spines to hide. If specified, the corresponding spines will be hidden on the color bar. These spines
           correspond to the lines that form the borders of the color bar.

        Returns
        -------
        Self
            For convenience, the same ColorBar with the updated visual parameters.
        """
        rc_params_dict = {
            "axes.labelcolor": label_color,
            "axes.labelpad": label_pad,
            "font.family": font_family,
            "font.size": font_size,
            "font.weight": font_weight,
            "text.usetex": use_latex,
        }

        if reset:
            self._edge_color = edge_color
            self._edge_width = edge_width
            for key in rc_params_dict.keys():
                self._user_rc_dict.pop(key, None)
        else:
            if edge_color is not None:
                self._edge_color = edge_color
            if edge_width is not None:
                self._edge_width = edge_width

        rc_params_dict = {key: value for key, value in rc_params_dict.items() if value is not None}
        self.set_rc_params(rc_params_dict)

        if hidden_spines is not None:
            if not isinstance(hidden_spines, Iterable):
                raise TypeError("hidden_spines must be an iterable of spine names.")
            for spine in hidden_spines:
                if spine not in ["right", "left", "top", "bottom"]:
                    raise ValueError(f"Invalid spine name: {spine}. Must be one of 'right', 'left', 'top' or 'bottom'.")
            self._hidden_spines = hidden_spines

        return self

    def set_rc_params(
        self,
        rc_params_dict: dict[str, str | float] = {},
        reset: bool = False,
    ) -> Self:
        """
        Customize the visual style of the :class:`~graphinglib.color_bar.ColorBar`.

        Any rc parameter that is not specified in the dictionary will be set to the default value for the specified
        ``figure_style`` from the parent :class:`~graphinglib.smart_figure.SmartFigure`.

        Parameters
        ----------
        rc_params_dict : dict[str, str | float]
            Dictionary of rc parameters to update.
            Defaults to empty dictionary.
        reset : bool
            Whether or not to reset the rc parameters to the default values.
            Defaults to ``False``.

        Returns
        -------
        Self
            For convenience, the same ColorBar with the updated rc parameters.
        """
        if reset:
            self._user_rc_dict.clear()
            self._edge_color = None
            self._edge_width = None
        for property_, value in rc_params_dict.items():
            self._user_rc_dict[property_] = value
        return self

    def set_ticks(
        self,
        reset: bool = False,
        ticks: Optional[Iterable[float]] = None,
        tick_labels: Optional[Iterable[str]] = None,
        tick_spacing: Optional[float] = None,
        minor_ticks: Optional[Iterable[float]] = None,
        minor_tick_spacing: Optional[float] = None,
        format: Optional[Callable | str] = None,
    ) -> Self:
        """
        Sets custom ticks and tick labels.

        Parameters
        ----------
        reset : bool
            Whether to reset the tick parameters to their default values.
            Defaults to ``False``.
        ticks : Iterable[float], optional
            Tick positions for the color bar. If a value is specified, the ``tick_spacing`` parameter must be ``None``.
        tick_labels : Iterable[str], optional
            Tick labels for the color bar. If a value is specified, the ``ticks`` parameter must also be given. The
            number of tick labels must match the number of ticks.
        tick_spacing : float, optional
            Spacing between major ticks on the color bar.
        minor_ticks : Iterable[float], optional
            Minor tick positions for the color bar.
        minor_tick_spacing : float, optional
            Spacing between minor ticks on the color bar.
        format : Callable | str, optional
            A string or callable function to format the tick labels of the color bar. The format could be defined as
            ``"%4.2e"`` or ``"{x:.2e}"``, or as follows::

            TODO: Docstring!!!

        Returns
        -------
        Self
            For convenience, the same ColorBar with the updated ticks.
        """
        if (tick_labels is not None) and ticks is None:
            raise GraphingException("Ticks position must be specified when ticks labels are specified.")

        if any(
            [
                (ticks is not None) and (tick_spacing is not None),
                (minor_ticks is not None) and (minor_tick_spacing is not None),
            ]
        ):
            raise GraphingException("Tick spacing and tick positions cannot be set simultaneously.")

        if ticks is not None and tick_labels is not None:
            if len(ticks) != len(tick_labels):
                raise GraphingException(
                    f"Number of ticks ({len(ticks)}) and number of tick labels ({len(tick_labels)}) must be the same."
                )

        if format is not None and not isinstance(format, (Callable, str)):
            raise GraphingException(f"The format of ticks must be a function or a string")

        if reset:
            self._ticks.clear()

        params = [
            "ticks",
            "tick_labels",
            "tick_spacing",
            "minor_ticks",
            "minor_tick_spacing",
            "format",
        ]
        for param in params:
            value = locals()[param]
            if value is not None:
                self._ticks[param] = value

        return self

    def set_ticks_params(
        self,
        reset: Optional[bool] = False,
        which: Optional[Literal["major", "minor", "both"]] = "major",
        direction: Optional[Literal["in", "out", "inout"]] = None,
        length: Optional[float] = None,
        width: Optional[float] = None,
        color: Optional[str] = None,
        pad: Optional[float] = None,
        label_size: Optional[float | str] = None,
        label_color: Optional[str] = None,
        label_rotation: Optional[float] = None,
        draw_ticks: Optional[bool] = None,
        draw_labels: Optional[bool] = None,
    ) -> Self:
        """
        Sets the tick parameters. These parameters are given to the :meth:`matplotlib.axes.Axes.tick_params` method.

        Parameters
        ----------
        reset : bool, optional
            If ``True``, all previously given tick parameters are reset to their default values before applying the new
            parameters.
            Defaults to ``False``
        which : Literal["major", "minor", "both"], optional
            The ticks to set the parameters for. This method can be called multiple times to set the tick parameters
            specifically for each ticks type.
            Defaults to ``"major"``.
        direction : Literal["in", "out", "inout"], optional
            The direction of the ticks.
        length : float, optional
            The length of the ticks.
        width : float, optional
            The width of the ticks.
        color : str, optional
            The color of the ticks.
        pad : float, optional
            The padding to add between the tick labels and the ticks themselves.
        label_size : float | str, optional
            The font size of the tick labels. This can be a float or a string (e.g. "large").
        label_color : str, optional
            The color of the tick labels.
        label_rotation : float, optional
            The rotation of the tick labels, in degrees.
        draw_ticks : bool, optional
            Whether to draw the ticks of the axis.
        draw_labels : bool, optional
            Whether to draw the tick labels of the axis.

        Returns
        -------
        Self
            For convenience, the same ColorBar with the updated tick parameters.
        """
        new_tick_params = {
            "direction": direction,
            "length": length,
            "width": width,
            "color": color,
            "pad": pad,
            "labelsize": label_size,
            "labelcolor": label_color,
            "labelrotation": label_rotation,
            "bottom": draw_ticks,
            "top": draw_ticks,
            "left": draw_ticks,
            "right": draw_ticks,
            "labelbottom": draw_labels,
            "labeltop": draw_labels,
            "labelleft": draw_labels,
            "labelright": draw_labels,
        }

        for which_i in [which] if which != "both" else ["major", "minor"]:
            if reset:
                self._tick_params[which_i].clear()
            for param, value in new_tick_params.items():
                if value is not None:
                    self._tick_params[which_i][param] = value

        return self

    def _customize_ticks(self, is_vertical: bool) -> None:
        """
        Customizes the ticks of the specified color bar.

        Parameters
        ----------
        is_vertical : bool
            If True, the color bar is oriented vertically; if False, the color bar is oriented horizontally.
        """
        if is_vertical:
            ax_set_ticks, axis_str, ax_axis = (
                self._color_bar.ax.set_yticks,
                "y",
                self._color_bar.ax.yaxis,
            )
        else:
            ax_set_ticks, axis_str, ax_axis = (
                self._color_bar.ax.set_xticks,
                "x",
                self._color_bar.ax.xaxis,
            )

        if self._ticks.get("ticks") is not None:
            ax_set_ticks(self._ticks.get("ticks"), self._ticks.get("tick_labels"))

        self._color_bar.ax.tick_params(axis=axis_str, which="major", **self._tick_params["major"])

        if self._ticks.get("tick_spacing") is not None:
            ax_axis.set_major_locator(ticker.MultipleLocator(self._ticks.get("tick_spacing")))

        if self._ticks.get("minor_ticks") is not None:
            ax_set_ticks(self._ticks.get("minor_ticks"), minor=True)

        self._color_bar.ax.tick_params(axis=axis_str, which="minor", **self._tick_params["minor"])

        if self._ticks.get("minor_tick_spacing") is not None:
            ax_axis.set_minor_locator(ticker.MultipleLocator(self._ticks.get("minor_tick_spacing")))

    def _plot_color_bar(
        self,
        figure: Figure,
        mappable: ScalarMappable | AxesImage | ContourSet,
    ) -> Self:
        """
        TODO: Docstring.
        """
        ax = figure.get_axes()[0]
        is_aspect_auto = ax.get_aspect() == "auto"

        color_bar_params = {
            "label": self._label,
            "extend": self._arrowed_ends,
            "extendfrac": self._arrowed_ends_length,
        }

        if is_aspect_auto:
            color_bar_params.update(
                {
                    "location": self._position,
                    "shrink": self._fraction,
                    "aspect": self._aspect,
                }
            )

        color_bar_params = {k: v for k, v in color_bar_params.items() if v != "default"}

        # Scale the color bar to match the height (if vertical) or width (if horizontal) of the axes
        if is_aspect_auto:
            self._color_bar = figure.colorbar(
                mappable,
                ax=figure.get_axes()[0],
                format=self._ticks.get("format"),
                **color_bar_params,
            )

        else:
            color_bar_ax = self._create_color_bar_axe(figure)

            self._color_bar = figure.colorbar(
                mappable,
                cax=color_bar_ax,
                format=self._ticks.get("format"),
                **color_bar_params,
            )

            # figure.canvas.draw()
            #
            # if self.position in ["left", "right"]:
            #     y_pos = []
            #     y_pos.extend(
            #         [
            #             getattr(
            #                 ax.get_position().transformed(ax.get_figure().transSubfigure - figure.transFigure),
            #                 attr,
            #             )
            #             for attr in ["y0", "y1"]
            #         ]
            #     )
            #
            #     x0 = 1.02 if self.position == "right" else -0.04
            #     y0 = np.min(y_pos)
            #     y1 = np.max(y_pos)
            #
            #     height = self.fraction * (y1 - y0)
            #     y0 += height * (1 - self.fraction) / self.fraction / 2
            #
            #     cax = figure.add_axes((x0, y0, height / self.aspect, height))
            #
            #
            # #     color_bar_params["fraction"] = (
            # #         0.05 * figure.bbox.height * ax.get_position().height / ax.get_position().width / figure.bbox.width
            # #     )
            # #
            # # else:
            # #     color_bar_params["fraction"] = (
            # #         0.05 * figure.bbox.width * ax.get_position().width / ax.get_position().height / figure.bbox.height
            # #     )

        self._customize_ticks(is_vertical=self.position in ["left", "right"])

        return self

    def _plot_general_color_bar(
        self,
        figure: Figure,
        mappable: ScalarMappable | AxesImage | ContourSet,
    ) -> Self:
        """
        TODO: Docstring.
        """
        color_bar_params = {
            "label": self._label,
            "extend": self._arrowed_ends,
            "extendfrac": self._arrowed_ends_length,
        }

        color_bar_params = {k: v for k, v in color_bar_params.items() if v != "default"}

        color_bar_ax = self._create_color_bar_axe(figure)

        self._color_bar = figure.colorbar(
            mappable,
            cax=color_bar_ax,
            format=self._ticks.get("format"),
            **color_bar_params,
        )
        self._customize_ticks(is_vertical=self.position in ["left", "right"])

        return self

    def _create_color_bar_axe(self, figure: Figure, ax: Optional[Axes] = None) -> Axes:
        """
        TODO: Docstring

        # Scale the color bar to match the height (if vertical) or width (if horizontal) of the axes
        Parameters
        ----------
        figure : Figure

        ax : Axes, optional


        Returns
        -------
        Axes

        """
        figure.canvas.draw()

        if self.position in ["left", "right"]:
            y_pos = []
            for ax in [ax] if ax is not None else figure.get_axes():
                y_pos.extend(
                    [
                        getattr(
                            ax.get_position().transformed(ax.get_figure().transSubfigure - figure.transFigure),
                            attr,
                        )
                        for attr in ["y0", "y1"]
                    ]
                )

            x0 = 1.02 if self.position == "right" else -0.04
            y0 = np.min(y_pos)
            y1 = np.max(y_pos)

            height = self.fraction * (y1 - y0)
            y0 += height * (1 - self.fraction) / self.fraction / 2

            color_bar_ax = figure.add_axes((x0, y0, height / self.aspect, height))

        else:
            x_pos = []
            for ax in [ax] if ax is not None else figure.get_axes():
                x_pos.extend(
                    [
                        getattr(
                            ax.get_position().transformed(ax.get_figure().transSubfigure - figure.transFigure),
                            attr,
                        )
                        for attr in ["x0", "x1"]
                    ]
                )

            x0 = np.min(x_pos)
            x1 = np.max(x_pos)
            y0 = 1.02 if self.position == "top" else -0.04

            width = self.fraction * (x1 - x0)
            x0 += width * (1 - self.fraction) / self.fraction / 2

            color_bar_ax = figure.add_axes((x0, y0, width, width / self.aspect))

        return color_bar_ax
