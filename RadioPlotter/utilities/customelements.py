"""
Custom elements which modify matplotlib elements.
"""

import numpy as np
from matplotlib import cbook
from matplotlib.widgets import AxesWidget, RadioButtons


class MyRadioButtons(RadioButtons):
    def __init__(
        self,
        ax,
        labels,
        active=0,
        activecolor="blue",
        size=49,
        orientation="vertical",
        **kwargs,
    ):
        """
        Add radio buttons to an `~.axes.Axes`.
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The axes to add the buttons to.
        labels : list of str
            The button labels.
        active : int
            The index of the initially selected button.
        activecolor : color
            The color of the selected button.
        size : float
            Size of the radio buttons
        orientation : str
            The orientation of the buttons: 'vertical' (default), or 'horizontal'.
        Further parameters are passed on to `Legend`.

        Notes
        -----
        Reference:
        https://stackoverflow.com/questions/55095111/displaying-radio-buttons-horizontally-in-matplotlib

        """
        AxesWidget.__init__(self, ax)
        self.activecolor = activecolor
        axcolor = ax.get_facecolor()
        self.value_selected = None

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)

        circles = []
        for i, label in enumerate(labels):
            if i == active:
                self.value_selected = label
                facecolor = activecolor
            else:
                facecolor = axcolor
            p = ax.scatter(
                [],
                [],
                s=size,
                marker="o",
                edgecolor="black",
                facecolor=facecolor,
            )
            circles.append(p)
        if orientation == "horizontal":
            kwargs.update(ncol=np.where(len(labels) > 6, 6, len(labels)), mode="expand")
        kwargs.setdefault("frameon", False)
        self.box = ax.legend(circles, labels, loc="center", **kwargs)
        self.labels = self.box.texts
        self.circles = self.box.legendHandles
        for c in self.circles:
            c.set_picker(5)

        self._observers = cbook.CallbackRegistry()

        self.connect_event("pick_event", self._clicked)

    def _clicked(self, event):
        if (
            self.ignore(event)
            or event.mouseevent.button != 1
            or event.mouseevent.inaxes != self.ax
        ):
            return
        if event.artist in self.circles:
            self.set_active(self.circles.index(event.artist))
