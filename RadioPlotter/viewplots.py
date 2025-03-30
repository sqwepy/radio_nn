"""
Plot Viewer which sees two scalar functions against each other.
"""

import numpy as np
import matplotlib.pyplot as plt

from RadioPlotter.utilities.customelements import MyRadioButtons
from RadioPlotter.viewevent import get_attributes, get_default_key


class PlotViewer:
    def __init__(self, fig, ax, data, scalar_fns, pulse_process):
        self._fig = fig
        self._ax = ax
        self._data = data
        self.dkey = get_default_key(data)
        self._scalar_fns = scalar_fns
        self.skey = get_default_key(scalar_fns)
        self._pulse_process = pulse_process

    @property
    def dkey(self):
        """key to the data dictionary."""
        return self._dkey

    @dkey.setter
    def dkey(self, key):
        assert key in self._data
        self._dkey = key
        self._pulses, self._pos, self._meta = get_attributes(self._data, self._dkey)
        mask = np.abs(self._pos[:, 0]) < 10
        self._pulses = self._pulses[mask]
        self._pos = self._pos[mask]
        self._meta = self._meta[mask]

    @property
    def skey(self):
        """key to the data dictionary."""
        return self._skey

    @skey.setter
    def skey(self, key):
        assert key in self._scalar_fns
        self._skey = key
        xfunc = self._scalar_fns[key]["xfunc"]
        yfunc = self._scalar_fns[key]["yfunc"]
        self._xscalar = xfunc(self._pulses, self._pos, self._meta)
        self._yscalar = yfunc(self._pulses, self._pos, self._meta)

    def onpick(self, event):
        if len(event.ind) == 0:
            return
        if event.artist != self._plot:
            return
        dataind = event.ind[0]
        # ax["B"].clear()
        # ax["C"].clear()
        self._ax["A"].scatter(
            self._xscalar[dataind],
            self._yscalar[dataind],
            edgecolor="r",
            facecolor="none",
            s=50.0,
            lw=5.0,
        )
        self._ax["A"].annotate(
            str(dataind),
            (self._xscalar[dataind], self._yscalar[dataind]),
            (self._xscalar[dataind], self._yscalar[dataind]),
            color="green",
            size="large",
            # arrowprops=dict(arrowstyle="fancy", connectionstyle="arc3"),
        )
        xpos, ypos = self._pos[dataind][[0, 1]]
        self._ax["B"].plot(
            self._pulse_process(self._pulses[dataind, :, 0]),
            label=f"{dataind}:{xpos:.2f}, {ypos:.2f}",
        )
        self._ax["B"].set_xticks([])
        self._ax["C"].plot(
            self._pulse_process(self._pulses[dataind, :, 1]), label=dataind
        )
        self._ax["C"].set_xticks([])
        if self._pulses.shape[-1] > 2:
            self._ax["D"].plot(
                self._pulse_process(self._pulses[dataind, :, 2]), label=dataind
            )
            self._ax["D"].set_xticks([])
        self._ax["B"].legend()
        self._fig.canvas.draw_idle()

    def plot(self):
        self._plot = self._ax["A"].scatter(
            self._xscalar,
            self._yscalar,
            marker=".",
            picker=5,
        )
        self._ax["A"].set_ylabel(self._scalar_fns[self.skey]["yfunc"].__name__)
        self._ax["A"].set_xlabel(self._scalar_fns[self.skey]["xfunc"].__name__)
        self._ax["A"].set_title(self.skey)
        self._fig.canvas.draw_idle()

    def clear(self, event=None):
        self._ax["A"].clear()
        self._ax["B"].clear()
        self._ax["C"].clear()
        self._ax["D"].clear()
        self._fig.canvas.draw_idle()

    def update_skey(self, key):
        self.dkey = self._dkey
        self.skey = key
        self.clear()
        self.plot()

    def update_dkey(self, key):
        self.dkey = key
        self.skey = self._skey
        self.clear()
        self.plot()


def view_plots(data, scalar_fns, pulse_process=lambda x: x):
    fig, ax = plt.subplot_mosaic(
        "AB\nAC\nAD",
        figsize=(20, 10),
        gridspec_kw={"width_ratios": [50, 50]},
    )

    # Plot the first key by default and setup pulse picker
    upplt = PlotViewer(fig, ax, data, scalar_fns, pulse_process)
    axradio = fig.add_axes(
        [
            0.03,
            0.92,
            0.18 * np.where(len(scalar_fns.keys()) > 6, 6, len(scalar_fns.keys())),
            0.025 * (len(scalar_fns.keys()) // 6 + 1),
        ]
    )
    radio = MyRadioButtons(
        axradio,
        scalar_fns.keys(),
        orientation="horizontal",
        fontsize="x-small",
        active=0,
    )
    radio.on_clicked(upplt.update_skey)
    axradio2 = fig.add_axes([0.06, 0.04, 0.10 * len(data.keys()), 0.025])
    radio2 = MyRadioButtons(
        axradio2, data.keys(), orientation="horizontal", active=0, fontsize="x-small"
    )
    radio2.on_clicked(upplt.update_dkey)
    upplt.plot()
    fig.canvas.callbacks.connect("pick_event", upplt.onpick)
    plt.show()
