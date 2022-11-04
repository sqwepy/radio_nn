#! /usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

plt.rc('font', size=25) #controls default text size
run = "000003"
antenna = "3"
file = f"./data/DATA{run}/SIM{run}_coreas/raw_pos_{antenna}.dat"
# here
datac7 = np.loadtxt(file)
# TODO: Get antenna info
# datac7 = np.abs(datac7)
# c7 clean data
timec7 = np.fft.rfftfreq(datac7[:,0].shape[-1], d=datac7[1,0] - datac7[0,0])
#timec7 = datac7[:,0]
exc7 = np.fft.rfft(datac7[:,1]) *2.99792458 * 10 ** 10
eyc7 = np.fft.rfft(datac7[:,2]) *2.99792458 * 10 ** 10
ezc7 = np.fft.rfft(datac7[:,3]) *2.99792458 * 10 ** 10

print(timec7.shape)
print(datac7.shape)

plotstyle1 = {'color': 'r',
                 'marker':'.'
              }
plotstyle2 = {'color': 'b',
                 'marker':'.'
              }
fn1 = np.real
fn2 = np.imag
# Plot West pol
fig, ax = plt.subplots(3, 2, sharex=True, figsize=(40,30))
ax[0,0].plot(timec7, fn1(eyc7), **plotstyle1)
ax[0,0].set_title(f'West - C7 CoREAS - {fn1.__name__}')
ax[0,1].plot(timec7, fn2(eyc7), **plotstyle2)
ax[0,1].set_title(f'West - C7 CoREAS - {fn2.__name__}')
# # Plot North pol
ax[1,0].plot(timec7, fn1(exc7), **plotstyle1)
ax[1,0].set_title(f'North - C7 CoREAS - {fn1.__name__}')
ax[1,1].plot(timec7, fn2(exc7), **plotstyle2)
ax[1,1].set_title(f'North - C7 CoREAS - {fn2.__name__}')
# # Plot Vertical pol
ax[2,0].plot(timec7, fn1(ezc7), **plotstyle1)
ax[2,0].set_title(f'Vertical - C7 CoREAS - {fn1.__name__}')
ax[2,1].plot(timec7,fn2(ezc7), **plotstyle2)
ax[2,1].set_title(f'Vertical - C7 CoREAS - {fn2.__name__}')

fig.suptitle(f"Run: {run} Antenna: {antenna}")
fig.supylabel('Electric Field [$\mu$V/m]')
fig.supxlabel('Time [s]')
[k.grid(which="both", linestyle="dashed") for i in ax for k in i]
plt.tight_layout()
plt.savefig("pulses.pdf")
#i#plt.show()