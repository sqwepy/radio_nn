#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

run = "000001"


def read_long_C7(filename):
    with open(filename) as f:
        lines = f.readlines()

    line_iter = enumerate(lines)

    s_particle_number = None
    s_energy_dep = None

    for i, l in line_iter:
        if l.startswith(" DEPTH"):
            s_particle_number = i + 1
            break

    for i, l in line_iter:
        if l.startswith(" LONGITUDINAL"):
            s_energy_dep = i
            break

    if not s_energy_dep or not s_particle_number:
        raise Exception('"{}" could not be parsed as longfile'.format(filename))

    array = []
    for line in lines[s_particle_number:s_energy_dep]:
        array.append(list(map(float, line.split())))

    longprof = np.array(array)
    X = longprof[:, 0]

    columns = (
        "X",
        "gamma",
        "e+",
        "e-",
        "mu+",
        "mu-",
        "hadron",
        "charged",
        "nuclei",
        "Cherenkov",
    )

    return (
        longprof[:, 0],
        longprof[:, 1],
        longprof[:, 2],
        longprof[:, 3],
        longprof[:, 4],
        longprof[:, 5],
        longprof[:, 6],
    )


depth, gammas, positrons, electrons, muplus, muminus, hadrons = read_long_C7(
    f"./data/DATA{run}/DAT{run}.long"
)  # place your DAT000XXXX.long C7 file here

print(depth.shape)
print(gammas.shape)
plt.plot(depth, electrons, label="electrons")
plt.legend(fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Shower Depth  [g $cm^{-2}$]", fontsize=20)
plt.ylabel("Nr of particles", fontsize=20)
plt.grid(which="both", linestyle="dashed")
plt.title("Longitudinal Profile", fontsize=20)
plt.tight_layout()
plt.show()
