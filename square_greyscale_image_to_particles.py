import sys
import unyt
from swiftsimio.units import cosmo_units
from swiftsimio import Writer
import PIL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import swiftascmaps

IMAGE_SIZE = 512
# %%
image = PIL.Image.open(sys.argv[1])
resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
grayscale = PIL.ImageOps.grayscale(resized)
posterized = PIL.ImageOps.posterize(grayscale, 2)

# %%
plt.imshow(posterized)
# plt.show()
# plt.close()
densities = (np.array(posterized) // (256 // 4)) + 1

xs = []
ys = []
dens = []
for x in np.arange(IMAGE_SIZE):
    for y in np.arange(IMAGE_SIZE):
        density = densities[x, y]

        these_xs, these_ys = np.meshgrid(
            *[(np.arange(density) + 0.5) / float(density)] * 2
        )

        xs.extend(list((these_xs + float(x)).flat))
        ys.extend(list((these_ys + float(y)).flat))
        dens.extend([density] * these_xs.size)

xs = np.array(xs)
ys = np.array(ys)
dens = np.array(dens)
print(np.max(xs), np.max(ys))
print(len(xs), len(dens))
# %%
fig, ax = plt.subplots(figsize=(4, 4))
fig.subplots_adjust(0, 0, 1, 1)
ax.axis("off")
# , cmap="swift.evermore_shifted")
h = plt.hist2d(ys, -xs, norm=LogNorm(), bins=IMAGE_SIZE)
print(h[0].max())
plt.show()
plt.close()
# %%
# Now create swift ics

# %%


# Box is image_size cm
boxsize = IMAGE_SIZE * unyt.Mpc
print("Boxsize:", boxsize, "Npart: %d^3" % len(xs) ** (1/3))
# Generate object. cosmo_units corresponds to default Gadget-oid units
# of 10^10 Msun, Mpc, and km/s
x = Writer(cosmo_units, boxsize, dimension=2)

# 32^3 particles.
n_p = len(xs)

# Lets make some velocities
vels = np.zeros((xs.size, 3))
# okinds = dens < 2
# vels[okinds, 0] = 1

# Randomly spaced coordinates from 0, 100 Mpc in each direction
x.gas.coordinates = np.array([xs, ys, np.zeros_like(xs)]).T * unyt.Mpc

delta = x.gas.coordinates.value - IMAGE_SIZE // 2
rs = np.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)
mid_point = np.argmin(rs)

# # Random velocities from 0 to 1 km/s
# x.gas.velocities = (
#     np.array(
#         [
#             3.0 * ((ys > IMAGE_SIZE[1] * 0.5) - 0.5),
#             10.0 * ((xs > IMAGE_SIZE[0] * 0.5) - 0.5),
#             np.zeros_like(xs),
#         ]
#     ).T
#     * (unyt.Mpc / unyt.s)
# )

x.gas.velocities = vels * (unyt.km / unyt.s)

# Generate uniform masses as 10^6 solar masses for each particle
x.gas.masses = np.ones(n_p, dtype=float) * unyt.Msun

# Generate internal energy corresponding to 10^4 K
int_en = np.ones(n_p, dtype=float) * (unyt.km / unyt.s) ** 2
int_en[mid_point] *= 1000 * (unyt.km / unyt.s) ** 2
x.gas.internal_energy = np.ones(n_p, dtype=float) * (unyt.km / unyt.s) ** 2

#
x.gas.smoothing_length = np.ones(len(xs)) * unyt.Mpc

# If IDs are not present, this automatically generates
x.write("img_ics.hdf5")
