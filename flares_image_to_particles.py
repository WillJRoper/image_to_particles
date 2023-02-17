import sys
import unyt
from swiftsimio.units import cosmo_units
from swiftsimio import Writer
import PIL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import swiftascmaps


def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta = (theta + 2 * np.pi) % (2 * np.pi)
    out = np.zeros((x.size, 2))
    out[:, 0] = r
    out[:, 1] = theta
    return np.array((r, theta)).T


downsample = 2
IMAGE_SIZE = 3000 // downsample

image = PIL.Image.open("/Users/willroper/Downloads/flares_team.png")
resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
rgb_img = resized.convert('RGB')
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
rs = []
gs = []
bs = []
for x in np.arange(IMAGE_SIZE):
    for y in np.arange(IMAGE_SIZE):
        density = densities[x, y]

        these_xs, these_ys = np.meshgrid(
            *[(np.arange(density) + 0.5) / float(density)] * 2
        )

        xs.extend(list((these_xs + float(x)).flat))
        ys.extend(list((these_ys + float(y)).flat))
        dens.extend([density] * these_xs.size)

        r, g, b = rgb_img.getpixel((x, y))
        rs.extend(len(list((these_xs + float(x)).flat)) * [r, ])
        gs.extend(len(list((these_xs + float(x)).flat)) * [g, ])
        bs.extend(len(list((these_xs + float(x)).flat)) * [b, ])

xs = np.array(xs)
ys = np.array(ys)
dens = np.array(dens)
rs = np.array(rs)
gs = np.array(gs)
bs = np.array(bs)
print(np.max(xs), np.max(ys))
print(len(xs), len(dens))
# %%
fig, ax = plt.subplots(figsize=(4, 4))
fig.subplots_adjust(0, 0, 1, 1)
ax.axis("off")
# , cmap="swift.evermore_shifted")
h = plt.hist2d(ys, -xs, norm=LogNorm(), bins=IMAGE_SIZE)
print(h[0].max())
# plt.show()
# plt.close()
# %%
# Now create swift ics

# %%

np.save("rgbs.npy", np.array([rs, gs, bs]).T)


# Box is image_size cm
boxsize = IMAGE_SIZE * unyt.Mpc
print("Boxsize:", boxsize, "Npart: %d^3" % len(xs) ** (1/3))
# Generate object. cosmo_units corresponds to default Gadget-oid units
# of 10^10 Msun, Mpc, and km/s
x = Writer(cosmo_units, boxsize, dimension=2)

# 32^3 particles.
n_p = len(xs)

# Randomly spaced coordinates from 0, 100 Mpc in each direction
coords = np.array([xs, ys, np.zeros_like(xs)]).T
x.gas.coordinates = coords * unyt.Mpc

# Lets make some velocities
vels = np.zeros((xs.size, 3))
top = coords[:, 1] < (1000 / downsample)
middle = np.logical_and(coords[:, 1] < (2000 / downsample),
                        coords[:, 1] >= (1000 / downsample))
bottom = coords[:, 1] >= (2000 / downsample)
left = coords[:, 0] < (1000 / downsample)
centre = np.logical_and(coords[:, 0] < (2000 / downsample),
                        coords[:, 0] >= (1000 / downsample))
right = coords[:, 0] >= (2000 / downsample)
vels[np.logical_and(top, left), 0] = 1
vels[np.logical_and(top, centre), 0] = 0
vels[np.logical_and(top, right), 0] = -1
vels[np.logical_and(middle, left), 0] = 1
vels[np.logical_and(middle, centre), 0] = 0
vels[np.logical_and(middle, right), 0] = -1
vels[np.logical_and(bottom, left), 0] = 1
vels[np.logical_and(bottom, centre), 0] = 1
vels[np.logical_and(bottom, right), 0] = -1
vels[np.logical_and(top, left), 1] = 0
vels[np.logical_and(top, centre), 1] = -1
vels[np.logical_and(top, right), 1] = 0
vels[np.logical_and(middle, left), 1] = 1
vels[np.logical_and(middle, centre), 1] = 0
vels[np.logical_and(middle, right), 1] = -1
vels[np.logical_and(bottom, left), 1] = 0
vels[np.logical_and(bottom, centre), 1] = 1
vels[np.logical_and(bottom, right), 1] = 0
x.gas.velocities = vels * (unyt.km / unyt.s)

# Generate uniform masses as 10^6 solar masses for each particle
x.gas.masses = np.ones(n_p, dtype=float) * unyt.Msun

# Generate internal energy corresponding to 10^4 K
int_en = np.ones(n_p, dtype=float) * (unyt.km / unyt.s) ** 2
x.gas.internal_energy = np.ones(n_p, dtype=float) * (unyt.km / unyt.s) ** 2
#
x.gas.smoothing_length = np.ones(len(xs)) * unyt.Mpc
x.gas.particle_ids = np.arange(n_p, dtype=int)

# If IDs are not present, this automatically generates
x.write("flares_ics.hdf5")
