import numpy as np
import matplotlib.pyplot as plt

def normalize(xs):
    xs = (xs.T / np.linalg.norm(xs, axis=1)).T
    return xs

class Sphere:
    def __init__(self, center, radius):
        self.center = np.array(center, dtype=np.float)
        self.radius = radius

    def hit(self, origins, directions):
        t = np.full(directions.shape[0], np.inf)
        temp = origins - self.center
        a = np.sum(np.square(directions), axis=1) 
        b = 2.0 * np.sum(temp * directions, axis=1)
        c = np.sum(np.square(temp), axis=1) - self.radius * self.radius
        discriminant = b * b - 4.0 *  a * c
        possible_hits = np.where(discriminant >= 0.0)

        discriminant = discriminant[possible_hits]
        b = b[possible_hits]
        a = a[possible_hits]
        c = c[possible_hits]

        distSqrt = np.sqrt(discriminant)
        distSqrt[np.where(b<0)] = -distSqrt[np.where(b<0)]

        q = (-b + distSqrt) / 2.0

        t0 = q / a 
        t1 = c / q

        roots = np.vstack([t0, t1]).T
        roots.sort(axis=1)

        # Remove if both roots negative 
        possible_hits = possible_hits[0][roots[:,0] >= 0]

        # This only returns the closest for now.
        # Need to account for if camera inside sphere

        t[possible_hits] = roots[roots[:,0] >= 0][:,0]

        return t

    def normal(self, intersections):
        return normalize(intersections - self.center)

def scatters(directions, normals):
    diffusion = 0.2

    reflect = normalize(directions - 2 * np.sum(directions * normals, axis=1)[:,None] * normals)

    gauss = np.random.normal(size=directions.shape)
    gauss = normalize(gauss * np.sum(gauss * normals))

    return reflect + diffusion * gauss

def trace_rays(scene, origins, directions):
    objects = np.empty(directions.shape[0], dtype=object)
    intersections = np.full(origins.shape, np.inf)
    normals = np.full(origins.shape, np.inf)
    colors = np.zeros(origins.shape)

    ts = np.full(directions.shape[0], np.inf)

    for obj in scene:
        obj_ts = obj.hit(origins, directions)

        objects[obj_ts < ts] = obj
        ts[obj_ts < ts] = obj_ts[obj_ts < ts]

    # Filter no hits 
    hits = ts != np.inf

    # Point of intersection
    intersections[hits] = origins[hits] + directions[hits] * np.array([ts[hits], ts[hits], ts[hits]]).T

    for obj in scene:
        normals[hits & (objects == obj)] = obj.normal(intersections[objects == obj])

    colors[hits] = 0.1
    colors[~hits] = 0.7

    return objects, intersections, normals, colors 


def main():
    w, h = (600, 300)

    x = np.linspace(-2, 2, w)
    y = np.linspace(-1, 1, h)

    directions = np.array(np.meshgrid(x, y, -1)).T.reshape(-1, 3)
    origins = np.zeros(directions.shape)

    scene = [
        Sphere([0, 0.2 , -1], 0.3),
        Sphere([0.7, 0, -1], 0.3),
        Sphere([0, -100.3, -1], 100)
    ]

    mask = np.ones(origins.shape[0], dtype=np.bool)

    objects = np.empty(directions.shape[0], dtype=object)
    intersections = np.full(origins.shape, np.inf)
    normals = np.full(origins.shape, np.inf)
    colors = np.full(origins.shape, 0.5)
    new_colors = np.zeros(origins.shape)

    depth = 1
    max_depth = 10
    while depth < max_depth:
        
        objects[mask], intersections[mask], normals[mask], new_colors[mask] = trace_rays(scene, origins[mask], directions[mask])
        colors[mask] = new_colors[mask] / depth

        mask = mask & np.any(intersections != np.inf, axis=1)

        origins[mask] = intersections[mask] + normals[mask]  * 0.001
        directions[mask] = scatters(directions[mask], normals[mask])

        depth += 1

    colors[mask] = 0

    colors = np.clip(colors, 0, 1)
    img = colors.reshape(w, h, 3)
    img = np.transpose(img, (1, 0, 2))

    plt.imsave('out.png', img, origin='lower')




import time
start=time.time()
main()
print("time: {0:.6f}".format(time.time()-start))
