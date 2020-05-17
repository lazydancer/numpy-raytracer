import numpy as np
import matplotlib.pyplot as plt

def normalize(xs):
    xs = (xs.T / np.linalg.norm(xs, axis=1)).T
    return xs

class Metal:
    def __init__(self, diffusion):
        self.diffusion = diffusion

    def scatters(self, directions, normals):
        reflect = normalize(directions - 2 * np.einsum('ij, ij->i', directions, normals)[:,None] * normals)
        gauss = np.random.normal(size=directions.shape)
        gauss = normalize(gauss * np.sum(gauss * normals))

        return reflect + self.diffusion * gauss

class Dielectric:
    def __init__(self, refraction_index):
        self.ri = refraction_index

    def scatters(self, directions, normals):
        pass

class Sphere:
    def __init__(self, center, radius, color, material):
        self.center = np.array(center, dtype=np.float32)
        self.radius = radius
        self.color = np.array(color, dtype=np.float32)
        self.material = material

    def hit(self, origins, directions):
        t = np.full(directions.shape[0], np.inf)
        temp = origins - self.center
        a = np.sum(np.square(directions), axis=1) 
        b = 2.0 * np.einsum('ij, ij->i', temp, directions)
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
        possible_hits = possible_hits[0][roots[:,1] >= 0]
        roots = roots[roots[:, 1] >= 0]

        # Find smallest > 0

        #breakpoint()
        t[possible_hits] = roots[np.arange(len(roots)), np.argmin(roots > 0, axis=1)]

        return t

    def scatters(self, directions, intersections):
        normals = normalize(intersections - self.center)
        origins = intersections + normals  * 0.001

        new_directions = self.material.scatters(directions, normals)

        return origins, new_directions

def trace_rays(scene, origins, directions):
    '''
    Find closest object fro each ray abd determine its color
    '''
    objects = np.empty(directions.shape[0], dtype=object)
    ts = np.full(directions.shape[0], np.inf).astype(np.float32)

    for obj in scene:
        obj_ts = obj.hit(origins, directions)

        objects[obj_ts < ts] = obj
        ts[obj_ts < ts] = obj_ts[obj_ts < ts]

    # Filter no hits 
    hits = np.where(ts != np.inf)[0]


    intersections = np.full(origins.shape, np.inf).astype(np.float32)
    normals = np.full(origins.shape, np.inf).astype(np.float32)
    colors = np.zeros(origins.shape).astype(np.float32)
    new_directions = np.empty(origins.shape)
    new_origins = np.empty(origins.shape)

    # Point of intersection
    intersections[hits] = origins[hits] + directions[hits] * np.array([ts[hits], ts[hits], ts[hits]]).T

    for obj in scene:
        colors[objects == obj] = obj.color
        new_origins[objects == obj], new_directions[objects == obj] = obj.scatters(directions[objects == obj], intersections[objects == obj])


    return hits, new_origins[hits], new_directions[hits], colors[hits]


def sample(scene, origins, directions):
    '''
    Inputs the scene and the rays, manages the colors for each depth
    '''
    origins = origins.copy()
    directions = directions.copy()
    mask = np.arange(origins.shape[0])

    colors = np.full(origins.shape, [1, 1, 1]).astype(np.float32)

    depth = 1
    max_depth = 15
    while depth < max_depth:        
        hits, ori, drt, col = trace_rays(scene, origins[mask], directions[mask])

        mask = mask[hits]
        colors[mask] = colors[mask] * col
        origins[mask] = ori
        directions[mask] = drt

        print(len(mask))
        depth += 1

    # Did not stop
    colors[mask] = 0

    return colors


def main():
    w, h = (600, 300)

    sub_pixels_across = 3

    x = np.linspace(-2, 2, w*sub_pixels_across)
    y = np.linspace(-1, 1, h*sub_pixels_across)

    directions = np.array(np.meshgrid(x, y, -1)).T.reshape(-1, 3).astype(np.float32)
    origins = np.zeros(directions.shape).astype(np.float32)

    scene = [
        Sphere([0, 0.2 , -1], 0.3, [0.2, 0.2, 0.2], Metal(0.01)),
        Sphere([0.7, 0, -1], 0.3, [0.7, 0.3, 0.2], Metal(0.5)),
        Sphere([0, -1000.3, -1], 1000, [0.2, 0.7, 0.2], Metal(0.8))
    ]

    colors = np.full(origins.shape, 0).astype(np.float32)

    rays_per_pixel = 10
    for i in range(rays_per_pixel):
        colors += sample(scene, origins, directions) / rays_per_pixel
    
    colors = np.sqrt(colors) # gamma adjustment
    colors = colors.reshape((w, sub_pixels_across, h, sub_pixels_across, 3)).mean(3).mean(1)

    colors = np.clip(colors, 0, 1)
    img = colors.reshape(w, h, 3)
    img = np.transpose(img, (1, 0, 2))

    plt.imsave('out.png', img, origin='lower')




import time
start=time.time()
main()
print("time: {0:.6f}".format(time.time()-start))
