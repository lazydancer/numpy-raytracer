import numpy as np
import matplotlib.pyplot as plt

def normalize(xs):
    return xs / np.linalg.norm(xs, axis=1)[:,np.newaxis]

class Metal:
    def __init__(self, color, diffusion):
        self.color = np.array(color, dtype=np.float32)
        self.diffusion = diffusion

    def _reflect(self, directions, normals):
        reflect = normalize(directions - 2 * np.einsum('ij, ij->i', directions, normals)[:,None] * normals)
        gauss = np.random.normal(size=directions.shape)
        gauss = normalize(gauss * np.sum(gauss * normals))

        new_directions = reflect + self.diffusion * gauss

        return new_directions

    def scatters(self, directions, normals, intersections):
        origins = intersections + normals * 0.001
        new_directions = self._reflect(directions, normals)

        return origins, new_directions


class Dielectric:
    def __init__(self, refraction_index):
        self.ri = refraction_index
        self.color = np.array([1, 1, 1], dtype=np.float32)

    def _refract(self, directions, normals):
        '''
        https://en.wikipedia.org/wiki/Snell's_law
        '''
        r = 1/1.0
        normals[np.where(np.einsum('ij, ij->i', directions, -normals) < 0)] *= -1
        c = np.einsum('ij, ij->i', directions, -normals)

        return r * directions + (r * c - np.sqrt(1 - r**2 * ( 1 - c**2 )))[:,np.newaxis] * normals


    def scatters(self, directions, normals, intersections):
        origins = intersections - normals * 0.001
        new_directions = self._refract(normalize(directions), normals)

        return origins, new_directions


class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=np.float32)
        self.radius = radius
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

        distSqrt = np.sqrt(discriminant)
        distSqrt[np.where( b[possible_hits] < 0 )] *= -1

        q = (-b[possible_hits] + distSqrt) / 2.0

        t0 = q / a[possible_hits] 
        t1 = c[possible_hits] / q

        roots = np.vstack([t0, t1]).T
        roots[np.where(roots < 0)] = np.inf
        roots.sort(axis=1)
        
        t[possible_hits] = roots[:,0]

        return t

    def scatters(self, directions, intersections):
        normals = normalize(intersections - self.center)
        normals[np.where(np.einsum('ij, ij->i', directions, normals) > 0)] *= -1

        origins, new_directions = self.material.scatters(directions, normals, intersections)

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
        colors[objects == obj] = obj.material.color
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

class Camera:
    def __init__(self, w, h, look_from, look_at):
        self.w = w
        self.h = h
        self.look_from = look_from
        self.look_at = look_at

    def rays(self):
        pass

    def take_picture(scene):
        pass

def main():
    w, h = (600, 300)

    sub_pixels_across = 5

    x = np.linspace(-2, 2, w*sub_pixels_across)
    y = np.linspace(-1, 1, h*sub_pixels_across)

    directions = np.array(np.meshgrid(x, y, -1)).T.reshape(-1, 3).astype(np.float32)
    origins = np.zeros(directions.shape).astype(np.float32)

    scene = [
        Sphere([-0.2, 0 , -1], 0.3, Metal([0.2, 0.2, 0.2], 0.01)),
        Sphere([0.7, 0, -1], 0.3, Metal([0.7, 0.3, 0.2], 0.5)),
        Sphere([-1, 0, -1], 0.3, Dielectric(1.5)),
        Sphere([0, -1000.3, -1], 1000, Metal([0.2, 0.7, 0.2], 0.8))
    ]

    colors = np.full(origins.shape, 0).astype(np.float32)

    rays_per_pixel = 1
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
