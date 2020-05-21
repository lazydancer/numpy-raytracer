import numpy as np
import matplotlib.pyplot as plt

def normalize(xs):
    if len(xs.shape) == 1:
        return xs / np.linalg.norm(xs)
    else:
        return xs / np.linalg.norm(xs, axis=1)[:,np.newaxis]

class Metal:
    def __init__(self, color, diffusion, emit=[0,0,0]):
        self.color = np.array(color, dtype=np.float32)
        self.diffusion = diffusion
        self.emit = np.array(emit, dtype=np.float32)

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
        r = 1/1.4
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
    colors = np.zeros(origins.shape).astype(np.float32)
    new_directions = np.empty(origins.shape)
    new_origins = np.empty(origins.shape)
    emit = np.zeros(origins.shape).astype(np.float32)

    # Point of intersection
    intersections[hits] = origins[hits] + directions[hits] * np.array([ts[hits], ts[hits], ts[hits]]).T

    for obj in scene:
        colors[objects == obj] = obj.material.color
        new_origins[objects == obj], new_directions[objects == obj] = obj.scatters(directions[objects == obj], intersections[objects == obj])
        emit[objects == obj] = obj.material.emit


    return hits, new_origins[hits], new_directions[hits], colors[hits], emit[hits]


def sample(scene, origins, directions, max_depth):
    '''
    Inputs the scene and the rays, manages the colors for each depth
    '''
    origins = origins.copy()
    directions = directions.copy()
    mask = np.arange(origins.shape[0])

    colors = np.full(origins.shape, [1, 1, 1]).astype(np.float32)

    depth = 0
    while depth <= max_depth:        
        hits, ori, drt, col, emit = trace_rays(scene, origins[mask], directions[mask])

        mask = mask[hits]
        colors[mask] = colors[mask] * col 
        origins[mask] = ori
        directions[mask] = drt

        mask = np.delete(mask, np.where(emit > 0)[0])

        print(len(mask))

        depth += 1

    colors[mask] = 0

    return colors



class Camera:
    def __init__(self, config, look_from, look_at):
        self.origin = np.array(look_from)
        self.look_at = np.array(look_at)
        
        self.width = config['image_width']
        self.height = int(config['image_width'] / config['aspect_ratio'])
        self.samples_per_pixel = config['samples_per_pixel']
        self.max_depth = config['max_depth']
        #self.lens_radius = config['aperture']

        w = normalize(self.origin - self.look_at)
        self.u = normalize(np.cross(config['vup'], w))
        self.v = np.cross(w, self.u)

        half_height = np.tan(config['vfov']/2)
        half_width = config['aspect_ratio'] * half_height

        focus_dist = np.linalg.norm(self.origin - self.look_at)
        self.lower_left = self.origin - \
            half_width * focus_dist * self.u - \
            half_height * focus_dist * self.v - \
            focus_dist * w

        self.horizontal = 2*half_width * focus_dist * self.u
        self.vertical = 2*half_height * focus_dist * self.v

        self.sub_pixels_across = config['sub_pixels_across']

    def take_picture(self, scene):
        i = np.linspace(0, 1, self.width * self.sub_pixels_across)
        j = np.linspace(0, 1, self.height * self.sub_pixels_across)

        xx, yy = np.meshgrid(i, j)
        xx = xx.T.reshape(-1, 1)
        yy = yy.T.reshape(-1, 1)

        directions = self.lower_left + xx * self.horizontal + yy * self.vertical - self.origin
        
        origins = np.full(directions.shape, self.origin).astype(np.float32)
        colors = np.zeros(origins.shape).astype(np.float32)

        for i in range(self.samples_per_pixel):
            colors += sample(scene, origins, directions, self.max_depth) / self.samples_per_pixel
        
        colors = np.sqrt(colors) # gamma adjustment
        colors = colors.reshape((self.width, self.sub_pixels_across, self.height, self.sub_pixels_across, 3)).mean(3).mean(1)

        colors = np.clip(colors, 0, 1)
        img = colors.reshape(self.width, self.height, 3)
        img = np.transpose(img, (1, 0, 2))

        plt.imsave('out.png', img, origin='lower')

def main():

    config = {
        'vup': np.array((0,1,0)),
        'vfov': np.pi/3, # In radians, give the virtical field of view
        'aspect_ratio': 16 / 9, 
        #'aperture': 0.5, # The radius of the apetrue
        'image_width': 700,
        'sub_pixels_across': 2,
        'samples_per_pixel': 1,
        'max_depth': 300,
    }

    camera = Camera(config, [0,1,1], [0,0,-1])

    scene = [
        Sphere([0, 0 , -1], 0.3, Metal([0.2, 0.2, 0.2], 0.01)),
        Sphere([-1, 0 , -1], 0.3, Metal([1, 1, 1], 0.5, [1,1,1])),
        Sphere([1, 0, -1], 0.3, Metal([1, 0.5, 0.4], 0.5)),
        Sphere([0, -1000.3, -1], 1000, Metal([1, 1, 1], 0.8)), # Ground
        Sphere([0, 1002, -1], 1000, Metal([1, 1, 1], 1)), # Ceiling 
        Sphere([1002, 0, -1], 1000, Metal([0.1, 0.1, 1], 0.8)), # Right
        Sphere([-1002, 0, -1], 1000, Metal([1, 0.1, 0.1], 0.8)), # Left
        Sphere([0, 0, -1002], 1000, Metal([1, 0.8, 1], 0.8)), # Back
        Sphere([0, 0, 1002], 1000, Metal([1, 1, 1], 0.8)), # Back
    ]

    camera.take_picture(scene)

import time
start=time.time()
main()
print("time: {0:.6f}".format(time.time()-start))
