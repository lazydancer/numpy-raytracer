import numpy as np
import matplotlib.pyplot as plt

def normalize(xs):
    if len(xs.shape) == 1:
        return xs / np.linalg.norm(xs)
    else:
        return xs / np.linalg.norm(xs, axis=1)[:,np.newaxis]

class Metal:
    def __init__(self, color, diffusion, emit=False):
        self.color = np.array(color, dtype=np.float32)
        self.diffusion = diffusion
        self.emit = emit

    def _random_in_unit_sphere(self, normals):
        a = np.random.rand(normals.shape[0]) * 2 * np.pi
        z = np.random.rand(normals.shape[0]) * 2 - 1
        r = np.sqrt(1 - z*z)

        x = r * np.cos(a)
        y = r * np.sin(a)

        unit_random = np.array([x, y, z]).T

        return unit_random + normals



    def _reflect(self, directions, normals):
        reflect = normalize(directions - 2 * np.einsum('ij, ij->i', directions, normals)[:,None] * normals)
        #gauss = np.random.normal(size=directions.shape)
        #gauss = normalize(gauss * np.sum(gauss * normals))
        gauss = self._random_in_unit_sphere(normals)
        new_directions = reflect + self.diffusion * gauss

        return new_directions

    def scatters(self, directions, normals, intersections):
        origins = intersections + normals * 0.001
        new_directions = self._reflect(directions, normals)

        return origins, new_directions

class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=np.float32)
        self.radius = radius
        self.material = material

    def hit(self, origins, directions):
        m = origins - self.center

        b = np.einsum('ij, ij->i', m, directions)
        c = np.einsum('ij, ij->i', m, m) - self.radius*self.radius
        t = -b - np.sqrt(b*b - c)

        t[np.isnan(t)] = np.inf
        t[np.where( (t < 0) | ((c > 0) & (b > 0)) )] = np.inf

        return t


    def scatters(self, directions, intersections):
        normals = normalize(intersections - self.center)
        normals[np.where(np.einsum('ij, ij->i', directions, normals) > 0)] *= -1

        origins, new_directions = self.material.scatters(directions, normals, intersections)

        return origins, new_directions


def ray_color(scene, origins, directions, max_depth):
    '''
    Inputs the scene and the rays, manages the colors for each depth
    '''
    origins = origins.copy()
    directions = normalize(directions.copy())

    mask = np.arange(origins.shape[0])
    colors = np.ones(origins.shape, dtype=np.float32)
    emit = np.zeros(origins.shape[0], dtype=np.bool)

    for i in range(max_depth):
        sub_objects = np.empty(mask.shape, dtype=object)
        ts = np.full(mask.shape, np.inf, dtype=np.float32)

        for obj in scene:
            obj_ts = obj.hit(origins[mask], directions[mask])
            is_closer = np.where(obj_ts < ts)
            sub_objects[is_closer] = obj
            ts[is_closer] = obj_ts[is_closer]

        objects = np.empty(origins.shape[0], dtype=object)
        objects[mask] = sub_objects

        intersections = np.full(origins.shape, np.inf, dtype=np.float32)
        intersections[mask] = origins[mask] + directions[mask] * ts[:, np.newaxis]

        for obj in scene:
            is_obj = np.where(objects == obj)
            colors[is_obj] *= obj.material.color
            origins[is_obj], directions[is_obj] = obj.scatters(directions[is_obj], intersections[is_obj])
            emit[is_obj] = obj.material.emit

        mask = np.setdiff1d(mask, np.where(emit))

        print(i, len(mask))


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

    def take_picture(self, scene, save_loc):
        i = np.linspace(0, 1, self.width * self.sub_pixels_across)
        j = np.linspace(0, 1, self.height * self.sub_pixels_across)

        xx, yy = np.meshgrid(i, j)
        xx = xx.T.reshape(-1, 1)
        yy = yy.T.reshape(-1, 1)

        directions = self.lower_left + xx * self.horizontal + yy * self.vertical - self.origin
        directions = directions.astype(np.float32)
        origins = np.full(directions.shape, self.origin).astype(np.float32)
        colors = np.zeros(origins.shape).astype(np.float32)

        for i in range(self.samples_per_pixel):
            colors += ray_color(scene, origins, directions, self.max_depth) / self.samples_per_pixel
        
        colors = np.sqrt(colors) # gamma adjustment
        colors = colors.reshape((self.width, self.sub_pixels_across, self.height, self.sub_pixels_across, 3)).mean(3).mean(1)

        colors = np.clip(colors, 0, 1)
        img = colors.reshape(self.width, self.height, 3)
        img = np.transpose(img, (1, 0, 2))

        plt.imsave(save_loc, img, origin='lower')

def main():

    config = {
        'vup': np.array((0,1,0)),
        'vfov': np.pi/3, # In radians, give the virtical field of view
        'aspect_ratio': 16 / 9, 
        'image_width': 700, 
        'sub_pixels_across': 7, # 7
        'samples_per_pixel': 10, # 5
        'max_depth': 4, 
    }

    camera = Camera(config, [0,1,1], [0,0,-1])

    scene = [
        Sphere([0, 0 , -1], 0.3, Metal([1, 1, 1], 0)),
        Sphere([-1, 0 , -1], 0.3, Metal([1, 0.9, 0.9], 0.5, True)),
        Sphere([1, 0, -1], 0.3, Metal([1, 0.6, 0.5], 0.5)),
        Sphere([0, -1000.3, -1], 1000, Metal([1, 1, 1], 1)), # Ground
        Sphere([0, 1002, -1], 1000, Metal([1, 1, 1], 1)), # Ceiling 
        Sphere([1002, 0, -1], 1000, Metal([0.1, 0.1, 1], 1)), # Right
        Sphere([-1002, 0, -1], 1000, Metal([1, 0.1, 0.1], 1)), # Left
        Sphere([0, 0, -1003], 1000, Metal([1, 0.8, 1], 1)), # Back
        Sphere([0, 0, 1002], 1000, Metal([1, 1, 1], 1)), # Back
    ]

    camera.take_picture(scene, f'out.png')        


def rotate():

    config = {
        'vup': np.array((0,1,0)),
        'vfov': np.pi/3, # In radians, give the virtical field of view
        'aspect_ratio': 16 / 9, 
        'image_width': 700, 
        'sub_pixels_across': 7, # 7
        'samples_per_pixel': 1,
        'max_depth': 5, 
    }


    i = 346
    for coord in rotate_camera_around_point()[346:453]:
        print(coord)

        camera = Camera(config, coord, [0,0,-1])

        scene = [
            Sphere([0, 0 , -1], 0.3, Metal([0.2, 0.2, 0.2], 0.01)),
            Sphere([-1, 0 , -1], 0.3, Metal([1, 1, 1], 0.5, True)),
            Sphere([1, 0, -1], 0.3, Metal([1, 0.5, 0.4], 0.5)),
            Sphere([0, -1000.3, -1], 1000, Metal([1, 1, 1], 0.9)), # Ground
            Sphere([0, 1002, -1], 1000, Metal([1, 1, 1], 0.9)), # Ceiling 
            Sphere([1002, 0, -1], 1000, Metal([0.1, 0.1, 1], 0.9)), # Right
            Sphere([-1002, 0, -1], 1000, Metal([1, 0.1, 0.1], 0.9)), # Left
            Sphere([0, 0, -1003], 1000, Metal([1, 0.8, 1], 0.9)), # Back
            Sphere([0, 0, 1002], 1000, Metal([1, 1, 1], 0.9)), # Back
        ]

        camera.take_picture(scene, f'gif/{i:03d}.png')
        i += 1

# def rotate_camera_around_point():
#     frames = 500
#     step = 2*np.pi / frames  

#     rads = np.linspace(0, 2*np.pi, frames)

#     result = np.array([2*np.sin(-rads), np.ones(rads.shape), 2*np.cos(-rads)-1])

#     return result.T


#rotate_camera_around_point()
import time
start=time.time()
main()
print("time: {0:.6f}".format(time.time()-start))

#"C:\Users\jpucula\Portable Apps\ffmpeg-20200522-38490cb-win64-static\bin\ffmpeg.exe" -framerate 30 -i  C:\Users\jpucula\Documents\ray_tracer_v2\gif\%03d.png C:\Users\jpucula\Documents\ray_tracer_v2\gif\output.mp4
