import numpy as np
import matplotlib.pyplot as plt

# Constants
THICKNESS_CUBEMAP_SCALE = 0.2
DIST_SCALE = 0.8
INTERSECTION_PRECISION = 0.001
BOUND = 10.0
FRESNEL_RATIO = 0.8
IOR = 1.33
DISPERSION = 0.02
AA_SAMPLES = 8
TWO_PI = 2 * np.pi
ITERATIONS = 100

# Simple fake noise
def fake_sampler(uv, lod):
    x, y = uv
    return np.array([
        0.5 + 0.5 * np.sin(10.0 * x + lod),
        0.5 + 0.5 * np.sin(10.0 * y + lod),
        0.5 + 0.5 * np.sin(10.0 * (x + y) + lod)
    ])

def fancy_cube(sampler, d, s, b):
    uv_x = 0.5 + s * d[[1,2]] / (d[0] + 1e-6)
    uv_y = 0.5 + s * d[[2,0]] / (d[1] + 1e-6)
    uv_z = 0.5 + s * d[[0,1]] / (d[2] + 1e-6)

    colx = sampler(uv_x, b)
    coly = sampler(uv_y, b)
    colz = sampler(uv_z, b)

    n = d * d
    denom = n[0] + n[1] + n[2]
    return (colx * n[0] + coly * n[1] + colz * n[2]) / (denom + 1e-6)

def sdf(p):
    return np.linalg.norm(p) - 1.0

def calc_normal(p):
    eps = 1e-4
    dx = np.array([eps,0,0])
    dy = np.array([0,eps,0])
    dz = np.array([0,0,eps])
    normal = np.array([
        sdf(p+dx) - sdf(p-dx),
        sdf(p+dy) - sdf(p-dy),
        sdf(p+dz) - sdf(p-dz)
    ])
    return normal / (np.linalg.norm(normal)+1e-6)

def fresnel(rd, normal, ior):
    r0 = ((1.0 - ior) / (1.0 + ior))**2
    return r0 + (1.0 - r0) * (1.0 - np.clip(np.dot(rd, normal), 0.0, 1.0))**5

def reflect(rd, normal):
    return rd - 2.0 * np.dot(rd, normal) * normal

def do_camera(time):
    an = 1.5 + np.sin(time * 0.05) * 4.0
    cam_pos = np.array([6.5*np.sin(an), 0.0, 6.5*np.cos(an)])
    cam_tar = np.array([0.0, 0.0, 0.0])
    return cam_pos, cam_tar

def calc_lookat_matrix(ro, ta):
    ww = ta - ro
    ww /= np.linalg.norm(ww)
    uu = np.cross(ww, np.array([0.0,1.0,0.0]))
    uu /= np.linalg.norm(uu)
    vv = np.cross(uu, ww)
    return np.stack([uu,vv,ww], axis=1)

def render(time):
    w,h = 256,256
    img = np.zeros((h,w,3))
    cam_pos, cam_tar = do_camera(time)
    cam_mat = calc_lookat_matrix(cam_pos, cam_tar)

    for y in range(h):
        for x in range(w):
            uv = (np.array([x,y]) - np.array([w/2,h/2])) / h
            color = np.zeros(3)
            for samp in range(AA_SAMPLES):
                a = TWO_PI * samp / AA_SAMPLES
                dxy = 0.666/h * np.array([np.cos(a),np.sin(a)])
                rd = cam_mat @ np.array([uv[0]+dxy[0], uv[1]+dxy[1], 1.5])
                rd /= np.linalg.norm(rd)
                pos = cam_pos.copy()
                hit = False
                for _ in range(ITERATIONS):
                    t = DIST_SCALE * sdf(pos)
                    pos += rd * t
                    if np.linalg.norm(np.clip(pos, -BOUND, BOUND) - pos) > 0.0 or abs(t) < INTERSECTION_PRECISION:
                        hit = abs(t) < INTERSECTION_PRECISION
                        break
                if hit:
                    normal = calc_normal(pos)
                    film_thickness = fancy_cube(fake_sampler, normal, THICKNESS_CUBEMAP_SCALE, 0.0)[0] + 0.1
                    refl = reflect(rd, normal)
                    f = fresnel(rd, normal, 1.0/IOR)
                    base_color = np.clip(0.5 + 0.5*np.array([refl[0],refl[1],refl[2]]), 0,1)
                    rainbow = np.array([np.sin(10*film_thickness), np.sin(12*film_thickness), np.sin(14*film_thickness)])
                    rainbow = 0.5 + 0.5 * rainbow
                    color += (1-f)*base_color + f*rainbow
                else:
                    color += np.array([0.8,0.9,1.0])
            img[y,x] = np.clip(color / AA_SAMPLES, 0,1)
    return img

if __name__ == "__main__":
    time = 10.0  # Try changing time to animate
    img = render(time)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    plt.imsave('test1.png', img)
