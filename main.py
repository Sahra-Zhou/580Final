import numpy as np
import matplotlib.pyplot as plt
import math
from OpenGL.GL import *
from OpenGL.arrays import vbo
from PIL import Image
# Constants
IOR = 0.9  # Index of refraction for the material
DISPERSION = 0.05  # Dispersion constant for wavelength-based IOR changes
FRESNEL_RATIO = 0.7  # Reflectance base ratio for Fresnel-Schlick
SIGMOID_CONTRAST = 8.0
THICKNESS_SCALE = 1.0  # Scaling factor for film thickness
THINKCKNESS_CUBEMAO_SCALE = 0.1
REFLECTANCE_SCALE = 3.0
REFLECTANCE_GAMMA_SCALE = 2.0
AA_SAMPLES = 1  # Anti-aliasing samples disabled for now 
ITERATIONS = 20  # Raymarch iterations
INTERSECTION_PRECISION = 0.01  # Precision for intersection tests
BOUND = 6.0  # Boundaries for scene
WAVELENGTHS = 6  # Number of wavelengths for dispersion calculation
GAMMA_CURVE = 50.0
GAMMA_SCALE = 4.5
#DIST_SCALE = 0.9 # ray marching

def fake_sampler(uv, lod):
    # Fake some smooth noise based on uv
    x, y = uv
    return np.array([
        0.5 + 0.5 * np.sin(10.0 * x + lod),
        0.5 + 0.5 * np.sin(10.0 * y + lod),
        0.5 + 0.5 * np.sin(10.0 * (x + y) + lod)
    ])

def fancy_cube(d, s, b):
    # Assume sampler is a function we provide to simulate a "texture lookup"
    uv_x = 0.5 + s * d[[1,2]] / d[0]
    uv_y = 0.5 + s * d[[2,0]] / d[1]
    uv_z = 0.5 + s * d[[0,1]] / d[2]

    colx = fake_sampler(uv_x, b)
    coly = fake_sampler(uv_y, b)
    colz = fake_sampler(uv_z, b)

    n = d * d
    denom = n[0] + n[1] + n[2]

    return (colx * n[0] + coly * n[1] + colz * n[2]) / denom

# Vector helper functions
def normalize(v):
    return v / np.linalg.norm(v)

def reflect(I, N):
    return I - 2 * np.dot(I, N) * N

def refract(I, N, eta):
    cosi = np.dot(I, N)
    etai = 1.0
    etat = eta
    if cosi < 0:
        cosi = -cosi
        N = -N
        etai, etat = etat, etai
    eta = etai / etat
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    if k < 0:
        return np.zeros(3)  # Total internal reflection
    return eta * I + (eta * cosi - np.sqrt(k)) * N

def hash(x):
    val = math.sin(x)*43758.5453
    return val - math.floor(val)

def lerp(a, b, t):
    return a*t + (1.0-t)*b
def noise(x):
    p = np.floor(x)
    f = x - np.floor(x)
    f = f*f*(3.0 - 2.0*f)
    n = p[0] + p[1]*57.0 + p[2]*113.0
    return lerp(lerp(lerp(hash(n+0.0), hash(n + 1.0), f[0]), lerp(hash(n+57.0), hash(n + 58.0), f[0]), f[1]), 
                lerp(lerp(hash(n+113.0), hash(n + 114.0), f[0]), lerp(hash(n+170.0), hash(n + 171.0), f[0]), f[1]), f[2])

def noise3(x):
    return np.array([noise(x+np.array([123.456, .567, .37])), 
                     noise(x+np.array([.11, 47.43, 19.17])), 
                     noise(x)])

def texture_sampler(direction):
    # For now, just return something simple:
    # Like mapping direction Y to sky color
    t = 0.5 * (direction[1] + 1.0)
    sky_color = np.array([0.6, 0.8, 1.0])
    ground_color = np.array([0.2, 0.25, 0.3])
    return (1.0 - t) * ground_color + t * sky_color

def sample_cube_map2(i, rd):
    col = texture_sampler(rd)
    return np.array([
        np.dot(tex_cube_sample_weights(i[0]), col),
        np.dot(tex_cube_sample_weights(i[1]), col),
        np.dot(tex_cube_sample_weights(i[2]), col)
    ])
def sample_weights(i, GREEN_WEIGHT=2.8):
    return np.array([(1.0-i)*(1.0-i), GREEN_WEIGHT*i*(1.0-i), i*i])

def sample_cube_map(i, rd0, rd1, rd2):
    # Flip Y axis as in rd * vec3(1.0, -1.0, 1.0)
    rd0 = np.array([rd0[0], -rd0[1], rd0[2]])
    rd1 = np.array([rd1[0], -rd1[1], rd1[2]])
    rd2 = np.array([rd2[0], -rd2[1], rd2[2]])
    # Sample the texture (cube map) at the modified ray directions
    col0 = texture_sampler(rd0)
    col1 = texture_sampler(rd1)
    col2 = texture_sampler(rd2)
    # Apply weighting
    result = np.array([
        np.dot(tex_cube_sample_weights(i[0]), col0),
        np.dot(tex_cube_sample_weights(i[1]), col1),
        np.dot(tex_cube_sample_weights(i[2]), col2)
    ])
    
    return result

def filmic_gamma(x):
    return np.log(GAMMA_CURVE * x + 1.0) / GAMMA_SCALE

def filmic_gamma_inverse(y):
    return (1.0 / GAMMA_CURVE) * (np.exp(GAMMA_SCALE * y) - 1.0)

def tex_cube_sample_weights(i, GREEN_WEIGHT=2.8):
    w = np.array([
        (1.0 - i) * (1.0 - i),
        GREEN_WEIGHT * i * (1.0 - i),
        i * i
    ])
    return w / np.sum(w)

def sample_weight(i):
    return np.array([(1.0 - i) * (1.0 - i), 2.8 * i * (1.0 - i), i * i])

def resample(wl0, wl1, i0, i1):
    w0 = sample_weight(wl0[0])
    w1 = sample_weight(wl0[1])
    w2 = sample_weight(wl0[2])
    w3 = sample_weight(wl1[0])
    w4 = sample_weight(wl1[1])
    w5 = sample_weight(wl1[2])
    return i0[0]*w0 + i0[1] * w1 + i0[2] * w2 + i1[0] * w3 + i1[1] * w4 + i1[2] * w5

def resample_color(rds, refl0, refl1, wl0, wl1):
    cube0 = sample_cube_map(wl0, rds[0], rds[1], rds[2])
    cube1 = sample_cube_map(wl1, rds[3], rds[4], rds[5])

    intensity0 = filmic_gamma_inverse(cube0) + refl0
    intensity1 = filmic_gamma_inverse(cube1) + refl1
    col = resample(wl0, wl1, intensity0, intensity1)
    return 1.4 * filmic_gamma(col / 6.0) # wavelength

def resample_color_simple(rd, wl0, wl1):
    cube0 = sample_cube_map2(wl0, rd)
    cube1 = sample_cube_map2(wl1, rd)
    intensity0 = filmic_gamma_inverse(cube0)
    intensity1 = filmic_gamma_inverse(cube1)
    col = resample(wl0, wl1, intensity0, intensity1)
    return 1.4 * filmic_gamma(col / 6.0) # wavelength
# Fresnel-Schlick approximation
# def fresnel(I, N, eta):
#     cosi = np.dot(I, N)
#     r0 = (1.0 - eta) / (1.0 + eta)
#     r0 = r0 * r0
#     return r0 + (1.0 - r0) * (1.0 - cosi) ** 5
def clamp(val, minV, maxV):
    return max(minV, min(val, maxV))

def fresnel(rd, norm, n2):
    r0 = np.power((1.0 - n2) / (1.0 + n2), 2)
    cos_theta = np.clip(1.0 + np.dot(rd, norm), 0.0, 1.0)
    return r0 + (1.0 - r0) * np.power(cos_theta, 5)


# Attenuation based on film thickness and wavelength
def attenuation(film_thickness, wavelengths, normal, rd):
    val = 0.5 + 0.5 * np.cos(((THICKNESS_SCALE * film_thickness) / (wavelengths + 1.0))* np.dot(normal, rd))
    return val #np.exp(-film_thickness * wavelengths) * np.abs(np.dot(normal, rd))

def contrast(x):
    return 1.0 / (1.0+np.exp(-SIGMOID_CONTRAST * (x - 0.5)))

# IOR curve for dispersion (for each wavelength)
def ior_curve(wavelength):
    return 1.5 + 0.02 * wavelength  # Simple model for IOR depending on wavelength

# Raymarching SDF (Signed Distance Function)
# def sdf(position):
#     return np.linalg.norm(position) - 1.0  # Sphere of radius 1 at origin
def sdf(position, iTime=3.0):
    # animated direction vector
    n = np.array([
        np.sin(iTime * 0.5),
        np.sin(iTime * 0.3),
        np.cos(iTime * 0.2)
    ])

    # you need to define noise3 separately (I'll give you a simple one below)
    q = 0.1 * (noise3(position + n) - 0.5)

    return np.linalg.norm(position + q) - 3.5

# Normal calculation (for SDF)
def calc_normal(position):
    eps = INTERSECTION_PRECISION
    grad = np.array([
        sdf(position + np.array([eps, 0, 0])) - sdf(position),
        sdf(position + np.array([0, eps, 0])) - sdf(position),
        sdf(position + np.array([0, 0, eps])) - sdf(position)
    ])
    return normalize(grad)

def intersect_sphere(ro, rd, center=np.array([0.0, 0.0, 0.0]), radius=1.0):
    oc = ro - center
    a = np.dot(rd, rd)
    b = 2.0 * np.dot(oc, rd)
    c = np.dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        # no hit
        return None  
    else:
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2.0 * a)
        t2 = (-b + sqrt_discriminant) / (2.0 * a)
        if t1 > 0:
            return t1
        if t2 > 0:
            return t2
        # behind the ray
        return None  


def thin_film_color(view_angle, thickness):
    n_air = 1.0
    n_bubble = 1.33  # Water soap film
    delta = (4.0 * np.pi * n_bubble * thickness) / np.array([0.7, 0.55, 0.4])  # wavelengths in microns
    cos_theta = view_angle
    iridescence = 0.5 + 0.5 * np.cos(delta * cos_theta)
    return np.clip(iridescence, 0.0, 1.0)

def environment_map(rd):
    t = 0.5 * (rd[1] + 1.0)  # -1 -> 0, +1 -> 1
    t = t**0.5  # gamma correction to balance horizon brightness
    sky_color = np.array([0.6, 0.8, 1.0])
    ground_color = np.array([0.2, 0.25, 0.3])
    return (1.0 - t) * ground_color + t * sky_color


def main_image(iTime, fragCoord, iResolution):
    col = np.zeros(3)

    wavelengths0 = np.array([1.0, 0.8, 0.6])
    wavelengths1 = np.array([0.4, 0.2, 0.0])
    iors0 = IOR + ior_curve(wavelengths0) * DISPERSION
    iors1 = IOR + ior_curve(wavelengths1) * DISPERSION

    uv = (fragCoord / iResolution) * 2.0 - 1.0
    uv[0] *= iResolution[0] / iResolution[1]  # fix aspect ratio
    ro = np.array([0.0, 0.0, -3.0])
    rd = normalize(np.array([uv[0], uv[1], 1.5]))

    rds = np.zeros((WAVELENGTHS, 3))
    t = intersect_sphere(ro, rd)
    if t is not None:
        pos = ro + t * rd
        normal = calc_normal(pos)

        view_angle = abs(np.dot(rd, normal))
        #film_thickness = 0.4 + 0.6 * (1.0 - view_angle)  # thicker at edges
        film_thickness = 0.4 + 0.6 * (1.0 - view_angle)
        film_thickness += 0.05 * np.sin(20.0 * view_angle + iTime * 2.0)

        att0 = attenuation(film_thickness, wavelengths0, normal, rd)
        att1 = attenuation(film_thickness, wavelengths1, normal, rd)

        f0 = (1.0 - FRESNEL_RATIO) + FRESNEL_RATIO * fresnel(rd, normal, 1.0 / iors0)
        f1 = (1.0 - FRESNEL_RATIO) + FRESNEL_RATIO * fresnel(rd, normal, 1.0 / iors1)

        rrd = reflect(rd, normal)

        cube0 = REFLECTANCE_GAMMA_SCALE * att0 * sample_cube_map2(wavelengths0, rrd)
        cube1 = REFLECTANCE_GAMMA_SCALE * att1 * sample_cube_map2(wavelengths1, rrd)

        refl0 = REFLECTANCE_SCALE * filmic_gamma_inverse(np.interp(np.zeros(3), cube0, f0))
        refl1 = REFLECTANCE_SCALE * filmic_gamma_inverse(np.interp(np.zeros(3), cube1, f1))

        rds[0] = refract(rd, normal, iors0[0])
        rds[1] = refract(rd, normal, iors0[1])
        rds[2] = refract(rd, normal, iors0[2])
        rds[3] = refract(rd, normal, iors1[0])
        rds[4] = refract(rd, normal, iors1[1])
        rds[5] = refract(rd, normal, iors1[2])
        col += resample_color(rds, refl0, refl1, wavelengths0, wavelengths1)
    
    else:
        # background
        col += resample_color_simple(rd, wavelengths0, wavelengths1)
    frag_color = contrast(col)
    return np.clip(frag_color, 0.0, 1.0)



# Render 
def render_image(iResolution=(800, 600)):
    img = np.zeros((iResolution[1], iResolution[0], 3))
    for y in range(iResolution[1]):
        for x in range(iResolution[0]):
            iTime = 1.0  # placeholder
            fragCoord = np.array([x, y])  # current pixel coordinate
            img[y, x] = main_image(iTime, fragCoord, np.array(iResolution))
    return img



iResolution = (200, 150)
output_image = render_image(iResolution)

# Display the output
# plt.imshow(output_image)
# plt.axis('off') 
# plt.show()
plt.imsave('bubble.png', output_image)
