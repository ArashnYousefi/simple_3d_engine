# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------
import pygame
import numpy as np
from numba import njit

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CAMERA_SPEED = 8.0
MOUSE_SENSITIVITY = 0.002
FOV = 75
NEAR_PLANE = 0.1
FAR_PLANE = 1000.0
ASPECT_RATIO = SCREEN_WIDTH / SCREEN_HEIGHT

# translation_matrix, rotation_matrix_y, rotation_matrix_x


def translation_matrix(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def rotation_matrix_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def rotation_matrix_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def scale_matrix(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def perspective_matrix(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    return np.array([
        [f / aspect, 0,  0,                                0],
        [0,          f,  0,                                0],
        [0,          0,  (far + near) / (near - far),
         (2 * far * near) / (near - far)],
        [0,          0,  -1,                               0]
    ], dtype=np.float32)


def view_matrix(camera_pos, yaw, pitch):
    T = translation_matrix(-camera_pos[0], -camera_pos[1], -camera_pos[2])
    R_yaw = rotation_matrix_y(-yaw)
    R_pitch = rotation_matrix_x(-pitch)
    return R_pitch @ R_yaw @ T


# ----------------------------------------------------------------------
# Shape generators create_cube, create_sphere, create_cylinder, create_toru
# ----------------------------------------------------------------------
def ensure_outward_normals(verts, tris, norms):
    fixed_norms = []
    for i, tri in enumerate(tris):
        v0 = verts[tri[0]]
        if np.dot(norms[i], v0) < 0:
            fixed_norms.append(-norms[i])
        else:
            fixed_norms.append(norms[i])
    return fixed_norms


def create_cube():
    verts = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # front (z=-1)
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]   # back  (z=1)
    ], dtype=np.float32)
    tris = [
        (0, 2, 1), (0, 3, 2),  # front  – outward (0,0,-1)
        (4, 5, 7), (5, 6, 7),  # back   – outward (0,0,1)
        (1, 5, 6), (1, 6, 2),  # right  – outward (1,0,0)
        (4, 0, 3), (4, 3, 7),  # left   – outward (-1,0,0)
        (3, 2, 6), (3, 6, 7),  # top    – outward (0,1,0)
        (4, 5, 1), (4, 1, 0)   # bottom – outward (0,-1,0)
    ]
    norms = []
    for tri in tris:

        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        n = np.cross(v1-v0, v2-v0)
        n = n / np.linalg.norm(n)
        norms.append(n)
    norms = ensure_outward_normals(verts, tris, norms)
    return verts, tris, norms


def create_sphere(radius=1.0, stacks=12, sectors=12):
    verts = []
    for i in range(stacks+1):
        theta = i * np.pi / stacks
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        for j in range(sectors):
            phi = j * 2 * np.pi / sectors
            sin_p, cos_p = np.sin(phi), np.cos(phi)
            x = radius * sin_t * cos_p
            y = radius * cos_t
            z = radius * sin_t * sin_p
            verts.append([x, y, z])
    verts = np.array(verts, dtype=np.float32)
    tris = []
    for i in range(stacks):
        for j in range(sectors):
            curr = i * sectors + j
            next_j = i * sectors + (j+1) % sectors
            down = (i+1) * sectors + j
            down_next = (i+1) * sectors + (j+1) % sectors
            tris.append((curr, down, next_j))
            tris.append((next_j, down, down_next))
    norms = []
    for tri in tris:
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        n = np.cross(v1-v0, v2-v0)
        n = n / np.linalg.norm(n)
        norms.append(n)
    norms = ensure_outward_normals(verts, tris, norms)
    return verts, tris, norms


def create_cylinder(radius=1.0, height=2.0, sectors=12):
    verts = []
    # top ring (y = height/2)
    for j in range(sectors):
        phi = j * 2*np.pi / sectors
        x = radius * np.cos(phi)
        z = radius * np.sin(phi)
        verts.append([x, height/2, z])
    # bottom ring (y = -height/2)
    for j in range(sectors):
        phi = j * 2*np.pi / sectors
        x = radius * np.cos(phi)
        z = radius * np.sin(phi)
        verts.append([x, -height/2, z])
    verts = np.array(verts, dtype=np.float32)
    tris = []
    for j in range(sectors):
        t0 = j
        t1 = (j+1) % sectors
        b0 = sectors + j
        b1 = sectors + (j+1) % sectors
        tris.append((t0, t1, b0))
        tris.append((t1, b1, b0))
    norms = []
    for tri in tris:
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        n = np.cross(v1-v0, v2-v0)
        n = n / np.linalg.norm(n)
        norms.append(n)
    norms = ensure_outward_normals(verts, tris, norms)
    return verts, tris, norms


def create_cone(radius=1.0, height=2.0, sectors=12):
    verts = []
    verts.append([0, height/2, 0])
    # base ring (y = -height/2)
    for j in range(sectors):
        phi = j * 2*np.pi / sectors
        x = radius * np.cos(phi)
        z = radius * np.sin(phi)
        verts.append([x, -height/2, z])
    # base centre
    center_idx = len(verts)
    verts.append([0, -height/2, 0])
    verts = np.array(verts, dtype=np.float32)
    tris = []
    # side triangles
    for j in range(sectors):
        apex = 0
        b1 = 1 + j
        b2 = 1 + (j+1) % sectors
        tris.append((apex, b1, b2))
    # base disc (fan)
    for j in range(sectors):
        b1 = 1 + j
        b2 = 1 + (j+1) % sectors
        tris.append((center_idx, b2, b1))
    norms = []
    for tri in tris:
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        n = np.cross(v1-v0, v2-v0)
        n = n / np.linalg.norm(n)
        norms.append(n)
    norms = ensure_outward_normals(verts, tris, norms)
    return verts, tris, norms


def create_torus(R=1.5, r=0.5, sectors=16, sides=8):
    verts = []
    for i in range(sectors):
        phi = i * 2*np.pi / sectors
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        for j in range(sides):
            theta = j * 2*np.pi / sides
            x = (R + r * np.cos(theta)) * cos_phi
            y = r * np.sin(theta)
            z = (R + r * np.cos(theta)) * sin_phi
            verts.append([x, y, z])
    verts = np.array(verts, dtype=np.float32)
    tris = []
    for i in range(sectors):
        for j in range(sides):
            i_next = (i+1) % sectors
            j_next = (j+1) % sides
            v0 = i * sides + j
            v1 = i * sides + j_next
            v2 = i_next * sides + j
            v3 = i_next * sides + j_next
            tris.append((v0, v2, v1))
            tris.append((v1, v2, v3))
    norms = []
    for tri in tris:
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        n = np.cross(v1-v0, v2-v0)
        n = n / np.linalg.norm(n)
        norms.append(n)
    norms = ensure_outward_normals(verts, tris, norms)
    return verts, tris, norms


# ----------------------------------------------------------------------
# Lighting
# ----------------------------------------------------------------------
light_dir = np.array([0.4, -1.0, 0.5], dtype=np.float32)
light_dir = light_dir / np.linalg.norm(light_dir)
ambient_intensity = 0.25
diffuse_intensity = 0.75

# ----------------------------------------------------------------------
# Numba‑optimised triangle rasteriser
# ----------------------------------------------------------------------


@njit
def fill_flat_bottom(x0, y0, z0, x1, y1, z1, x2, y2, z2, color, depth_buffer, screen_array):
    """v0 is the top vertex, v1 and v2 are bottom vertices (same y, y1==y2)."""
    height = y2 - y0
    if height <= 0:
        return
    # pre‑compute steps
    dxl = (x1 - x0) / height
    dxr = (x2 - x0) / height
    dzl = (z1 - z0) / height
    dzr = (z2 - z0) / height

    xl = x0
    xr = x0
    zl = z0
    zr = z0
    for y in range(max(0, int(y0)), min(SCREEN_HEIGHT, int(y2) + 1)):
        # integer pixel bounds
        x_start = int(xl)
        x_end = int(xr)
        if x_start > x_end:
            x_start, x_end = x_end, x_start
        x_start = max(0, x_start)
        x_end = min(SCREEN_WIDTH - 1, x_end)

        if x_end >= x_start:
            # interpolate depth along the scanline
            z = zl
            dz = (zr - zl) / (x_end - x_start + 1) if x_end >= x_start else 0.0
            for x in range(x_start, x_end + 1):
                if z < depth_buffer[x, y]:
                    depth_buffer[x, y] = z
                    screen_array[x, y, 0] = color[0]
                    screen_array[x, y, 1] = color[1]
                    screen_array[x, y, 2] = color[2]
                z += dz

        # advance edge parameters
        xl += dxl
        xr += dxr
        zl += dzl
        zr += dzr


@njit
def fill_flat_top(x0, y0, z0, x1, y1, z1, x2, y2, z2, color, depth_buffer, screen_array):
    """v0 is the bottom vertex, v1 and v2 are top vertices (same y, y1==y2)."""
    height = y0 - y1
    if height <= 0:
        return
    # pre‑compute steps
    dxl = (x0 - x1) / height
    dxr = (x0 - x2) / height
    dzl = (z0 - z1) / height
    dzr = (z0 - z2) / height

    xl = x1
    xr = x2
    zl = z1
    zr = z2
    for y in range(max(0, int(y1)), min(SCREEN_HEIGHT, int(y0) + 1)):
        x_start = int(xl)
        x_end = int(xr)
        if x_start > x_end:
            x_start, x_end = x_end, x_start
        x_start = max(0, x_start)
        x_end = min(SCREEN_WIDTH - 1, x_end)

        if x_end >= x_start:
            z = zl
            dz = (zr - zl) / (x_end - x_start + 1) if x_end >= x_start else 0.0
            for x in range(x_start, x_end + 1):
                if z < depth_buffer[x, y]:
                    depth_buffer[x, y] = z
                    screen_array[x, y, 0] = color[0]
                    screen_array[x, y, 1] = color[1]
                    screen_array[x, y, 2] = color[2]
                z += dz

        xl += dxl
        xr += dxr
        zl += dzl
        zr += dzr


@njit
def rasterize_triangle(p0_x, p0_y, p0_z,
                       p1_x, p1_y, p1_z,
                       p2_x, p2_y, p2_z,
                       color, depth_buffer, screen_array):
    """Sort vertices by y and dispatch to flat-bottom / flat-top fillers."""
    # sort by y increasing (p0 topmost, p2 bottommost)
    if p0_y > p1_y:
        p0_x, p1_x = p1_x, p0_x
        p0_y, p1_y = p1_y, p0_y
        p0_z, p1_z = p1_z, p0_z
    if p0_y > p2_y:
        p0_x, p2_x = p2_x, p0_x
        p0_y, p2_y = p2_y, p0_y
        p0_z, p2_z = p2_z, p0_z
    if p1_y > p2_y:
        p1_x, p2_x = p2_x, p1_x
        p1_y, p2_y = p2_y, p1_y
        p1_z, p2_z = p2_z, p1_z

    # Flat top (p0.y == p1.y)
    if p0_y == p1_y:
        if p0_x > p1_x:
            p0_x, p1_x = p1_x, p0_x
            p0_z, p1_z = p1_z, p0_z
        fill_flat_top(p2_x, p2_y, p2_z,
                      p0_x, p0_y, p0_z,
                      p1_x, p1_y, p1_z,
                      color, depth_buffer, screen_array)
        return

    # Flat bottom (p1.y == p2.y)
    if p1_y == p2_y:
        if p1_x > p2_x:
            p1_x, p2_x = p2_x, p1_x
            p1_z, p2_z = p2_z, p1_z
        fill_flat_bottom(p0_x, p0_y, p0_z,
                         p1_x, p1_y, p1_z,
                         p2_x, p2_y, p2_z,
                         color, depth_buffer, screen_array)
        return

    # General triangle – split at the middle vertex
    alpha = (p1_y - p0_y) / (p2_y - p0_y)
    mid_x = p0_x + alpha * (p2_x - p0_x)
    mid_z = p0_z + alpha * (p2_z - p0_z)

    # Left / right determination
    if mid_x < p1_x:
        fill_flat_bottom(p0_x, p0_y, p0_z,
                         mid_x, p1_y, mid_z,
                         p1_x, p1_y, p1_z,
                         color, depth_buffer, screen_array)
        fill_flat_top(p2_x, p2_y, p2_z,
                      mid_x, p1_y, mid_z,
                      p1_x, p1_y, p1_z,
                      color, depth_buffer, screen_array)
    else:
        fill_flat_bottom(p0_x, p0_y, p0_z,
                         p1_x, p1_y, p1_z,
                         mid_x, p1_y, mid_z,
                         color, depth_buffer, screen_array)
        fill_flat_top(p2_x, p2_y, p2_z,
                      p1_x, p1_y, p1_z,
                      mid_x, p1_y, mid_z,
                      color, depth_buffer, screen_array)

# ----------------------------------------------------------------------
# GameObject
# ----------------------------------------------------------------------


class GameObject:
    def __init__(self, position, vertices, triangles, normals, color,
                 scale=(1, 1, 1), rotation_y=0.0, rot_speed=0.0):
        self.position = np.array(position, dtype=np.float32)
        self.vertices = np.array(vertices, dtype=np.float32)
        self.triangles = triangles
        self.normals = np.array(normals, dtype=np.float32)
        self.color = color
        self.scale = np.array(scale, dtype=np.float32)
        self.rotation_y = rotation_y
        self.rot_speed = rot_speed
        # Pre‑built homogeneous vertices (N x 4) for fast transform
        self.hom_vertices = np.hstack(
            [self.vertices, np.ones((len(self.vertices), 1), dtype=np.float32)])

    def get_model_matrix(self):
        return (translation_matrix(*self.position) @
                rotation_matrix_y(self.rotation_y) @
                scale_matrix(*self.scale))

    def update(self, dt):
        self.rotation_y += self.rot_speed * dt


# ----------------------------------------------------------------------
# Pygame init
# ----------------------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("3D Engine - Numba Optimised")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 20)
pygame.event.set_grab(True)
pygame.mouse.set_visible(False)

# ----------------------------------------------------------------------
# Create objects
# ----------------------------------------------------------------------
cube = GameObject((4, 1.2, 3), *create_cube(),
                  (0, 200, 0), scale=(1.2, 1.2, 1.2), rot_speed=0.4)
sphere = GameObject((-4, 1.2, 2), *create_sphere(radius=1.2, stacks=12, sectors=12),
                    (200, 80, 80), rot_speed=0.3)
cylinder = GameObject((0, 1.5, -4), *create_cylinder(radius=0.8, height=2.2, sectors=12),
                      (80, 80, 220), rot_speed=0.5)
torus = GameObject((0, 1.0, 0), *create_torus(R=1.8, r=0.6, sectors=16, sides=8),
                   (220, 140, 40), scale=(0.8, 0.8, 0.8), rot_speed=0.6)
objects = [cube, sphere, cylinder, torus]

# Grid and axes setup
grid_size = 12
grid_spacing = 1.0
grid_lines = []
for i in range(-grid_size, grid_size+1):
    start = (i * grid_spacing, -0.2, -grid_size * grid_spacing)
    end = (i * grid_spacing, -0.2,  grid_size * grid_spacing)
    grid_lines.append((start, end))
    start = (-grid_size * grid_spacing, -0.2, i * grid_spacing)
    end = (grid_size * grid_spacing, -0.2, i * grid_spacing)
    grid_lines.append((start, end))

axes_lines = [
    ((0, 0, 0), (5, 0, 0), (255, 0, 0)),
    ((0, 0, 0), (0, 5, 0), (0, 255, 0)),
    ((0, 0, 0), (0, 0, 5), (0, 0, 255))
]

# Camera
camera_pos = np.array([0.0, 2.0, 8.0], dtype=np.float32)
camera_yaw = 0.0
camera_pitch = 0.0
proj_matrix = perspective_matrix(FOV, ASPECT_RATIO, NEAR_PLANE, FAR_PLANE)


def draw_line_segments(screen, segments_info):
    for start, end, color in segments_info:
        if start is not None and end is not None:
            pygame.draw.line(screen, color, start, end, 1)


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
running = True
while running:
    dt = clock.tick(60) / 1000.0
    if dt > 0.03:
        dt = 0.03

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.MOUSEMOTION and pygame.mouse.get_focused():
            dx, dy = event.rel
            camera_yaw -= dx * MOUSE_SENSITIVITY
            camera_pitch -= dy * MOUSE_SENSITIVITY
            camera_pitch = max(-np.pi/2.2, min(np.pi/2.2, camera_pitch))

    keys = pygame.key.get_pressed()
    move_speed = CAMERA_SPEED * dt
    forward = np.array(
        [np.sin(camera_yaw), 0, np.cos(camera_yaw)], dtype=np.float32)
    right = np.array(
        [np.cos(camera_yaw), 0, -np.sin(camera_yaw)], dtype=np.float32)

    if keys[pygame.K_w]:
        camera_pos -= forward * move_speed
    if keys[pygame.K_s]:
        camera_pos += forward * move_speed
    if keys[pygame.K_a]:
        camera_pos -= right * move_speed
    if keys[pygame.K_d]:
        camera_pos += right * move_speed
    if keys[pygame.K_q]:
        camera_pos[1] -= move_speed
    if keys[pygame.K_e]:
        camera_pos[1] += move_speed

    for obj in objects:
        obj.update(dt)

    view_mat = view_matrix(camera_pos, camera_yaw, camera_pitch)

    # Gradient background
    screen.fill((10, 10, 30))
    for y in range(SCREEN_HEIGHT):
        t = y / SCREEN_HEIGHT
        val = 20 + int(40 * (1-t))
        pygame.draw.line(screen, (10, 15, val), (0, y), (SCREEN_WIDTH, y))

    # Re‑acquiring the screen pixel array after drawing the background
    screen_array = pygame.surfarray.pixels3d(screen)  # shape (W, H, 3) uint8
    depth_buffer = np.full((SCREEN_WIDTH, SCREEN_HEIGHT),
                           np.inf, dtype=np.float32)

    # ------------------------------------------------------------------
    # Draw objects – per‑triangle transform + Numba rasteriser
    # ------------------------------------------------------------------
    for obj in objects:
        model = obj.get_model_matrix()
        mvp = proj_matrix @ view_mat @ model

        # only keeping the world vertices for lighting / culling
        world_verts = (model @ obj.hom_vertices.T).T[:, :3]   # (N,3)

        for i, tri in enumerate(obj.triangles):
            i0, i1, i2 = tri

            # Transform each vertex separately
            v0_clip = mvp @ obj.hom_vertices[i0]
            v1_clip = mvp @ obj.hom_vertices[i1]
            v2_clip = mvp @ obj.hom_vertices[i2]

            # Near-plane culling
            if v0_clip[3] <= 1e-6 or v1_clip[3] <= 1e-6 or v2_clip[3] <= 1e-6:
                continue

            # Perspective divide
            inv_w0 = 1.0 / v0_clip[3]
            inv_w1 = 1.0 / v1_clip[3]
            inv_w2 = 1.0 / v2_clip[3]

            sx0 = (v0_clip[0] * inv_w0 + 1.0) * SCREEN_WIDTH / 2.0
            sy0 = (1.0 - v0_clip[1] * inv_w0) * SCREEN_HEIGHT / 2.0
            sz0 = v0_clip[2] * inv_w0

            sx1 = (v1_clip[0] * inv_w1 + 1.0) * SCREEN_WIDTH / 2.0
            sy1 = (1.0 - v1_clip[1] * inv_w1) * SCREEN_HEIGHT / 2.0
            sz1 = v1_clip[2] * inv_w1

            sx2 = (v2_clip[0] * inv_w2 + 1.0) * SCREEN_WIDTH / 2.0
            sy2 = (1.0 - v2_clip[1] * inv_w2) * SCREEN_HEIGHT / 2.0
            sz2 = v2_clip[2] * inv_w2

            # Lighting
            world_normal = model[:3, :3] @ obj.normals[i]
            intensity = ambient_intensity + \
                max(0.0, np.dot(world_normal, -light_dir)) * diffuse_intensity
            r = max(0, min(255, int(obj.color[0] * intensity)))
            g = max(0, min(255, int(obj.color[1] * intensity)))
            b = max(0, min(255, int(obj.color[2] * intensity)))
            color = (np.uint8(r), np.uint8(g), np.uint8(b))

            rasterize_triangle(sx0, sy0, sz0,
                               sx1, sy1, sz1,
                               sx2, sy2, sz2,
                               color, depth_buffer, screen_array)

    # Releasing the screen array so Pygame can use it again
    del screen_array

    # Draw grid and axes
    grid_segments = []
    mvp_grid = proj_matrix @ view_mat
    for start, end in grid_lines:
        s4 = np.append(np.array(start, dtype=np.float32), 1.0)
        e4 = np.append(np.array(end, dtype=np.float32), 1.0)
        s_t = mvp_grid @ s4
        e_t = mvp_grid @ e4
        if s_t[3] > NEAR_PLANE and e_t[3] > NEAR_PLANE:
            sx = int((s_t[0]/s_t[3] + 1) * SCREEN_WIDTH/2)
            sy = int((1 - s_t[1]/s_t[3]) * SCREEN_HEIGHT/2)
            ex = int((e_t[0]/e_t[3] + 1) * SCREEN_WIDTH/2)
            ey = int((1 - e_t[1]/e_t[3]) * SCREEN_HEIGHT/2)
            grid_segments.append(((sx, sy), (ex, ey), (70, 70, 90)))
    draw_line_segments(screen, grid_segments)

    axis_segments = []
    for start, end, color in axes_lines:
        s4 = np.append(np.array(start, dtype=np.float32), 1.0)
        e4 = np.append(np.array(end, dtype=np.float32), 1.0)
        s_t = mvp_grid @ s4
        e_t = mvp_grid @ e4
        if s_t[3] > NEAR_PLANE and e_t[3] > NEAR_PLANE:
            sx = int((s_t[0]/s_t[3] + 1) * SCREEN_WIDTH/2)
            sy = int((1 - s_t[1]/s_t[3]) * SCREEN_HEIGHT/2)
            ex = int((e_t[0]/e_t[3] + 1) * SCREEN_WIDTH/2)
            ey = int((1 - e_t[1]/e_t[3]) * SCREEN_HEIGHT/2)
            axis_segments.append(((sx, sy), (ex, ey), color))
    draw_line_segments(screen, axis_segments)

    # HUD
    fps = int(clock.get_fps())
    info = f"FPS: {fps}  Pos: ({camera_pos[0]:.1f}, {camera_pos[1]:.1f}, {camera_pos[2]:.1f})"
    surf = font.render(info, True, (220, 220, 220))
    screen.blit(surf, (10, 10))

    cx, cy = SCREEN_WIDTH//2, SCREEN_HEIGHT//2
    pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 4, 1)
    pygame.draw.line(screen, (255, 255, 255), (cx-10, cy), (cx+10, cy), 1)
    pygame.draw.line(screen, (255, 255, 255), (cx, cy-10), (cx, cy+10), 1)

    pygame.display.flip()

pygame.quit()
