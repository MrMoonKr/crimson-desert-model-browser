import ctypes
import math
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *

from browser.models import GpuMesh, SceneMesh, SubmeshInfo, RenderMode


# -- Orbit camera -----------------------------------------------------

class OrbitCamera:
    def __init__(self):
        self.yaw = 0.0
        self.pitch = 0.3
        self.radius = 2.0
        self.target = np.zeros(3, dtype=np.float32)
        self.fov_y = 45.0
        self._last_x = 0
        self._last_y = 0

    def fit_to_sphere(self, center, radius):
        self.target = center.copy()
        half_fov = math.radians(self.fov_y * 0.5)
        self.radius = radius / math.sin(half_fov) * 1.3
        self.yaw = math.pi
        self.pitch = 0.3

    def eye_position(self):
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        return self.target + self.radius * np.array([cp * sy, sp, cp * cy], dtype=np.float32)

    def view_matrix(self):
        eye = self.eye_position()
        fwd = self.target - eye
        fwd_len = np.linalg.norm(fwd)
        if fwd_len < 1e-8:
            return np.eye(4, dtype=np.float32)
        fwd /= fwd_len
        world_up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(fwd, world_up)
        r_len = np.linalg.norm(right)
        if r_len < 1e-8:
            right = np.array([1, 0, 0], dtype=np.float32)
        else:
            right /= r_len
        up = np.cross(right, fwd)

        m = np.eye(4, dtype=np.float32)
        m[0, :3] = right
        m[1, :3] = up
        m[2, :3] = -fwd
        m[0, 3] = -np.dot(right, eye)
        m[1, 3] = -np.dot(up, eye)
        m[2, 3] = np.dot(fwd, eye)
        return m

    def proj_matrix(self, aspect):
        near = max(self.radius * 0.001, 0.001)
        far = self.radius * 100.0
        f = 1.0 / math.tan(math.radians(self.fov_y) * 0.5)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2 * far * near) / (near - far)
        m[3, 2] = -1.0
        return m

    def handle_press(self, x, y):
        self._last_x = x
        self._last_y = y

    def handle_move(self, buttons, x, y):
        dx = x - self._last_x
        dy = y - self._last_y
        self._last_x = x
        self._last_y = y

        if buttons & Qt.MouseButton.LeftButton:
            self.yaw -= dx * 0.005
            self.pitch += dy * 0.005
            self.pitch = max(-1.5, min(1.5, self.pitch))
        elif buttons & Qt.MouseButton.MiddleButton:
            cp, sp = math.cos(self.pitch), math.sin(self.pitch)
            cy, sy = math.cos(self.yaw), math.sin(self.yaw)
            right = np.array([cy, 0, -sy], dtype=np.float32)
            up = np.array([-sp * sy, cp, -sp * cy], dtype=np.float32)
            scale = self.radius * 0.002
            self.target += right * (-dx * scale) + up * (dy * scale)

    def handle_scroll(self, delta):
        self.radius *= 0.9 ** (delta / 120.0)
        self.radius = max(0.01, self.radius)


# -- OpenGL shaders ---------------------------------------------------

VERT_SRC = """#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uMVP;
out vec3 vNormal;
out vec3 vPos;
void main() {
    vPos = aPos;
    vNormal = aNormal;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

VERT_UV_SRC = """#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
uniform mat4 uMVP;
out vec3 vNormal;
out vec2 vUV;
void main() {
    vNormal = aNormal;
    vUV = aUV;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

FRAG_SOLID_SRC = """#version 330 core
in vec3 vNormal;
out vec4 FragColor;
uniform vec3 uLightDir;
uniform vec3 uColor;
void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    float diff = max(abs(dot(N, L)), 0.0);
    vec3 ambient = 0.18 * uColor;
    vec3 diffuse = 0.82 * diff * uColor;
    FragColor = vec4(ambient + diffuse, 1.0);
}
"""

FRAG_WIREFRAME_SRC = """#version 330 core
uniform vec3 uColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(uColor, 1.0);
}
"""

FRAG_NORMALS_SRC = """#version 330 core
in vec3 vNormal;
out vec4 FragColor;
void main() {
    FragColor = vec4(abs(normalize(vNormal)), 1.0);
}
"""

FRAG_UV_SRC = """#version 330 core
in vec2 vUV;
out vec4 FragColor;
void main() {
    float cx = floor(mod(vUV.x * 8.0, 2.0));
    float cy = floor(mod(vUV.y * 8.0, 2.0));
    float checker = mod(cx + cy, 2.0);
    vec3 col = mix(vec3(0.25), vec3(0.85), checker);
    FragColor = vec4(col, 1.0);
}
"""


# -- 3D viewer widget -------------------------------------------------

def _compile_program(vert_src, frag_src):
    vs = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vs, vert_src)
    glCompileShader(vs)
    if not glGetShaderiv(vs, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(vs).decode())
    fs = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fs, frag_src)
    glCompileShader(fs)
    if not glGetShaderiv(fs, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(fs).decode())
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(prog).decode())
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog


class ModelViewer(QOpenGLWidget):
    def __init__(self, parent=None):
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setSamples(4)
        fmt.setDepthBufferSize(24)
        super().__init__(parent)
        self.setFormat(fmt)
        self._camera = OrbitCamera()
        self._programs = {}
        self._vao = 0
        self._vbo_pos = 0
        self._vbo_nor = 0
        self._vbo_uv = 0
        self._ebo = 0
        self._index_count = 0
        self._has_mesh = False
        self._scene = None
        self._render_mode = RenderMode.SOLID

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glClearColor(0.10, 0.10, 0.18, 1.0)
        self._compile_all_shaders()
        self._setup_buffers()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if not self._has_mesh:
            return

        aspect = self.width() / max(self.height(), 1)
        mvp = self._camera.proj_matrix(aspect) @ self._camera.view_matrix()
        program = self._programs[self._render_mode]
        glUseProgram(program)

        glUniformMatrix4fv(glGetUniformLocation(program, "uMVP"),
                           1, GL_TRUE, mvp.astype(np.float32))
        light = np.array([0.6, 0.8, 0.5], dtype=np.float32)
        light /= np.linalg.norm(light)
        light_loc = glGetUniformLocation(program, "uLightDir")
        if light_loc >= 0:
            glUniform3fv(light_loc, 1, light)

        if self._render_mode == RenderMode.WIREFRAME:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glBindVertexArray(self._vao)

        if self._scene is not None:
            for sm in self._scene.submeshes:
                if not sm.visible:
                    continue
                color_loc = glGetUniformLocation(program, "uColor")
                if color_loc >= 0:
                    glUniform3f(color_loc, *sm.base_color)
                glDrawElements(GL_TRIANGLES, sm.index_count, GL_UNSIGNED_INT,
                               ctypes.c_void_p(sm.index_offset))
        else:
            color_loc = glGetUniformLocation(program, "uColor")
            if color_loc >= 0:
                glUniform3f(color_loc, 0.72, 0.72, 0.76)
            glDrawElements(GL_TRIANGLES, self._index_count, GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

        if self._render_mode == RenderMode.WIREFRAME:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def load_mesh(self, mesh):
        """Accept GpuMesh (legacy) or SceneMesh (new)."""
        self.makeCurrent()

        if isinstance(mesh, SceneMesh):
            self._scene = mesh
            positions = mesh.positions
            normals = mesh.normals
            uvs = mesh.uvs
            indices = mesh.indices
        else:
            self._scene = None
            positions = mesh.positions
            normals = mesh.normals
            uvs = np.zeros((len(positions), 2), dtype=np.float32)
            indices = mesh.indices

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes,
                     positions.tobytes(), GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_nor)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes,
                     normals.tobytes(), GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_uv)
        glBufferData(GL_ARRAY_BUFFER, uvs.nbytes,
                     uvs.tobytes(), GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes,
                     indices.tobytes(), GL_STATIC_DRAW)
        self._index_count = len(indices)
        self._has_mesh = True
        self._camera.fit_to_sphere(mesh.center, mesh.radius)
        self.doneCurrent()
        self.update()

    def clear_mesh(self):
        self._has_mesh = False
        self._scene = None
        self.update()

    def set_render_mode(self, mode: RenderMode):
        self._render_mode = mode
        self.update()

    def mousePressEvent(self, e):
        self._camera.handle_press(e.position().x(), e.position().y())

    def mouseMoveEvent(self, e):
        self._camera.handle_move(e.buttons(), e.position().x(), e.position().y())
        self.update()

    def wheelEvent(self, e):
        self._camera.handle_scroll(e.angleDelta().y())
        self.update()

    def _compile_all_shaders(self):
        self._programs[RenderMode.SOLID] = _compile_program(VERT_SRC, FRAG_SOLID_SRC)
        self._programs[RenderMode.WIREFRAME] = _compile_program(VERT_SRC, FRAG_WIREFRAME_SRC)
        self._programs[RenderMode.NORMALS] = _compile_program(VERT_SRC, FRAG_NORMALS_SRC)
        self._programs[RenderMode.UV] = _compile_program(VERT_UV_SRC, FRAG_UV_SRC)

    def _setup_buffers(self):
        self._vao = glGenVertexArrays(1)
        self._vbo_pos, self._vbo_nor, self._vbo_uv = glGenBuffers(3)
        self._ebo = glGenBuffers(1)
        glBindVertexArray(self._vao)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, None)
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_nor)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, None)
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_uv)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_STATIC_DRAW)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8, None)
        glEnableVertexAttribArray(2)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, None, GL_STATIC_DRAW)
        glBindVertexArray(0)
