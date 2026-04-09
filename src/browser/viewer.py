import math
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *

from browser.models import GpuMesh


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

FRAG_SRC = """#version 330 core
in vec3 vNormal;
in vec3 vPos;
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


# -- 3D viewer widget -------------------------------------------------

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
        self._program = 0
        self._vao = 0
        self._vbo_pos = 0
        self._vbo_nor = 0
        self._ebo = 0
        self._index_count = 0
        self._has_mesh = False

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glClearColor(0.10, 0.10, 0.18, 1.0)
        self._compile_shaders()
        self._setup_buffers()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if not self._has_mesh:
            return
        aspect = self.width() / max(self.height(), 1)
        MVP = self._camera.proj_matrix(aspect) @ self._camera.view_matrix()
        glUseProgram(self._program)
        glUniformMatrix4fv(glGetUniformLocation(self._program, "uMVP"),
                           1, GL_TRUE, MVP.astype(np.float32))
        light = np.array([0.6, 0.8, 0.5], dtype=np.float32)
        light /= np.linalg.norm(light)
        glUniform3fv(glGetUniformLocation(self._program, "uLightDir"), 1, light)
        glUniform3f(glGetUniformLocation(self._program, "uColor"), 0.72, 0.72, 0.76)
        glBindVertexArray(self._vao)
        glDrawElements(GL_TRIANGLES, self._index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def load_mesh(self, mesh: GpuMesh):
        self.makeCurrent()
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, mesh.positions.nbytes,
                     mesh.positions.tobytes(), GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_nor)
        glBufferData(GL_ARRAY_BUFFER, mesh.normals.nbytes,
                     mesh.normals.tobytes(), GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.nbytes,
                     mesh.indices.tobytes(), GL_STATIC_DRAW)
        self._index_count = len(mesh.indices)
        self._has_mesh = True
        self._camera.fit_to_sphere(mesh.center, mesh.radius)
        self.doneCurrent()
        self.update()

    def clear_mesh(self):
        self._has_mesh = False
        self.update()

    def mousePressEvent(self, e):
        self._camera.handle_press(e.position().x(), e.position().y())

    def mouseMoveEvent(self, e):
        self._camera.handle_move(e.buttons(), e.position().x(), e.position().y())
        self.update()

    def wheelEvent(self, e):
        self._camera.handle_scroll(e.angleDelta().y())
        self.update()

    def _compile_shaders(self):
        vs = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vs, VERT_SRC)
        glCompileShader(vs)
        if not glGetShaderiv(vs, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(vs).decode())
        fs = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fs, FRAG_SRC)
        glCompileShader(fs)
        if not glGetShaderiv(fs, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(fs).decode())
        self._program = glCreateProgram()
        glAttachShader(self._program, vs)
        glAttachShader(self._program, fs)
        glLinkProgram(self._program)
        if not glGetProgramiv(self._program, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(self._program).decode())
        glDeleteShader(vs)
        glDeleteShader(fs)

    def _setup_buffers(self):
        self._vao = glGenVertexArrays(1)
        self._vbo_pos, self._vbo_nor = glGenBuffers(2)
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
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, None, GL_STATIC_DRAW)
        glBindVertexArray(0)
