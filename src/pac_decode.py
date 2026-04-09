"""Numpy-accelerated vertex and index decode for BlackSpace Engine meshes."""

import numpy as np
from model_types import VertexBuffer, IndexBuffer

PAC_STRIDE = 40


def decode_pac_vertices(data: bytes, section_offset: int, vertex_count: int,
                        center: tuple, half_extent: tuple,
                        vertex_start: int = 0) -> VertexBuffer:
    """Decode PAC vertices (40-byte stride) into numpy arrays."""
    n = vertex_count
    base = section_offset + vertex_start

    buf = np.frombuffer(data, dtype=np.uint8, count=n * PAC_STRIDE, offset=base)
    vert = buf.reshape(n, PAC_STRIDE)

    # Position: 3 x uint16 at +0, dequantize with center/half_extent / 32767
    pos_u16 = vert[:, 0:6].copy().view('<u2')  # (n, 3)
    cx, cy, cz = center
    hx, hy, hz = half_extent
    positions = np.empty((n, 3), dtype=np.float32)
    positions[:, 0] = cx + (pos_u16[:, 0] / 32767.0) * hx
    positions[:, 1] = cy + (pos_u16[:, 1] / 32767.0) * hy
    positions[:, 2] = cz + (pos_u16[:, 2] / 32767.0) * hz

    # UV: 2 x float16 at +8
    uv_f16 = vert[:, 8:12].copy().view('<f2')  # (n, 2)
    uvs = uv_f16.astype(np.float32)

    # Normal: R10G10B10A2 packed uint32 at +16, axes permuted (G, B, R)
    packed = vert[:, 16:20].copy().view('<u4').ravel()  # (n,)
    r_raw = (packed >> 0) & 0x3FF
    g_raw = (packed >> 10) & 0x3FF
    b_raw = (packed >> 20) & 0x3FF
    normals = np.empty((n, 3), dtype=np.float32)
    normals[:, 0] = g_raw / 511.5 - 1.0  # nx = channel G
    normals[:, 1] = b_raw / 511.5 - 1.0  # ny = channel B
    normals[:, 2] = r_raw / 511.5 - 1.0  # nz = channel R

    return VertexBuffer(positions=positions, normals=normals, uvs=uvs)


def decode_pam_vertices(data: bytes, geom_offset: int, byte_offset: int,
                        vertex_count: int, bbox_min: tuple, bbox_max: tuple,
                        stride: int = 20) -> VertexBuffer:
    """Decode PAM vertices (variable stride) into numpy arrays."""
    n = vertex_count
    base = geom_offset + byte_offset
    extent = (bbox_max[0] - bbox_min[0],
              bbox_max[1] - bbox_min[1],
              bbox_max[2] - bbox_min[2])

    buf = np.frombuffer(data, dtype=np.uint8, count=n * stride, offset=base)
    vert = buf.reshape(n, stride)

    # Position: 3 x uint16 at +0, dequantize with bbox_min + u16/65535 * extent
    pos_u16 = vert[:, 0:6].copy().view('<u2')  # (n, 3)
    positions = np.empty((n, 3), dtype=np.float32)
    positions[:, 0] = bbox_min[0] + (pos_u16[:, 0] / 65535.0) * extent[0]
    positions[:, 1] = bbox_min[1] + (pos_u16[:, 1] / 65535.0) * extent[1]
    positions[:, 2] = bbox_min[2] + (pos_u16[:, 2] / 65535.0) * extent[2]

    # UV: 2 x float16 at +8 (if stride >= 12)
    if stride >= 12:
        uv_f16 = vert[:, 8:12].copy().view('<f2')  # (n, 2)
        uvs = uv_f16.astype(np.float32)
    else:
        uvs = np.zeros((n, 2), dtype=np.float32)

    # Normal: R10G10B10A2 at +12 (if stride >= 16), same permutation as PAC
    if stride >= 16:
        packed = vert[:, 12:16].copy().view('<u4').ravel()  # (n,)
        r_raw = (packed >> 0) & 0x3FF
        g_raw = (packed >> 10) & 0x3FF
        b_raw = (packed >> 20) & 0x3FF
        normals = np.empty((n, 3), dtype=np.float32)
        normals[:, 0] = g_raw / 511.5 - 1.0
        normals[:, 1] = b_raw / 511.5 - 1.0
        normals[:, 2] = r_raw / 511.5 - 1.0
    else:
        normals = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (n, 1))

    return VertexBuffer(positions=positions, normals=normals, uvs=uvs)


def decode_indices(data: bytes, offset: int, count: int,
                   index_size: int = 2) -> IndexBuffer:
    """Decode index buffer. index_size=2 for uint16, 4 for uint32."""
    dtype = '<u2' if index_size == 2 else '<u4'
    indices = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
    return IndexBuffer(indices=indices.astype(np.uint32))
