import re
import sys
import math

import numpy as np


def parse_face(items):
    pat = re.compile('([0-9]+)\/([0-9]+)\/([0-9]+)')
    mat = pat.search(items[0])
    if mat is None:
        pat = re.compile('([0-9]+)\/\/([0-9]+)')
        mat = pat.search(items[0])

    if mat is None:
        pat = re.compile('([0-9]+)\/([0-9]+)')
        mat = pat.search(items[0])

    if mat is None:
        pat = re.compile('([0-9]+)')
        mat = pat.search(items[0])

    indices = [int(pat.search(it).group(1)) - 1 for it in items]
    return indices


class Vector(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def cross(self, v):
        x = self.y * v.z - self.z * v.y
        y = self.z * v.x - self.x * v.z
        z = self.x * v.y - self.y * v.x
        return Vector(x, y, z)

    def normalize(self):
        try:
            self = self / self.norm()
        except ZeroDivisionError:
            raise Exception('Vector length is zero!')

        return self

    def norm(self):
        return math.sqrt(self.norm2())

    def norm2(self):
        return self.dot(self)

    def __add__(self, v):
        return Vector(self.x + v.x, self.y + v.y, self.z + v.z)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __sub__(self, v):
        return Vector(self.x - v.x, self.y - v.y, self.z - v.z)

    def __mul__(self, v):
        if isinstance(v, Vector):
            return Vector(self.x * v.x, self.y * v.y, self.z * v.z)
        else:
            return Vector(self.x * v, self.y * v, self.z * v)

    def __rmul__(self, v):
        return self.__mul__(v)

    def __truediv__(self, v):
        if isinstance(v, Vector):
            if v.x == 0.0 or v.y == 0.0 or v.z == 0.0:
                raise ZeroDivisionError()
            return Vector(self.x / v.x, self.y / v.y, self.z / v.z)
        else:
            if v == 0.0:
                raise ZeroDivisionError()
            return Vector(self.x / v, self.y / v, self.z / v)

    def __repr__(self):
        return '( %.4f, %.4f, %.4f )' % (self.x, self.y, self.z)


class ObjMesh(object):
    def __init__(self, filename=None):
        self._vertices = np.array([], dtype=np.float32)
        self.vs = np.array([], dtype=np.float32)
        self.faces = np.array([], dtype=np.float32)
        self._normals = np.array([], dtype=np.float32)
        self._texcoords = np.array([], dtype=np.float32)
        self._indices = np.array([], dtype=np.uint32)

        if filename is not None:
            self.load(filename)

    # Property for vertices
    def _get_vertices(self):
        return self._vertices.reshape((self.n_vertices, 3))

    def _set_vertices(self, vertices):
        self._vertices = np.array(vertices, dtype=np.float32).flatten()
        self.compute_normals()

    vertices = property(_get_vertices, _set_vertices)

    # Property for normals
    def _get_normals(self):
        return self._normals.reshape((self.n_vertices, 3))

    def _set_normals(self, normals):
        self._normals = np.array(normals, dtype=np.float32).flatten()

    normals = property(_get_normals, _set_normals)

    # Property for texcoords
    def _get_texcoords(self):
        return self._texcoords.reshape((self.n_vertices, 2))

    def _set_texcoords(self, texcoords):
        self._texcoords = np.array(texcoords, dtype=np.float32).flatten()

    texcoords = property(_get_texcoords, _set_texcoords)

    # Property for indices
    def _get_indices(self):
        return self._indices

    def _set_indices(self, indices):
        self._indices = np.array(indices, dtype=np.uint32).flatten()

    indices = property(_get_indices, _set_indices)

    # Load mesh
    def load(self, filename, verbose=False):
        with open(filename, 'r') as f:
            lines = [l.strip() for l in f]
            lines = filter(lambda l: l != '' and not l.startswith('#'), lines)

            vertices = []
            normals = []
            texcoords = []
            indices = []
            for l in lines:
                it = [x for x in re.split('\s+', l.strip())]
                if it[0] == 'v':
                    it = [float(i) for i in it[1:]]
                    vertices.append(it)

                elif it[0] == 'vt':
                    texcoords.append((float(it[1]), float(it[2])))

                elif it[0] == 'vn':
                    it = [float(i) for i in it[1:]]
                    normals.append(it)

                elif it[0] == 'f':
                    it = it[1:]
                    indices.append(parse_face(it))
                elif verbose:
                    sys.stderr.write('Unknown identifier: {}\n'.format(it[0]))

            if len(indices) > 0:
                self._indices = np.array(indices, dtype='uint32').flatten()

            if len(vertices) > 0:
                self._vertices = np.array(vertices, dtype='float32').flatten()

            if len(normals) > 0:
                self._normals = np.array(normals, dtype='float32').flatten()

            if len(texcoords) > 0:
                self._texcoords = np.array(texcoords, dtype='float32').flatten()

    def save(self, filename):
        assert len(self.vertices) > 0
        assert self._vertices.ndim == 1
        assert self._vertices.dtype == np.float32
        assert self._indices.ndim == 1
        assert self._indices.dtype == np.uint32

        with open(filename, 'w') as fp:
            # Write positions
            for i in range(0, self._vertices.size, 3):
                x = self._vertices[i + 0]
                y = self._vertices[i + 1]
                z = self._vertices[i + 2]
                fp.write('v {0:.8f} {1:.8f} {2:.8f}\n'.format(x, y, z))

            # Write normals
            has_normal = False
            if self._normals.size > 0:
                has_normal = True
                for i in range(0, self._normals.size, 3):
                    x = self._normals[i + 0]
                    y = self._normals[i + 1]
                    z = self._normals[i + 2]
                    fp.write('vn {0:.8f} {1:.8f} {2:.8f}\n'.format(x, y, z))

            # Write texcoords
            has_texcoord = False
            if self._texcoords.size > 0:
                has_texcoord = True
                for i in range(0, self._texcoords.size, 2):
                    x = self._texcoords[i + 0]
                    y = self._texcoords[i + 1]
                    fp.write('vt {0:.8f} {1:.8f}\n'.format(x, y))

            # Write indices
            for i in range(0, len(self._indices), 3):
                i0 = self._indices[i + 0] + 1
                i1 = self._indices[i + 1] + 1
                i2 = self._indices[i + 2] + 1

                if has_normal and has_texcoord:
                    fp.write('f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}\n'.format(i0, i1, i2))

                elif has_texcoord:
                    fp.write('f {0}/{0} {1}/{1} {2}/{2}\n'.format(i0, i1, i2))

                elif has_normal:
                    fp.write('f {0}//{0} {1}//{1} {2}//{2}\n'.format(i0, i1, i2))

                else:
                    fp.write('f {0} {1} {2}\n'.format(i0, i1, i2))

    def compute_normals(self):
        vectors = [
            Vector(self._vertices[i + 0], self._vertices[i + 1], self._vertices[i + 2])
            for i in range(0, self._vertices.size, 3)
        ]
        normals = [Vector(0.0, 0.0, 0.0) for i in range(self.n_vertices)]

        for i in range(0, self._indices.size, 3):
            i0 = self._indices[i + 0]
            i1 = self._indices[i + 1]
            i2 = self._indices[i + 2]
            v0 = vectors[i0]
            v1 = vectors[i1]
            v2 = vectors[i2]
            normal = (v1 - v0).cross(v2 - v0)
            normals[i0] += normal
            normals[i1] += normal
            normals[i2] += normal

        for i in range(self.n_vertices):
            n = normals[i]
            l = n.norm()
            if l != 0.0:
                normals[i] /= l

        self._normals = np.asarray([(n.x, n.y, n.z) for n in normals], dtype='float32').flatten()

    @staticmethod
    def from_data(indices, positions, texcoords=None, normals=None):
        obj = ObjMesh()
        obj.indices = indices
        obj.vertices = positions

        if texcoords is not None:
            obj.texcoords = texcoords

        if normals is not None:
            obj.normals = normals

        return obj

    @property
    def n_vertices(self):
        return self._vertices.size // 3
