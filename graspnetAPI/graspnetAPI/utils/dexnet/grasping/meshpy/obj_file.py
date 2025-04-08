"""
File for loading and saving meshes from .OBJ files
Author: Jeff Mahler
"""
import os
try:
    from . import mesh
except ImportError:
    import mesh


class ObjFile(object):
    """
    A Wavefront .obj file reader and writer.

    Attributes
    ----------
    filepath : :obj:`str`
        The full path to the .obj file associated with this reader/writer.
    """

    def __init__(self, filepath):
        """Construct and initialize a .obj file reader and writer.

        Parameters
        ----------
        filepath : :obj:`str`
            The full path to the desired .obj file

        Raises
        ------
        ValueError
            If the file extension is not .obj.
        """
        self.filepath_ = filepath
        file_root, file_ext = os.path.splitext(self.filepath_)
        if file_ext != '.obj':
            raise ValueError('Extension %s invalid for OBJs' %(file_ext))

    @property
    def filepath(self):
        """Returns the full path to the .obj file associated with this reader/writer.

        Returns
        -------
        :obj:`str`
            The full path to the .obj file associated with this reader/writer.
        """
        return self.filepath_

    def read(self):
        """Reads in the .obj file and returns a Mesh3D representation of that mesh.

        Returns
        -------
        :obj:`Mesh3D`
            A Mesh3D created from the data in the .obj file.
        """
        numVerts = 0  
        verts = []
        norms = None
        faces = []
        tex_coords = []
        face_norms = []
        f = open(self.filepath_, 'r')

        for line in f:  
            # Break up the line by whitespace
            vals = line.split()
            if len(vals) > 0:
                # Look for obj tags (see http://en.wikipedia.org/wiki/Wavefront_.obj_file)
                if vals[0] == 'v':
                    # Add vertex
                    v = list(map(float, vals[1:4]))
                    verts.append(v)
                if vals[0] == 'vn':
                    # Add normal
                    if norms is None:
                        norms = []
                    n = list(map(float, vals[1:4]))
                    norms.append(n)  
                if vals[0] == 'f':
                    # Add faces (includes vertex indices, texture coordinates, and normals)
                    vi = []
                    vti = []
                    nti = []
                    if vals[1].find('/') == -1:
                        vi = list(map(int, vals[1:]))
                        vi = [i - 1 for i in vi]
                    else:
                        for j in range(1, len(vals)):
                            # Break up like by / to read vert inds, tex coords, and normal inds
                            val = vals[j]
                            tokens = val.split('/')
                            for i in range(len(tokens)):
                                if i == 0:
                                    vi.append(int(tokens[i]) - 1) # adjust for python 0 - indexing
                                elif i == 1:
                                    if tokens[i] != '':
                                        vti.append(int(tokens[i]))
                                elif i == 2:
                                    nti.append(int(tokens[i]))
                    faces.append(vi)
                    # Below two lists are currently not in use
                    tex_coords.append(vti)
                    face_norms.append(nti)
        f.close()

        return mesh.Mesh3D(verts, faces, norms)

    def write(self, mesh):
        """Writes a Mesh3D object out to a .obj file format

        Parameters
        ----------
        mesh : :obj:`Mesh3D`
            The Mesh3D object to write to the .obj file.

        Note
        ----
        Does not support material files or texture coordinates.
        """
        f = open(self.filepath_, 'w')
        vertices = mesh.vertices
        faces = mesh.triangles
        normals = mesh.normals

        # write human-readable header
        f.write('###########################################################\n')
        f.write('# OBJ file generated by UC Berkeley Automation Sciences Lab\n')
        f.write('#\n')
        f.write('# Num Vertices: %d\n' %(vertices.shape[0]))
        f.write('# Num Triangles: %d\n' %(faces.shape[0]))
        f.write('#\n')
        f.write('###########################################################\n')
        f.write('\n')

        for v in vertices:
            f.write('v %f %f %f\n' %(v[0], v[1], v[2]))

        # write the normals list
        if normals is not None and normals.shape[0] > 0:
            for n in normals:
                f.write('vn %f %f %f\n' %(n[0], n[1], n[2]))

        # write the normals list
        for t in faces:
            f.write('f %d %d %d\n' %(t[0]+1, t[1]+1, t[2]+1)) # convert back to 1-indexing

        f.close()
