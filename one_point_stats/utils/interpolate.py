import numpy as np

__all__ = ['box_degrade']


def box_degrade(cube, factor):
    """
    Degrade the resolution of the cube by box averaging.

    Parameters
    ----------
    cube: array-like
        A 3D cube.
    factor: integer
        Degrade the cube by this factor.
        Ex. for a 16x16x16 cube, factor=4 will return a 4x4x4 cube.

    """
    new_shape = (np.array(cube.shape) / factor).astype(int)
    x, y, z = np.mgrid[0:new_shape[0], 0:new_shape[1], 0:new_shape[2]]
    new_cube = np.empty(new_shape)
    for i, j, k in zip(x.ravel(), y.ravel(), z.ravel()):
        new_cube[i, j, k] = np.average(cube[i*factor:(i+1)*factor,
                                       j*factor:(j+1)*factor,
                                       k*factor:(k+1)*factor])
    return new_cube
