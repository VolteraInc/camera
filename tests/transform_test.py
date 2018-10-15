"""
Tests of the transformation class
"""
import pytest
import numpy as np
import transforms3d as tfd

from volteracamera.analysis.transform import Transform
from volteracamera.analysis.plane import Plane

def test_default_initialization():
    """
    Test default initialization (Null transform) 
    """
    transform = Transform() 
    assert [0, 0, 0] == transform._rotation
    assert [0, 0, 0] == transform._translation
    assert np.array_equal(np.identity(4), transform._matrix)

    assert np.array_equal([1, 2, 3], transform.transform_point([1, 2, 3]))

def test_bad_initialization():
    """
    bad initializers.
    """
    with pytest.raises(TypeError):
        Transform (rotation = 5)
    with pytest.raises(TypeError):
        Transform (translation=[1, 2])

def test_simple_transforms():
    """
    test translation without rotation.
    """
    rot_90_x = [np.pi/2, 0, 0]
    trans = [1, 2, -1]
    point = [1, 1, 0]
    trans_tran_only = Transform(translation = trans)
    trans_rot_only = Transform(rotation = rot_90_x)
    trans_full = Transform(translation = trans, rotation = rot_90_x)
    
    assert np.array_equal([2, 3, -1], trans_tran_only.transform_point(point))
    np.testing.assert_array_almost_equal([1, 0, 1], trans_rot_only.transform_point(point))
    np.testing.assert_array_almost_equal([2, 2, 0], trans_full.transform_point(point))
  
def test_transform_plane():
    """
    Test transfroming a simple plane. 
    """ 
    rot_90_x = [np.pi/2, 0, 0]
    trans = [1, 2, -1]
    transform = Transform(translation = trans, rotation = rot_90_x)
    plane = Plane()

    new_plane = transform.transform_plane(plane)

    np.testing.assert_array_almost_equal(trans, new_plane.point)
    np.testing.assert_array_almost_equal([0, -1, 0], new_plane.normal)

def test_transform_bad_input():
    """
    Test the plane transform for bad inputs.
    """
    transform = Transform()
    with pytest.raises(TypeError):
        transform.transform_point(5) 
  
