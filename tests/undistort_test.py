"""
Tests for the analysis plane class.
"""
import pytest
import tempfile
import numpy as np
from volteracamera.analysis.undistort import Undistort

def test_default_initializer():
    """
    Defaults test
    """
    undistort = Undistort (camera_matrix = np.identity(3), distortion=[0, 0, 0, 1, 2])
    assert np.identity(3) == pytest.approx(undistort.camera_matrix)
    assert [0, 0, 0, 1, 2] == pytest.approx(undistort.distortion)

def test_bad_initialization():
    """
    Test initializer with bad values
    """
    with pytest.raises(TypeError):
        Undistort (np.identity(3), "not a matrix")
    with pytest.raises(TypeError):
        Undistort ("not a matrix", [1, 2, 3, 4, 5])
    with pytest.raises(RuntimeError):
        Undistort (np.identity(4), [1, 2, 3, 4, 5])
    with pytest.raises(RuntimeError):
        Undistort (np.identity(3), [1, 2, 3, 4])


def test_file_saving_loading():
    """
    Test the file saving and loading.
    """
    undistort = Undistort (camera_matrix = np.identity(3), distortion=[0, 0, 0, 1, 2])
    with tempfile.NamedTemporaryFile() as fid:
        undistort.write_file(fid.name)
        new_object = Undistort.read_json_file(fid.name)
        assert isinstance (new_object, Undistort)
        assert np.identity(3) == pytest.approx(new_object.camera_matrix)
        assert [0, 0, 0, 1, 2] == pytest.approx(new_object.distortion)

     
