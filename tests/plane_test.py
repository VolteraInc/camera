"""
Tests for the analysis plane class.
"""
import pytest
import numpy as np
from volteracamera.analysis.plane import Plane
from volteracamera.analysis.line import Line

def test_default_initializer():
    """
    Defaults test
    """
    plane = Plane()
    assert [0, 0, 0] == pytest.approx(plane.point)
    assert [0, 0, 1] == pytest.approx(plane.normal)

def test_good_initialization():
    """
    Test initializer with non-default values.
    """
    plane = Plane(point_on_plane = np.array([1.0, 1.1, 1.2]), normal = np.array([1., -1., 0.]))
    assert [1.0, 1.1, 1.2] == pytest.approx(plane.point)
    assert [1.0/np.sqrt(2), -1.0/np.sqrt(2), 0.0] == pytest.approx(plane.normal)

def test_bad_input_size_initializer():
    """
    Test incorrect dimension of input arguements.
    """
    with pytest.raises(TypeError):
        Plane(point_on_plane = 5.0)
    with pytest.raises(TypeError):
        Plane(normal = [5.0, 2.0])


def test_plane_initialization_bad_normal():
    """
    Test plane initializer with bad normal (0)
    """
    with pytest.raises(ZeroDivisionError):
        Plane(normal = [0, 0, 0]) 

def test_line_from_two_planes_working():
    """
    Test the generation of a line from two planes.
    """
    plane1 = Plane([0, 0, 0], [0, 0, 1])
    plane2 = Plane([0, 0, 0], [1, 0, 0])

    line = plane1.intersection_line(plane2)

    assert [0, 1, 0] == pytest.approx(line.direction)
    assert [0, 0, 0] == pytest.approx(line.point)


def test_line_from_two_planes_parallel():
    """
    Test the fault case for two planes not intersecting / parallel
    """
    plane1 = Plane()
    plane2 = Plane()
    with pytest.raises(RuntimeError):
        plane1.intersection_line(plane2)

    plane1 = Plane(point_on_plane = [1, 1, 50], normal =[1, 0, 1] )
    plane2 = Plane(point_on_plane = [50, 2, -5], normal = [-1, 0, -1])
    with pytest.raises(RuntimeError):
        plane1.intersection_line(plane2)

def test_line_from_two_planes_invalid_input():
    """
    Test the fault case for the intersection of two planes
    """
    plane = Plane()
    with pytest.raises(TypeError):
        plane.intersection_line("not a plane")

def test_point_from_line_plane_parallel():
    """
    Check that a runtime error is raised if the line lies in the plane.
    """
    plane = Plane()
    line = Line([0, 0, 0], [1, 0, 0])
    with pytest.raises(RuntimeError):
        plane.intersection_point (line)

def test_point_from_line_plane_bad_input():
    """
    Test bad inputs to line plane intersection
    """
    plane = Plane()
    with pytest.raises(TypeError):
        plane.intersection_point ("not a line.")

def test_point_from_line_plane_good():
    """
    Test that line/plane intersection.
    """
    plane = Plane ([1, 1, 10], [0, 0, 1])
    line = Line([1, 1, 0], [1, 1, 1])

    assert pytest.approx(plane.intersection_point(line)) == [11, 11, 10]
