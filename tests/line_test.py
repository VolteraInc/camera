"""
Tests of the line class.
"""
import pytest
import numpy as np
from volteracamera.analysis.line import Line

def test_constructor_incorrect_inputs():
    """
    Test the line constructor given incorrect inputs.
    """
    with pytest.raises(TypeError):
        Line(point = 5)
    with pytest.raises(TypeError):
        Line(direction = [1, 2, 3, 4])

def test_constructor_default_inputs():
    """
    test default inputs.
    """
    line = Line ()
    assert np.array_equal([0, 0, 0], line.point)
    assert np.array_equal([0, 0, 1],line.direction)

def test_constructor_inputs():
    """
    test default inputs.
    """
    line = Line (point = [1, 2, 3], direction=[0, 1, -6])
    assert np.array_equal([1, 2, 3], line.point)
    assert np.array_equal([0, 1, -6], line.direction)

def test_point_on_line():
    """
    Test getting a point along a line.
    """
    line = Line(point = [1, 0, 0], direction = [0, 1, 1])
    assert np.array_equal([1, 1, 1], line.get_point_on_line(1))
    assert np.array_equal([1, -1, -1], line.get_point_on_line(-1))
    assert np.array_equal([1, 10.5, 10.5], line.get_point_on_line(10.5))

def test_distance_between_lines():
    """
    Test the minimum distance between lines method.
    """
    line1 = Line(point=[1, 1, 1], direction = [1, 1, 1])
    line2 = Line(point = [1, 2, 0], direction = [0, 1, 1])
    
    assert line1.distance_to(line2) == pytest.approx(np.sqrt(2))

def test_distance_between_lines_crossing():
    """
    Test the minimum distance between lines method.
    """
    line1 = Line(point=[1, 1, 1], direction = [1, 1, 1])
    line2 = Line(point = [1, 1, 1], direction = [0, 1, 1])
    
    assert line1.distance_to(line2) == 0

def test_distance_between_parallele_lines():
    """
    Test the minimum distance between lines method.
    """
    line1 = Line( direction = [1, 1, 1])
    line2 = Line(point = [1, 0, 0], direction = [1, 1, 1])
    
    assert line1.distance_to(line2) == 1


    
