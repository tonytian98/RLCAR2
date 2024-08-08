import math

p2 = (2, 0)
p1 = (0, 2)
print(
    math.degrees(
        math.atan2(
            (p1[1] - p2[1]),
            (p1[0] - p2[0]),
        )
    )
    % 360
)

l = [f + "ddd" for f in []]
print(l)

import shapely.geometry as sg
from shapely.geometry import (
    Polygon,
    Point,
    LineString,
    LinearRing,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
    GeometryCollection,
)
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
from Car import Car

# List of coordinates for the polygon
coordinates = [(0, 0), (0, 600), (800, 600), (800, 0)]
track_block1 = [(50, 50), (100, 50), (100, 100), (50, 100)]
track_block2 = [(101, 100), (101, 50), (200, 150), (200, 200)]
car_start = Point(0, 500)
# Create a Shapely Polygon object
polygon = sg.Polygon(coordinates)
print(car_start.distance(polygon))

print(isinstance([1, 2, 4], list))
