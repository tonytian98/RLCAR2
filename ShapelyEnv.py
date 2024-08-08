from threading import Thread
from shapely.geometry import Polygon, LineString, Point
import geopandas as gpd
import matplotlib.pyplot as plt
import math

import time

from ImageProcessor import ImageProcessor
from Car import Car
from multiprocessing import Process

#from pynput import keyboard


class ShapelyEnv:
    def __init__(
        self,
        width: int,
        height: int,
        img_processor: ImageProcessor,
        car: Car,
        show_game: bool = True,
        save_processed_track: bool = True,
        auto_config_car_start: bool = True,
        number_of_rays=5,
    ) -> None:
        self.width: int = width
        self.height: int = height
        self.background = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
        self.walls: list[list[LineString]] = [[], []]
        self.reward_lines: list[LineString] = []
        self.track: Polygon = None
        self.inverse_track: Polygon = None
        self.segmented_track_in_order: list[Polygon] = []
        self.reward_lines_in_order: list[LineString] = []
        self.car: Car = None
        self.rays: list[LineString] = []
        self.image_path: str = ""
        self.save_processed_track = save_processed_track
        self.auto_config_car_start = auto_config_car_start
        self.number_of_rays = number_of_rays
        self.set_track_environment(img_processor)
        self.set_car(car)

        if self.auto_config_car_start:
            self.config_car_start()

        self.update_rays()

        self.show_game: bool = show_game
        if show_game:
            # turn on interactive mode
            plt.ion()

            self.fig, self.ax = plt.subplots()
            self.fig.set_figheight(self.height / 70)
            self.fig.set_figwidth(self.width / 70)
            # add a non-updating background axis to plot the track
            self.background_axis = self.fig.add_axes(
                self.ax.get_position(), frameon=False
            )

            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.background_axis.set_xlim(0, self.width)
            self.background_axis.set_ylim(0, self.height)
            self.background_axis.xaxis.set_visible(False)
            self.background_axis.yaxis.set_visible(False)

    def set_track_environment(self, img_processor: ImageProcessor):
        """
        This method sets up the track environment based on the given image processor.

        Parameters:
        img_processor (ImageProcessor): An instance of the ImageProcessor class, which contains the image data.

        Returns:
        None. The method sets up the track environment by calling other methods within the ShapeEnv class.
        It will set the track, segmented tracks and inverse track.
        """
        self.image_path = img_processor.get_image_path()
        segments = img_processor.find_contour_segments()
        outer = img_processor.find_segment_points(segments[0])
        inner = img_processor.find_segment_points(segments[1])
        self.set_track_from_segment_points(outer, inner)
        self.set_walls(segments)
        self.set_reward_lines_from_walls()
        self.set_inverse_track()
        self.segment_track()

    def set_car(self, car: Car):
        """
        Set the car object for the game environment.

        Parameters:
        car (Car): The car object to be set for the game environment.

        Returns:
        None
        """
        self.car = car

    def set_track_from_segment_points(self, outer, inner):
        """
        Set the track polygon by subtracting the inner polygon from the outer polygon.

        Parameters:
        outer (list of tuples): A list of tuples representing the outer polygon's vertices.
        inner (list of tuples): A list of tuples representing the inner polygon's vertices.

        Returns:
        None. The track polygon is updated in the 'self.track' attribute.
        """
        outer_polygon = Polygon(outer)
        inner_poly = Polygon(inner)
        self.track = outer_polygon.difference(inner_poly)

    def set_walls(self, list_of_walls: list[list[list[float, float]]]):
        """
        This function sets the walls in the game environment.

        Parameters:
        list_of_walls (list[list[list[float, float]]]): A list of walls. Each wall is a line represented as a list of four coordinates.
        list of walls should be in the format of [out_walls, inner_walls], order is irrelevant

        Returns:
        None
        """
        for i, walls in enumerate(list_of_walls):
            for wall in walls:
                self.walls[i].append(
                    LineString([(wall[0], wall[1]), (wall[2], wall[3])])
                )

    def get_walls(self, flat=False):
        """
        Returns the walls of the game environment.

        Parameters:
        flat (bool, default=False): If True, returns all walls in a flat list. If False, returns walls in a list of lists.

        Returns:
        list: A list of walls. If flat is True, the list contains all walls. If flat is False, the list contains walls in two lists: self.walls[0] and self.walls[1].
        """
        if flat:
            return [wall for wall in self.walls[0] + self.walls[1]]
        return self.walls

    def set_reward_lines_from_walls(
        self,
    ) -> None:
        """
        This method calculates the reward lines from the walls.
        Reward lines are the lines connecting the centroids(midpoints) of two walls.
        These lines are used to segment the track into ordered blocks, to guide the car to navigate through the track .

        Parameters:
        None: Need to set walls before calling this method

        Returns:
        None: This method does not return any value. It updates the reward_lines attribute of the ShapeEnv instance.
        """
        wall1s = self.walls[0]
        wall2s = self.walls[1]
        assigned_wall2s = []
        for wall1 in wall1s:
            mid_point: Point = wall1.centroid
            min_distance = 99999999.9
            closest_wall = None

            for wall2 in wall2s:
                distance = wall2.centroid.distance(mid_point)
                if distance < min_distance and wall2 not in assigned_wall2s:
                    min_distance = distance
                    closest_wall = wall2
            reward_Line = LineString([mid_point, closest_wall.centroid])
            intersect = False
            for wall in self.get_walls(flat=True):
                if (
                    wall is not wall1
                    and wall is not closest_wall
                    and wall.intersects(reward_Line)
                ):
                    intersect = True
                    break
            if not intersect:
                self.reward_lines.append(reward_Line)
                assigned_wall2s.append(closest_wall)

    def set_inverse_track(self):
        self.inverse_track = self.background.difference(self.track)

    def segment_track(self):
        """
        This method segments the track into smaller blocks based on the reward lines.
        Each block is represented as a Shapely Polygon.

        Parameters:
        None: This method does not require any parameters. It uses the reward lines and the track
              stored in the ShapeEnv instance.

        Returns:
        None: This method does not return any value. It updates the segmented_track_in_order attribute
              of the ShapeEnv instance with the segmented track blocks.

        """
        buffer_size = 2.5  # 1e-5
        reward_lines_size = len(self.reward_lines)
        step = 2
        for i in range(0, reward_lines_size, step):
            # use two lines to segment the track into two polygons: a block and track.difference(block),
            # the block is the smaller polygon
            segmented_track = sorted(
                self.track.difference(self.reward_lines[i].buffer(buffer_size))
                .difference(
                    self.reward_lines[
                        i + step if i + step < reward_lines_size else 0
                    ].buffer(buffer_size)
                )
                .geoms,
                key=lambda x: x.area,
            )[0]

            self.segmented_track_in_order.append(segmented_track)
            self.reward_lines_in_order.append(self.reward_lines[i])

        # self.plot_shapely_objs(self.segmented_track_in_order)

    def get_number_of_segmented_tracks(self):
        return len(self.segmented_track_in_order)

    def get_unstopping_ray_endpoints_by_quadrant(
        self, car_x, car_y, up: bool, right: bool
    ) -> list[tuple[float, float]]:
        """
        This method calculates the endpoints of unstopping rays for a specific quadrant.

        Parameters:
        car_x (float): The x-coordinate of the car's position.
        car_y (float): The y-coordinate of the car's position.
        up (bool): A flag indicating whether the quadrant is in the upper half of the track.
        right (bool): A flag indicating whether the quadrant is in the right half of the track.

        Returns:
        list[tuple[float, float]]: A list of tuples representing the endpoints of the unstopping rays.
                                Each tuple contains the x and y coordinates of an endpoint.
        """
        if up and right:
            corner = [self.width, self.height]
        elif up and not right:
            corner = [0, self.height]
        elif not up and right:
            corner = [self.width, 0]
        else:
            corner = [0, 0]
        boundary_y = int(up) * self.height
        boundary_x = int(right) * self.width
        return [
            tuple(corner),
            tuple([car_x, boundary_y]),
            tuple([boundary_x, car_y]),
            tuple([boundary_x, (car_y + corner[1]) / 2]),
            tuple([(car_x + corner[0]) / 2, boundary_y]),
        ]

    def get_stopped_ray(
        self, unstopping_ray: LineString, car_point: Point
    ) -> LineString:
        """
        This method calculates the stopped ray from the unstopping ray.

        Parameters:
        unstopping_ray (LineString): The unstopping "ray" starting from the car to a boundary point.
        car_point (Point): The current position of the car.

        Returns:
        LineString: The stopped ray, which is a segment that is the part of the unstopping ray
                    that originates from the car and stopped at the first contact with the track's border,
                    simulating a sensor on the car to detect nearby obstacles.
        """

        intersections = self.track.intersection(unstopping_ray)

        if isinstance(intersections, LineString):
            return intersections

        # if there are multiple intersections, return the one that originates from the car
        for intersection in intersections.geoms:
            if intersection.intersects(car_point):
                return intersection

    def depreciated_get_unstopping_ray_endpoints_by_quadrants(
        self, car_point: Point, quadrant_indices: list[int]
    ) -> list[list[tuple[float, float]]]:
        """
        This method calculates the unstopping ray endpoints for specific quadrants, and returns them following the order of i to iv.

        Parameters:
        car_point (Point): The current position of the car.
        quadrant_indices (list[int]): A list of integers representing the quadrant indices.
                                    Each index corresponds to a specific quadrant.
                                    For example, [0, 1, 2, 3] represents all quadrants.

        Returns:
        list[list[tuple[float, float]]]: A list of lists of tuples.
                                        Each inner list represents the unstopping ray endpoints for a specific quadrant.
                                        Each tuple contains the x and y coordinates of an endpoint.
        """

        result = []
        if 0 in quadrant_indices:
            result.append(
                self.get_unstopping_ray_endpoints_by_quadrant(
                    car_point.x, car_point.y, True, True
                )
            )

        if 1 in quadrant_indices:
            result.append(
                self.get_unstopping_ray_endpoints_by_quadrant(
                    car_point.x, car_point.y, True, False
                )
            )

        if 2 in quadrant_indices:
            result.append(
                self.get_unstopping_ray_endpoints_by_quadrant(
                    car_point.x, car_point.y, False, False
                )
            )

        if 3 in quadrant_indices:
            result.append(
                self.get_unstopping_ray_endpoints_by_quadrant(
                    car_point.x, car_point.y, False, True
                )
            )
        return result

    def get_rays(self) -> list[LineString]:
        """
        Returns the rays that simulate the sensor on the car to detect nearby obstacles.

        Returns:
            list[LineString]
        """
        return self.rays

    def get_ray_lengths(self) -> list[float]:
        """
        Returns length of each ray in self.rays

        Returns: list[float]
            [ray.length for ray in self.rays]
        """
        if self.rays == []:
            return [0 for i in range(self.number_of_rays)]

        return [ray.length if ray is not None else 0 for ray in self.rays]

    def get_unstopping_ray_endpoints_on_boundary(self, angle):
        if angle == 90:
            x = self.car.get_car_x()
            y = self.height
        elif angle == 270:
            x = self.car.get_car_x()
            y = 0
        elif angle < 90 or angle > 270:
            x = self.width
            y = self.car.get_car_y() + (self.width - self.car.get_car_x()) * math.tan(
                math.radians(angle)
            )
        else:
            x = 0
            y = self.car.get_car_y() - self.car.get_car_x() * math.tan(
                math.radians(angle)
            )
        return x, y

    # only 9 nine rays no matter the situation
    def depreciated_update_rays(self):
        """
        Depreciated.
        Updates the rays based on the car's angle and position.

        The rays are calculated based on the car's position and angle, and are used to detect obstacles.
        The car's angle can be in any of the eight sections:
            [22.5, 67.5), [67.5, 112.5), [112.5, 157.5), [157.5, 202.5), [202.5, 247.5), [247.5, 292.5), [292.5, 337.5), [337.5, 360) & [0, 22.5).
        There are nines rays, each having to a specific angle, covering about 180 degrees of the car's front view.

        Parameters:
        None: uses the car's position and angle, to first get unstopping rays then compute the stopped rays.

        Returns:
        None: it will wipe out the old rays and add new ones based on the car's current position and angle.
        """
        angle = self.car.get_car_angle()
        car_point = self.car.get_shapely_point()
        self.rays = []
        unstopping_ray_endpoints = set()
        # [22.5, 67.5)
        if angle % 360 >= 90 / 4 * 1 and angle % 360 < 90 / 4 * 3:
            quadrant_i, quadrant_ii, quadrant_iv = (
                self.depreciated_get_unstopping_ray_endpoints_by_quadrants(
                    car_point, [0, 1, 3]
                )
            )
            left_boundary, left_mid = quadrant_ii[0], quadrant_ii[-1]
            right_boundary, right_mid = quadrant_iv[0], quadrant_iv[3]
            unstopping_ray_endpoints.update(
                set(quadrant_i + [left_boundary, left_mid, right_mid, right_boundary])
            )
        # [67.5, 112.5)
        elif angle % 360 >= 90 / 4 * 3 and angle % 360 < 90 / 4 * 5:
            quadrant_i, quadrant_ii = (
                self.depreciated_get_unstopping_ray_endpoints_by_quadrants(
                    car_point, [0, 1]
                )
            )
            unstopping_ray_endpoints.update(set(quadrant_i + quadrant_ii))
        # [112.5, 157.5)
        elif angle % 360 >= 90 / 4 * 5 and angle % 360 < 90 / 4 * 7:
            quadrant_i, quadrant_ii, quadrant_iii = (
                self.depreciated_get_unstopping_ray_endpoints_by_quadrants(
                    car_point, [0, 1, 2]
                )
            )
            left_boundary, left_mid = quadrant_iii[0], quadrant_iii[3]
            right_boundary, right_mid = quadrant_i[0], quadrant_i[-1]
            unstopping_ray_endpoints.update(
                set(quadrant_ii + [left_boundary, left_mid, right_mid, right_boundary])
            )
        # [157.5, 202.5)
        elif angle % 360 >= 90 / 4 * 7 and angle % 360 < 90 / 4 * 9:
            quadrant_ii, quadrant_iii = (
                self.depreciated_get_unstopping_ray_endpoints_by_quadrants(
                    car_point, [1, 2]
                )
            )
            unstopping_ray_endpoints.update(set(quadrant_ii + quadrant_iii))
        # [202.5, 247.5)
        elif angle % 360 >= 90 / 4 * 9 and angle % 360 < 90 / 4 * 11:
            quadrant_ii, quadrant_iii, quadrant_iv = (
                self.depreciated_get_unstopping_ray_endpoints_by_quadrants(
                    car_point, [1, 2, 3]
                )
            )
            left_boundary, left_mid = quadrant_ii[0], quadrant_ii[3]
            right_boundary, right_mid = quadrant_iv[0], quadrant_iv[-1]
            unstopping_ray_endpoints.update(
                quadrant_iii + [left_boundary, left_mid, right_mid, right_boundary]
            )

        # [247.5, 292.5)
        elif angle % 360 >= 90 / 4 * 11 and angle % 360 < 90 / 4 * 13:
            quadrant_iii, quadrant_iv = (
                self.depreciated_get_unstopping_ray_endpoints_by_quadrants(
                    car_point, [2, 3]
                )
            )
            unstopping_ray_endpoints.update(set(quadrant_iii + quadrant_iv))
        # [292.5, 337.5)
        elif angle % 360 >= 90 / 4 * 13 and angle % 360 < 90 / 4 * 15:
            quadrant_i, quadrant_iii, quadrant_iv = (
                self.depreciated_get_unstopping_ray_endpoints_by_quadrants(
                    car_point, [0, 2, 3]
                )
            )
            left_boundary, left_mid = quadrant_iii[0], quadrant_iii[-1]
            right_boundary, right_mid = quadrant_i[0], quadrant_i[3]
            unstopping_ray_endpoints.update(
                quadrant_iv + [left_boundary, left_mid, right_mid, right_boundary]
            )
        # [337.5, 360) & [0, 22.5)
        elif angle % 360 >= 90 / 4 * 15 or angle % 360 < 90 / 4 * 1:
            quadrant_i, quadrant_iv = (
                self.depreciated_get_unstopping_ray_endpoints_by_quadrants(
                    car_point, [0, 3]
                )
            )
            unstopping_ray_endpoints.update(set(quadrant_i + quadrant_iv))

        for x, y in unstopping_ray_endpoints:
            unstopping_ray = LineString([(car_point.x, car_point.y), (x, y)])
            self.rays.append(self.get_stopped_ray(unstopping_ray, car_point))

    def update_side_rays(self, car_point, angle):
        x, y = self.get_unstopping_ray_endpoints_on_boundary(angle)
        unstopping_ray = LineString([(car_point.x, car_point.y), (x, y)])
        self.rays.append(self.get_stopped_ray(unstopping_ray, car_point))

    def _join_threads(self, threads: list) -> None:
        for t in threads:
            t.join()

    def update_rays(self):
        if self.number_of_rays % 2 == 0:
            raise ValueError(
                "Total number of rays must be odd, because it has a center ray, and it is symmetrical."
            )
        self.rays = []

        car_angle = self.car.get_car_angle()
        car_point = self.car.get_shapely_point()

        number_of_rays_on_one_side = self.number_of_rays // 2

        for j in [-1, 1]:
            for i in range(1, number_of_rays_on_one_side + 1):
                angle = (car_angle + j * i * 90 / number_of_rays_on_one_side) % 360
                self.update_side_rays(car_point, angle)

        self.update_side_rays(car_point, car_angle)

    def multiProcess_update_rays(self, total_number_of_rays: int = 7):
        start_time = time.time()
        if total_number_of_rays % 2 == 0:
            raise ValueError(
                "Total number of rays must be odd, because it has a center ray, and it is symmetrical."
            )
        self.rays = []

        car_angle = self.car.get_car_angle()
        car_point = self.car.get_shapely_point()

        number_of_rays_on_one_side = total_number_of_rays // 2
        processes = []
        for j in [-1, 1]:
            for i in range(1, number_of_rays_on_one_side + 1):
                angle = (car_angle + j * i * 90 / number_of_rays_on_one_side) % 360
                t = Process(target=self.update_side_rays, args=(car_point, angle))
                t.start()
                processes.append(t)

        self._join_threads(processes)

        # the center ray is always at index -1
        front_x, front_y = self.get_unstopping_ray_endpoints_on_boundary(car_angle)
        unstopping_ray = LineString([(car_point.x, car_point.y), (front_x, front_y)])
        self.rays.append(self.get_stopped_ray(unstopping_ray, car_point))

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Execution time: {execution_time} seconds")

    def multiThread_update_rays(self, total_number_of_rays: int = 7):
        start_time = time.time()
        if total_number_of_rays % 2 == 0:
            raise ValueError(
                "Total number of rays must be odd, because it has a center ray, and it is symmetrical."
            )
        self.rays = []

        car_angle = self.car.get_car_angle()
        car_point = self.car.get_shapely_point()

        number_of_rays_on_one_side = total_number_of_rays // 2

        threads = []
        for j in [-1, 1]:
            for i in range(1, number_of_rays_on_one_side + 1):
                angle = (car_angle + j * i * 90 / number_of_rays_on_one_side) % 360
                t = Thread(target=self.update_side_rays, args=(car_point, angle))
                t.start()
                threads.append(t)

        self._join_threads(threads)

        # the center ray is always at index -1
        front_x, front_y = self.get_unstopping_ray_endpoints_on_boundary(car_angle)
        unstopping_ray = LineString([(car_point.x, car_point.y), (front_x, front_y)])
        self.rays.append(self.get_stopped_ray(unstopping_ray, car_point))

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Execution time: {execution_time} seconds")

    def plot_shapely_objs(
        self,
        shapely_objs: list,
        ax=None,
        pause_in_sec: float = 0.000001,
        color="red",
        save: bool = False,
    ):
        """
        Plots a list of Shapely objects on a Matplotlib axis.

        Parameters:
        shapely_objs (list): A list of Shapely objects to be plotted.
        ax (Matplotlib Axes, optional): The Matplotlib axis on which to plot the objects. If not provided, a new axis will be created.
        pause_in_sex (float, optional, default = 0.0001): The time in seconds the plot stays on the screen.
        color (str, optional, default = "red"): The color of the plotted objects. Default is "red".
        save (bool, optional, default: False): A flag indicating whether to save the plotted figure. Default is False.

        Returns:
        None
        """
        geos = gpd.GeoSeries(shapely_objs)
        df = gpd.GeoDataFrame({"geometry": geos})
        if ax:
            df.plot(ax=ax, color=color)

        else:
            df.plot(color=color)
        plt.pause(pause_in_sec)

        if save:
            current_time = time.time()
            plt.save(f"figure_{current_time}.png")

    def update_game_frame(self, shapely_objs: list, save=False):
        """
        Updates the game frame by clearing the current axis, setting the x and y limits as the game width and height,
        and plotting the updated car and rays.

        Parameters:
        shapely_objs (list): A list of Shapely objects to be plotted, in this case is the car and the rays.
        save (bool, optional): A flag indicating whether to save the plotted figure. Default is False.

        Returns:
        None
        """
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.plot_shapely_objs(shapely_objs, ax=self.ax, save=save)

    def draw_car(self):
        """
        Draws the car on the game environment.

        The car is represented as a red point on the game environment.
        The position of the car is determined by the car's current x and y coordinates.

        Parameters:
        None

        Returns:
        None

        Note:
        This method uses Matplotlib's scatter function to draw the car on the game environment.
        The scatter function plots a single point on the axis, which represents the car.
        The color of the point is set to 'red' to distinguish it from other elements in the game environment.
        The pause function is used to ensure that the drawing is visible for a short duration in interactive mode.
        """
        self.ax.scatter([self.car.get_car_x()], [self.car.get_car_y()], color="red")
        plt.pause(0.0001)

    def game_end(self) -> bool:
        """
        Checks if the game has ended.

        The game ends when the car collides with the track.
        This is determined by checking if the car's position is within the inverse track polygon.

        Parameters:
        None

        Returns:
        bool: True if the game has ended (car collided with the track), False otherwise.
        """
        return self.inverse_track.contains(self.car.get_shapely_point())

    def draw_background(self):
        """
        Draws the background of the game environment.

        The background is represented as a black polygon on the game environment.
        The polygon is created from the inverse of the track, which is obtained from the processed track image.

        Parameters:
        None

        Returns:
        None: It shows the inverse track as a black polygon, leaving the white space as the track

        Note:
        This method uses the GeoPandas library to create a GeoDataFrame from the inverse track polygon.
        The GeoDataFrame is then plotted on the background axis using the Matplotlib library.
        If the save_processed_track flag is set to True, the processed track image is saved as a PNG file.
        """

        geos = gpd.GeoSeries([self.inverse_track])
        df = gpd.GeoDataFrame({"geometry": geos})
        df.plot(ax=self.background_axis, color="black")
        if self.save_processed_track:
            plt.savefig(f"processed_{self.image_path.split('.')[0]}.png")

    def reset(self):
        """
        Resets the car's speed, rays, position and angle according to the auto configuration rules.

        This method sets the car's speed to 0 and calls the `config_car_start` method to set the car's position and angle.
        The car's position and angle are determined based on the processed track image.

        Parameters:
        None

        Returns:
        None: It resets the car's speed, rays, position and angle.

        Note:
        This method is called when the game needs to be reset, such as when the user presses the space bar (or when the game ends).
        """
        self.car.set_car_speed(0)
        self.config_car_start()
        self.update_rays()

    def keyboard_rule(self, key):
        """
        Handles keyboard inputs and performs corresponding actions on the car.

        Parameters:
        key (keyboard.Key): The key that was pressed on the keyboard.

        Returns:
        None: The function updates the self.car's position, angle and speed
        """
        if key == keyboard.Key.up:
            self.car.accelerate()

        if key == keyboard.Key.down:
            self.car.decelerate()

        if key == keyboard.Key.left:
            self.car.turn_left()

        if key == keyboard.Key.right:
            self.car.turn_right()

        if key == keyboard.Key.space:
            self.reset()

    def calculate_line_angle(self, p1: Point, p2: Point) -> float:
        """
        Calculates the angle [0,360) between two points in a 2D plane,
        the originating point is p1, the first argument

        Parameters:
        p1 (Point): The first point.
        p2 (Point): The second point.

        Returns:
        float: The angle between the two points in degrees [0, 360).
        """
        return (
            math.degrees(
                math.atan2(
                    (p2.y - p1.y),
                    (p2.x - p1.x),
                )
            )
            % 360
        )

    def calculate_car_start_angle(self):
        """
        Calculates the initial angle for the car at the start position.

        The start position is the centroid of the first track segment.
        The angle of the car is that of the line connecting the centroids of the first two reward lines.

        Parameters:
        None

        Returns:
        float: The initial angle for the car in degrees [0, 360).
        """
        return self.calculate_line_angle(
            self.reward_lines[0].centroid,
            self.reward_lines[1].centroid,
        )

    def config_car_start(self, angle_adjustment: float = 0.0) -> None:
        """
        Configures the initial position and angle of the car.

        The car's initial position is set to the centroid of the first track reward line.
        The car's initial angle is set to the angle of the line connecting the centroids of the first two reward lines.

        Parameters:
        angle_adjustment (float, optional): Increment car's initial angle by angle_adjustment. Default is 0.0.

        Returns:
        None: It will set car position and angle according to pre-define rules.

        Note:
        This method is called when the auto_config_car_start flag is set to True during the game initialization.
        It sets the car's position and angle based on the processed track image.
        """
        start_point = self.reward_lines[0].centroid
        self.car.set_car_coords(start_point.x, start_point.y)
        self.car.set_car_angle(self.calculate_car_start_angle() + angle_adjustment)

    def get_car_distance_to_segmented_track(self, index) -> float:
        """
        Calculates the distance between the car and a specific segment of the track.

        The distance is calculated as the difference between the car's shape and the shape of the track segment.
        The negative value of this method can be used as part of the reward in RL Models

        Parameters:
        index (int): The index of the track segment to calculate the distance to.

        Returns:
        float: The distance between the car and the specified track segment.
        """
        return self.car.get_shapely_point().distance(
            self.segmented_track_in_order[index]
        )

    def get_current_segmented_track_index(self) -> int:
        """
        Determines the index of the segmented track that the car is currently in.

        The function iterates over each segmented track in the order they were created.
        It checks if the car's current position is within the boundaries of the current segmented track.
        If the car's position is within the boundaries of a segmented track, the function returns the index of that segmented track.

        Parameters:
        None

        Returns:
        int: The index of the segmented track that the car is currently in.
        """
        for i, seg_track in enumerate(self.segmented_track_in_order):
            if seg_track.contains(self.car.get_shapely_point()):
                return i
        return None

    def start_game(self):
        """
        Starts the game loop.

        The game loop continuously updates the car's position, rays, and checks for game end conditions.
        If the game end condition is met (car collides with the track), the game ends.

        Parameters:
        None

        Returns:
        None
        """
        running = True

        listener = keyboard.Listener(
            on_press=self.keyboard_rule,
        )
        listener.start()
        if self.show_game:
            self.draw_background()
        while running:
            self.car.update_car_position()
            self.update_rays()

            if self.show_game:
                self.update_game_frame([self.car.get_shapely_point()] + self.rays)
            running = not self.game_end()

        listener.stop()


if __name__ == "__main__":
    width = 800
    height = 600

    # object creation
    img_processor = ImageProcessor("map1.png", resize=[width, height])
    car = Car(650, 100, 0, 90)
    game_env = ShapelyEnv(
        width,
        height,
        img_processor,
        car,
        show_game=True,
        save_processed_track=True,
        auto_config_car_start=True,
    )
    # game_env.set_track_environment(img_processor)
    # start game
    # game_env.plot_shapely_objs(game_env.segmented_track_in_order, color="black")
    # time.sleep(5)
    game_env.start_game()
    # game_env.update_rays()
#
