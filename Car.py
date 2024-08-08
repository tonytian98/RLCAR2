import math
from shapely.geometry import Point


class Car:
    def __init__(
        self,
        start_x: float,
        start_y: float,
        start_speed: float,
        start_angle: float,
        max_forward_speed: float = 4,
        max_backward_speed: float = 0,
        acceleration: float = 0.1,
        turn_angle: float = 5,
    ):
        self.car_x = start_x
        self.car_y = start_y
        self.car_speed = start_speed
        self.car_angle = start_angle
        self.max_forward_speed = max_forward_speed
        self.max_backward_speed = max_backward_speed
        self.acceleration = acceleration
        self.turn_angle = turn_angle

    def get_car_x(self) -> float:
        return self.car_x

    def set_car_x(self, car_x: float) -> None:
        self.car_x = car_x

    def get_car_y(self) -> float:
        return self.car_y

    def set_car_y(self, car_y: float) -> None:
        self.car_y = car_y

    def set_car_coords(self, car_x: float, car_y: float) -> None:
        self.car_x = car_x
        self.car_y = car_y

    def get_car_coords(self) -> tuple[float, float]:
        return self.car_x, self.car_y

    def get_car_speed(self) -> float:
        return self.car_speed

    def set_car_speed(self, car_speed: float) -> None:
        self.car_speed = car_speed

    def get_car_angle(self) -> float:
        return self.car_angle % 360

    def set_car_angle(self, car_angle: float) -> None:
        self.car_angle = car_angle

    def get_max_forward_speed(self) -> float:
        return self.max_forward_speed

    def set_max_forward_speed(self, max_forward_speed: float) -> None:
        self.max_forward_speed = max_forward_speed

    def get_max_backward_speed(self) -> float:
        return self.max_backward_speed

    def set_max_backward_speed(self, max_backward_speed: float) -> None:
        self.max_backward_speed = max_backward_speed

    def get_acceleration(self) -> float:
        return self.acceleration

    def set_acceleration(self, acceleration: float) -> None:
        self.acceleration = acceleration

    def get_turn_angle(self) -> float:
        return self.turn_angle

    def set_turn_angle(self, turn_angle: float) -> None:
        self.turn_angle = turn_angle

    def turn_right(self) -> None:
        """
        Turns the car to the right by decrementing its angle by the turn angle.

        Parameters:
        - None

        Returns:
        - None: This method does not return any value.

        The method updates the car's angle by subtracting the turn angle.
        This allows the car to change its direction to the right.
        """
        self.car_angle -= self.turn_angle

    def turn_left(self) -> None:
        """
        Turns the car to the left by incrementing its angle by the turn angle.

        Parameters:
        - None

        Returns:
        - None: This method does not return any value.

        The method updates the car's angle by adding the turn angle.
        This allows the car to change its direction to the left.
        """
        self.car_angle += self.turn_angle

    def accelerate(self) -> None:
        if self.car_speed < self.max_forward_speed:
            self.car_speed += self.acceleration

    def decelerate(self) -> None:
        if self.car_speed > -self.max_backward_speed:
            self.car_speed = max(
                -self.max_backward_speed, self.car_speed - self.acceleration
            )

    def update_car_position(self) -> tuple[float, float]:
        """
        Updates the car's position based on its current speed and angle.

        Returns:
        - tuple[float, float]: A tuple containing the updated car's x and y coordinates.

        The method calculates the new x and y coordinates of the car based on its current speed and angle.
        The car's x coordinate is updated by adding the product of its speed and the cosine of its angle.
        The car's y coordinate is updated by subtracting the product of its speed and the sine of its angle.
        The updated coordinates are then returned as a tuple.
        """
        self.car_x += self.car_speed * math.cos(math.radians(self.car_angle))
        self.car_y += self.car_speed * math.sin(math.radians(self.car_angle))
        return self.car_x, self.car_y

    def get_shapely_point(self) -> Point:
        return Point([self.car_x, self.car_y])

    def get_angle_in_radians(self) -> float:
        return math.radians(self.car_angle)
