import torch
from Car import Car
from ImageProcessor import ImageProcessor
import numpy as np
from RecordEnv import RecordEnv
from ActionSpace import ActionSpace


class RLEnv(RecordEnv):
    def __init__(
        self,
        action_space: ActionSpace,
        img_processor: ImageProcessor,
        car: Car,
        width: int = 800,
        height: int = 600,
        show_game: bool = False,
        save_processed_track: bool = True,
        maximum_steps: int = 1500,
        record_queue=["reward"],
    ):
        """
        Initialize a new instance of RLEnv.

        Parameters:
        device (str): The device to run the model on. It can be either 'cuda:0' or 'cpu'.
        action_space (ActionSpace): The action space object that defines the available actions.
        hidden_sizes (list[int]): A list of integers representing the sizes of the output of the hidden layers.
        width (int, optional): The width of the game environment. Defaults to 800.
        height (int, optional): The height of the game environment. Defaults to 600.
        show_game (bool, optional): A flag indicating whether to show the game environment. Defaults to True.
        save_processed_track (bool, optional): A flag indicating whether to save the processed track. Defaults to True.


        Returns:
        None: It initializes the RLEnv instance.
        """
        super().__init__(
            width=width,
            height=height,
            img_processor=img_processor,
            car=car,
            show_game=show_game,
            save_processed_track=save_processed_track,
            auto_config_car_start=True,
            record_queue=record_queue,
        )
        self.action_space: ActionSpace = action_space
        self.AVG_DISTANCE_TO_NEXT_SEGMENT, self.STD_DISTANCE_TO_NEXT_SEGMENT = (
            self.get_segment_distance_avg_std()
        )
        self.AVG_RAY_LENGTH, self.STD_RAY_LENGTH = self.get_ray_length_avg_std()

        self.AVG_CAR_SPEED, self.STD_CAR_SPEED = (
            (self.car.max_backward_speed + self.car.max_forward_speed) / 2,
            1,
        )

        self.AVG_ANGLE_DIFFERENCE, self.STD_ANGLE_DIFFERENCE = 0, 180

        self.MAXIMUM_STEPS = maximum_steps
        # Because auto_config_car_start is hard coded to be True in super().__init__,
        # the car's initial position is always the centroid of the first track segment.
        self.current_segmented_track_index = 0
        # self.visited_segmented_track_indices = [0]
        self.state_size = len(self.get_state())
        # An ML model that represents the driver

        self.current_step = 0

        self.LIFE_REWARD = 0
        self.GOAL_REWARD = 1
        self.CRASH_REWARD = -1

    def get_ray_length_avg_std(self) -> tuple[float, float]:
        arr = np.array(self.get_ray_lengths())
        return np.mean(arr), np.std(arr)

    def get_segment_distance_avg_std(self) -> tuple[float, float]:
        distances = []
        for i in range(self.get_number_of_segmented_tracks()):
            distances.append(
                self.segmented_track_in_order[i].centroid.distance(
                    self.segmented_track_in_order[
                        (i + 1) % self.get_number_of_segmented_tracks()
                    ].centroid
                )
            )
        arr = np.array(distances)
        return np.mean(arr), np.std(arr)

    def normalize(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    def standardize(self, value, mean, std):
        if isinstance(value, list):
            return [self.standardize(x, mean, std) for x in value]
        return (value - mean) / std

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

    def get_target_angle(self) -> float:
        """
        Target angle is the angle of the line that connects the car and centroid of the next track segment,
        representing a sense of direction for the driver to modify the car's angle.
        Parameters:
        None

        Returns:
        float: returns the target angle [0, 360).
        """
        return self.calculate_line_angle(
            self.car.get_shapely_point(),
            self.reward_lines_in_order[
                (self.current_segmented_track_index + 1)
                % self.get_number_of_segmented_tracks()
            ].centroid,
        )

    def get_difference_car_angle_target_angle(self):
        """
        Calculate the difference between the car's current angle and the target angle.
        A value of 0 means the car is heading straight to the next track segment (regardless of walls in-between them).
        (-180, 0) means car need to turn left (counter-clockwise), increasing car angle
        (0, 180] means car need to turn right (clockwise), decreasing car angle

        Returns:
            float: The difference between the car's current angle and the target angle (-180, 180].
        """
        difference = self.car.get_car_angle() - self.get_target_angle()
        if difference > 180:
            return difference - 360
        if difference <= -180:
            return difference + 360
        return difference

    def get_state(self):
        """
        Calculates and returns the current state of the RL environment.

        The state is a list containing standardized values of various parameters:
        - Distance to the next track segment
        - Car speed
        - Difference between the car's current angle and the target angle

        - Ray lengths from the car to the track boundaries

        The state is standardized using the mean and standard deviation of each parameter.

        Parameters:
        None

        Returns:
        list: A list containing standardized values representing the current state of the RL environment.
        """

        angle_difference_standardized = self.standardize(
            self.get_difference_car_angle_target_angle(),
            self.AVG_ANGLE_DIFFERENCE,
            self.STD_ANGLE_DIFFERENCE,
        )
        """
        car_speed_standardized = self.standardize(
            self.car.get_car_speed(), self.AVG_CAR_SPEED, self.STD_CAR_SPEED
        )
        """
        target_segment_index = (self.current_segmented_track_index + 1) % len(
            self.segmented_track_in_order
        )

        distance_to_next_segment = self.get_car_distance_to_segmented_track(
            target_segment_index
        )
        distance_to_next_segment_standardized = self.standardize(
            distance_to_next_segment,
            self.AVG_DISTANCE_TO_NEXT_SEGMENT,
            self.STD_DISTANCE_TO_NEXT_SEGMENT,
        )
        ray_lengths_standardized = self.standardize(
            self.get_ray_lengths(), self.AVG_RAY_LENGTH, self.STD_RAY_LENGTH
        )
        # If the car arrived in the next target segment, but not segment 0 (final destination)

        return [
            distance_to_next_segment_standardized,
            # car_speed_standardized,
            angle_difference_standardized,
        ] + ray_lengths_standardized

    def get_state_size(self):
        return self.state_size

    def execute_car_logic(self, action: int):
        descriptive_action: str = self.action_space.descriptive_action_by_action(action)
        if descriptive_action == "accelerate":
            self.car.accelerate()
        elif descriptive_action == "decelerate":
            self.car.decelerate()
        elif descriptive_action == "steer_left":
            self.car.turn_left()
        elif descriptive_action == "steer_right":
            self.car.turn_right()

    def reached_next_segment(
        self, distance_to_next_segment_standardized: float
    ) -> bool:
        # if reached new segment
        if distance_to_next_segment_standardized <= self.standardize(
            0, self.AVG_DISTANCE_TO_NEXT_SEGMENT, self.STD_DISTANCE_TO_NEXT_SEGMENT
        ):
            self.current_segmented_track_index = (
                self.current_segmented_track_index + 1
            ) % len(self.segmented_track_in_order)
            return True
        return False

    def step(self, action: int):
        """execute the action in game env and return the new state, reward, terminated, (truncated, info)"""
        reward = 0

        self.execute_car_logic(action)
        self.action_record.set_current_value(action)
        self.action_record.add_current_Value_to_record()  # print
        self.car.update_car_position()
        self.update_rays()

        if self.show_game:
            self.update_game_frame([self.car.get_shapely_point()] + self.rays)
        self.current_step += 1
        new_state = self.get_state()

        running = not self.game_end()
        if not running:
            reward += self.CRASH_REWARD

        elif self.reached_next_segment(new_state[0]):
            reward += self.GOAL_REWARD

        return (
            new_state,
            reward,
            (not running) or self.current_step >= self.MAXIMUM_STEPS,
        )

    def reset(self):
        self.car.set_car_speed(0)
        self.config_car_start()
        self.update_rays()
        self.action_record.save_record_to_txt()
        self.current_step = 0
        self.current_segmented_track_index = 0

    def start_game_RL(self):
        """
        Starts the game in RL environment.

        Parameters:
        None

        Returns:
        None
        """
        if self.show_game:
            self.draw_background()
        if self.auto_config_car_start:
            self.config_car_start()


if __name__ == "__main__":
    width = 800
    height = 600
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()

    # object creation
    img_processor = ImageProcessor("map1.png", resize=[width, height])
    car = Car(650, 100, 0, 90)
    game_env = RLEnv(
        ActionSpace(["hold", "accelerate", "decelerate", "steer_left", "steer_right"]),
        img_processor,
        car,
    )
