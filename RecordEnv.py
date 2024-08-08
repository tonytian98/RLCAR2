from ShapelyEnv import ShapelyEnv
from time import sleep
from Record import Record


class RecordEnv(ShapelyEnv):
    def __init__(
        self,
        img_processor,
        car,
        show_game,
        save_processed_track,
        record_queue: list = [],
        width=800,
        height=600,
        auto_config_car_start=True,
    ):
        """
        Initialize a RecordEnv object.

        Parameters:
        *args: Variable length argument list. Passed to ShapelyEnv.
        **kwargs: Arbitrary keyword arguments. Passed to ShapelyEnv.

        Additional Attributes:
        action_record (Record): An instance of the Record class, used to record actions.
        """
        super().__init__(
            width=width,
            height=height,
            img_processor=img_processor,
            car=car,
            show_game=show_game,
            save_processed_track=save_processed_track,
            auto_config_car_start=auto_config_car_start,
        )

        self.action_record = Record("actions")
        self.records = {}
        for item in record_queue:
            self.records[item] = []
        print("records are stored in", self.action_record.get_dir_name())

    def add_to_records(self, name, record):
        if name in self.records:
            self.records[name].append(record)
        else:
            raise ValueError(
                "Record name does not exist. Please define it in class constructor"
            )

    def get_records(self, name):
        if name in self.records:
            return self.records[name]
        else:
            raise ValueError(
                "Record name does not exist.  Please define it in class constructor"
            )

    def clear_records(self, name):
        if name in self.records:
            self.records[name] = []

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
            self.action_record.set_current_value("UP")

        if key == keyboard.Key.down:
            self.car.decelerate()
            self.action_record.set_current_value("DOWN")

        if key == keyboard.Key.left:
            self.car.turn_left()
            self.action_record.set_current_value("LEFT")

        if key == keyboard.Key.right:
            self.car.turn_right()
            self.action_record.set_current_value("RIGHT")

        if key == keyboard.Key.space:
            self.reset()
            self.action_record.set_current_value("SPACE")

    def start_game_with_record(self):
        """
        Starts the game loop with record.

        Note: Record does not support continuous keyboard input (holding a key down).
        It will register continuous input of the same key to be only one press,
        because it is designed for RL models, whose action is always discrete.

        The game loop continuously updates the car's position, rays, and checks for game end conditions.
        If the game end condition is met (car collides with the track), the game ends.
        """
        running = True

        listener = keyboard.Listener(
            on_press=self.keyboard_rule,
        )
        listener.start()
        if self.show_game:
            self.draw_background()
        while running:
            if self.action_record.get_current_value() is None:
                self.action_record.set_current_value("NO_ACTION")
            self.action_record.add_current_Value_to_record()  # print
            self.car.update_car_position()
            self.update_rays()
            if self.show_game:
                self.update_game_frame([self.car.get_shapely_point()] + self.rays)
            running = not self.game_end()
        listener.stop()

        self.action_record.save_record_to_txt()

    def replay(self, offset: int = 0):
        actions = self.action_record.get_replay_records()
        if len(actions) == 0:
            raise ValueError("Replay record is not set")

        if self.show_game:
            self.draw_background()
        for i, action in enumerate(actions):
            if "1" in action:
                self.car.accelerate()
            elif "2" in action:
                self.car.decelerate()
            elif "3" in action:
                self.car.turn_left()
            elif "4" in action:
                self.car.turn_right()
            elif "SPACE" in action:
                self.reset()

            self.car.update_car_position()
            self.update_rays()
            if self.show_game:
                self.update_game_frame([self.car.get_shapely_point()] + self.rays)
            # Normal exit condition
            if self.game_end() and i == len(actions) - 1:
                break
            # Game ended before all actions are replayed
            if game_env.game_end() and i != len(actions) - 1:
                raise ValueError("Game ended early.")
            # Game did not end after all actions are replayed
            if not game_env.game_end() and i == len(actions) - 1:
                raise ValueError("Game did not end after all actions. It should.")


if __name__ == "__main__":
    from Car import Car
    from ImageProcessor import ImageProcessor

    width = 800
    height = 600

    # object creation
    img_processor = ImageProcessor("map1.png", resize=[width, height])
    car = Car(650, 100, 0, 90)
    game_env = RecordEnv(img_processor, car, show_game=True, save_processed_track=False)

    # start game
    # game_env.start_game_with_record()
    # reset game
    # game_env.reset()
    sleep(2)
    # replay game
    game_env.action_record.set_replay_record_from_file("record6.txt")
    game_env.replay()
