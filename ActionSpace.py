import torch


class ActionSpace:
    def __init__(self, descriptive_actions: list[str]):
        """
        Initialize the ActionSpace object with descriptive actions.

        Parameters:
        descriptive_actions (list[str]): A list of strings representing the descriptive actions.

        Returns:
        None: It will generate the action space in the form of [0, 1, ..., len(descriptive actions)].
            Action space consists of integers(actions) from 0 to (len(descriptive actions) - 1).
        """
        self.descriptive_actions: list[str] = descriptive_actions
        self.actions = [i for i in range(len(descriptive_actions))]
        self.n = len(self.actions)

    def descriptive_action_by_action(self, i):
        """
        Returns the descriptive action corresponding to the given index.

        Parameters:
        i (int): The index of the action.

        Returns:
        str: The descriptive action corresponding to the given index.
        """
        return self.descriptive_actions[i]

    def action_by_descriptive_action(self, descriptive_action):
        """
        Returns the index of the given descriptive action in the action space.

        Parameters:
        descriptive_action (str): The descriptive action for which the index is to be found.

        Returns:
        int: The index of the given descriptive action in the action space. If the descriptive action is not found, it returns None.
        """
        return self.descriptive_actions.index(descriptive_action)

    def sample(self):
        """
        This function is used to sample an action from the action space, which is a list [0, 1, ...].

        Parameters:
        None

        Returns:
        int: A random action from the action space.
        """
        return torch.randint(0, self.n, (1,)).item()

    def __len__(self):
        return self.n

    def __str__(self):
        print([f"{i}: {self.descriptive_action_by_action(i)}" for i in range(self.n)])
