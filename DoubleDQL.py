from Car import Car
from ImageProcessor import ImageProcessor
import numpy as np
import copy
from collections import deque
import random
from RLEnv import RLEnv, ActionSpace
from DQN import DQN
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data.dataset import IterableDataset
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import time


class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize a new instance of ReplayBuffer.

        Parameters:
        capacity (int): The maximum number of experiences that can be stored in the buffer.
            When the buffer is full, older experiences will be discarded to make space for new ones.

        Returns:
        None: It initializes the ReplayBuffer instance.
            Use .sample(size) method as the getter method to get items inside the buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_size=300):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        for experience in self.buffer.sample(self.sample_size):
            yield experience


class DeepQLearning(LightningModule):
    # Initialize.
    def __init__(
        self,
        env: RLEnv,  # game environment with an epsilon greedy policy
        capacity: int = 5000,  # capacity of the replay buffer
        batch_size: int = 256,  # batch size for training
        hidden_sizes: list[int] = [64],  # hidden sizes for the neural network
        lr: float = 1e-3,  # learning rate for optimizer
        loss_fn=F.smooth_l1_loss,  # loss function
        optimizer=AdamW,  # optimizer that updates the model parameters
        gamma: float = 0.99,  # discount factor for accumulating rewards
        eps_start: float = 1.0,  # starting epsilon for epsilon greedy policy
        eps_end: float = 0.1,  # ending epsilon for epsilon greedy policy
        eps_last_episode: int = 150,  # number of episodes used to decay epsilon to eps_end
        samples_per_epoch: int = 2500,  # number of samples needed per training episode
        sync_rate: int = 10,  # number of epochs before we update the policy network using the target network
    ):
        """
        Initialize a new instance of DeepQLearning.

        Parameters:
        env (RLEnv): The game environment with an epsilon greedy policy.
        capacity (int, optional): The capacity of the replay buffer. Defaults to 10,000.
        batch_size (int, optional): The batch size for training. Defaults to 256.
        hidden_sizes (list[int], optional): The hidden sizes for the neural network. Defaults to [128].
        lr (float, optional): The learning rate for optimizer. Defaults to 1e-3.
        loss_fn (function, optional): The loss function. Defaults to F.smooth_l1_loss.
        optimizer (function, optional): The optimizer that updates the model parameters. Defaults to AdamW.
        gamma (float, optional): The discount factor for accumulating rewards. Defaults to 0.99.
        eps_start (float, optional): The starting epsilon for epsilon greedy policy. Defaults to 1.0.
        eps_end (float, optional): The ending epsilon for epsilon greedy policy. Defaults to 0.1.
        eps_last_episode (int, optional): The number of episodes used to decay epsilon to eps_end. Defaults to 150.
        samples_per_epoch (int, optional): The number of samples needed per training episode. Defaults to 5_000.
        sync_rate (int, optional): The number of epochs before we update the policy network using the target network. Defaults to 10.

        Returns:
        None: It initializes the DeepQLearning instance.
        """
        super().__init__()
        self.env = env
        self.state_size = env.get_state_size()
        # policy network
        self.q_net = DQN(self.state_size, hidden_sizes, env.action_space.n)
        # target network
        self.target_q_net = copy.deepcopy(self.q_net)

        self.buffer = ReplayBuffer(capacity)

        self.save_hyperparameters()

        while len(self.buffer) < samples_per_epoch:
            self.play_episode(epsilon=eps_start)

        self.training_step_outputs = []

    def epsilon_greedy(self, device, epsilon=0.0) -> int:
        """
        Returns an action based on the current state of the environment.

        Args:
            epsilon (float, optional): The probability of choosing a random action. Defaults to 0.0.

        Returns:
            int: The chosen action.

        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.env.get_state()]).to(device)
            q_values = self.q_net(state)
            # action is the index of the max q value
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())
        return action

    @torch.no_grad()
    def play_episode(self, epsilon: float = 0.0) -> float:
        self.env.reset()
        state = self.env.get_state()
        game_over = False
        total_return = 0
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        while not game_over:
            action = self.epsilon_greedy(device, epsilon)
            next_state, reward, game_over = self.env.step(action)
            experience = (state, action, reward, game_over, next_state)
            self.buffer.append(experience)
            state = next_state
            total_return += reward
        self.env.add_to_records("reward", total_return)

        return total_return

    def forward(self, x):
        return self.q_net(x)

    def configure_optimizers(self):
        return [self.hparams.optimizer(self.q_net.parameters(), lr=self.hparams.lr)]

    def train_dataloader(self):
        """
        This function is used to create a DataLoader object for training the model.

        The Dataset object is created from an RLDataset object, which is initialized with the replay buffer and a sample size.
        It yields from a list of randomly sampled experiences of size self.samples_per_epoch.

        The DataLoader object sends training data from Dataset in the size of self.batch_size to the model,
            until the total number of training data sent reaches self.samples_per_epoch

        Parameters:
        None

        Returns:
        DataLoader: A DataLoader object that yields batches of randomly sampled experiences from the replay buffer.
        """
        # yield list of randomly sampled experiences of length self.hparams.samples_per_epoch

        dataset = RLDataset(self.buffer, sample_size=self.hparams.samples_per_epoch)
        max_num_worker_suggest = os.cpu_count()
        if max_num_worker_suggest:
            num_workers = max_num_worker_suggest
        else:
            num_workers = 10
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, num_workers=num_workers
        )

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        This method calculates the Q-value for the current state-action pairs,
        computes the expected Q-value for the next state-action pairs, and then
        calculates the loss between the current Q-value and the expected Q-value.
        The loss is then logged for monitoring purposes.

        Parameters:
        batch (tuple): A tuple containing the batch of training data.
            The tuple contains the following elements:
            - states (torch.Tensor): A tensor representing the states.
            - actions (torch.Tensor): A tensor representing the actions.
            - rewards (torch.Tensor): A tensor representing the rewards.
            - game_overs (torch.Tensor): A tensor representing the game over status.
            - next_states (torch.Tensor): A tensor representing the next states.

        batch_idx (int): The index of the batch within the epoch.

        Returns:
        torch.Tensor: The loss value for the current batch.
        """
        states, actions, rewards, game_overs, next_states = batch
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        game_overs = game_overs.unsqueeze(1)
        # Q-value (Action-Value): Represents the value of taking a specific action in a specific state.
        # Q-value (Action-Value): Represents the value of taking a specific action in a specific state.

        # It gets the value of the 'actions' that was taken in 'states' [ [q(ai,si)], [q(aj,sj)], ...  ]
        state_action_value = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            _, next_actions = self.q_net(next_states).max(dim=1, keepdim=True)
            next_action_values = self.target_q_net(next_states).gather(1, next_actions)
            next_action_values[game_overs] = (
                0.0  # set the value of terminal states to 0
            )
        expected_state_action_value = rewards + self.hparams.gamma * next_action_values
        loss = self.hparams.loss_fn(state_action_value, expected_state_action_value)
        self.log("episode/Q-Error", loss)
        self.training_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        """
        This function is called at the end of each training epoch.
        It updates the epsilon value for the epsilon-greedy policy,
        plays an episode with the updated epsilon value,
        logs the episode return, and synchronizes the target network with the policy network at sync_rate.

        Parameters:
        training_step_outputs: a dictionary with the values returned by the training_step method

        Returns:
        None
        """
        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.current_epoch / self.hparams.eps_last_episode,
        )
        episode_return = self.play_episode(epsilon=epsilon)
        self.log("episode/Return", episode_return)
        if self.current_epoch % self.hparams.sync_rate == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average)
        self.training_step_outputs.clear()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    # object creation
    width = 800
    height = 600

    logger = CSVLogger(save_dir="logs/", name="my_model")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()
    print(device, "number of GPUs", num_gpus)
    print("Number of CPUs", os.cpu_count())
    img_processor = ImageProcessor("map1.png", resize=[width, height])
    car = Car(650, 100, 5, 90, max_forward_speed=5)
    game_env = RLEnv(
        ActionSpace(["hold", "steer_left", "steer_right"]),
        img_processor,
        car,
        show_game=False,
        save_processed_track=True,
    )
    algo = DeepQLearning(game_env)

    trainer = Trainer(
        max_epochs=1000,
        callbacks=EarlyStopping(monitor="episode/Return", mode="max", patience=400),
    )

    start_time = time.time()
    trainer.fit(algo)
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")
