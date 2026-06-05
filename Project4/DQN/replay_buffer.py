import random
import collections
import torch
import numpy as np

class ReplayBufferDQN:
    def __init__(self, buffer_size:int, seed:int=42):
        self.buffer_size = buffer_size
        self.seed = seed
        # deque gives O(1) append and popleft vs O(n) list.pop(0)
        self.buffer = collections.deque(maxlen=buffer_size)
        random.seed(self.seed)

    def add(self, state:np.ndarray, action:int, reward:float, next_state:np.ndarray
            , done:bool):
        """
        Add a new experience to the buffer

        Args:
            state (np.ndarray): the current state of shape [n_c,h,w]
            action (int): the action taken
            reward (float): the reward received
            next_state (np.ndarray): the next state of shape [n_c,h,w]
            done (bool): whether the episode is done
        """
        # Store states as uint8 (0-255) to reduce memory 8x vs float64.
        # float32 states (0.0-1.0) -> uint8 (0-255), converted back in sample().
        state_u8 = (state * 255).astype(np.uint8)
        next_state_u8 = (next_state * 255).astype(np.uint8)
        self.buffer.append((state_u8, action, reward, next_state_u8, done))

    def sample(self, batch_size:int, device='cpu'):
        """
        Randomly sample a batch of experiences from the replay buffer.

        Args:
            batch_size (int): the number of samples to take

        Returns:
            states (torch.Tensor): Tensor of shape (batch_size, n_channels, height, width), dtype torch.float32.
            actions (torch.Tensor): Tensor of shape (batch_size,), dtype torch.int64 (converted via `.long()`).
            rewards (torch.Tensor): Tensor of shape (batch_size,), dtype torch.float32.
            next_states (torch.Tensor): Tensor of shape (batch_size, n_channels, height, width), dtype torch.float32.
            dones (torch.Tensor): Tensor of shape (batch_size,), dtype torch.bool.

        Notes:
            1. Use `random.sample` for uniform sampling without replacement.
            2. Convert NumPy arrays to torch tensors with the correct dtype before moving to `device`.
            3. Use `torch.stack` to combine individual tensors into a batch dimension.
            4. Keep the output shapes and dtypes consistent.
        """

        # ========== YOUR CODE HERE ==========
        # TODO:
        # 1. sample random indices
        # 2. collect experiences using the sampled indices
        # 3. stack and move batches to the specified device, making sure to convert to the correct dtype
        # ====================================
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for s, a, r, s_, d in batch:
            # Convert uint8 back to float32 in [0, 1]
            states.append(torch.from_numpy(s.astype(np.float32) / 255.0))
            actions.append(torch.tensor(a, dtype=torch.int64))
            rewards.append(torch.tensor(r, dtype=torch.float32))
            next_states.append(torch.from_numpy(s_.astype(np.float32) / 255.0))
            dones.append(torch.tensor(d, dtype=torch.bool))

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).long().to(device)
        rewards = torch.stack(rewards).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.stack(dones).bool().to(device)
        # ========== YOUR CODE ENDS ==========

        return states, actions, rewards, next_states, dones


    def __len__(self):
        return len(self.buffer)
