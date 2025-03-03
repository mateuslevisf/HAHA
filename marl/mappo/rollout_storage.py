import torch as th
import numpy as np

class RolloutStorage:
    """
    Storage for rollouts collected during MAPPO training.
    Stores and processes experiences for both agents.
    """

    def __init__(self, num_agents, buffer_size, observation_space, action_space,
                 device=th.device("cpu"), gae_lambda=0.95, gamma=0.99):
        """
        Initialize the rollout storage.

        Args:
            num_agents: Number of agents (2 for this implementation)
            buffer_size: Maximum buffer size (steps per update)
            observation_space: Observation space of each agent
            action_space: Action space of each agent
            device: Device to store tensors on
            gae_lambda: GAE lambda parameter
            gamma: Discount factor
        """
        self.num_agents = num_agents
        self.buffer_size = buffer_size
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        # Initialize buffers for each agent
        self.observations = [[] for _ in range(num_agents)]
        self.action_masks = [[] for _ in range(num_agents)]
        self.actions = [[] for _ in range(num_agents)]
        self.rewards = [[] for _ in range(num_agents)]
        self.returns = [[] for _ in range(num_agents)]
        self.values = [[] for _ in range(num_agents)]
        self.advantages = [[] for _ in range(num_agents)]
        self.log_probs = [[] for _ in range(num_agents)]
        self.dones = [[] for _ in range(num_agents)]

        # For centralized critic
        self.centralized_observations = []

        # Current position in buffer
        self.pos = 0
        self.full = False

    def add(self, obss, centralized_obs, action_masks, actions, log_probs, values, rewards, dones):
        """
        Add a transition to the buffer.

        Args:
            obs: List of agent observations
            centralized_obs: Centralized observations for critic
            action_masks: List of action masks for each agent
            actions: List of actions taken by each agent
            log_probs: List of log probabilities of actions
            values: List of value estimates
            rewards: List of rewards received
            dones: List of done flags
        """
        for i in range(self.num_agents):
            if len(self.observations[i]) < self.buffer_size:
                self.observations[i].append(obss[i])
                self.action_masks[i].append(action_masks[i])
                self.actions[i].append(actions[i])
                self.log_probs[i].append(log_probs[i])
                self.values[i].append(values[i])
                self.rewards[i].append(rewards[i])
                self.dones[i].append(dones[i])
            else:
                # Replace oldest entry (circular buffer)
                idx = self.pos % self.buffer_size
                self.observations[i][idx] = obss[i]
                self.action_masks[i][idx] = action_masks[i]
                self.actions[i][idx] = actions[i]
                self.log_probs[i][idx] = log_probs[i]
                self.values[i][idx] = values[i]
                self.rewards[i][idx] = rewards[i]
                self.dones[i][idx] = dones[i]

        # Store centralized observation
        if len(self.centralized_observations) < self.buffer_size:
            self.centralized_observations.append(centralized_obs)
        else:
            idx = self.pos % self.buffer_size
            self.centralized_observations[idx] = centralized_obs

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_values):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).

        Args:
            last_values: Value estimates for the next state after the buffer ends
        """
        for agent_idx in range(self.num_agents):
            # Get all values for the agent
            values = self.values[agent_idx]
            rewards = self.rewards[agent_idx]
            dones = self.dones[agent_idx]

            # Initialize returns and advantages arrays
            returns = []
            advantages = []
            last_value = last_values[agent_idx]

            gae = 0
            for step in reversed(range(len(rewards))):
                # If this is the last step, use the provided last value
                # Otherwise, use the value from our buffer
                next_value = last_value if step == len(rewards) - 1 else values[step + 1]
                next_done = 1.0 if step == len(rewards) - 1 else dones[step + 1]

                # Compute delta and GAE
                delta = rewards[step] + self.gamma * next_value * (1.0 - next_done) - values[step]
                gae = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * gae

                # Insert at the beginning of the list
                returns.insert(0, gae + values[step])
                advantages.insert(0, gae)

            self.returns[agent_idx] = returns
            self.advantages[agent_idx] = advantages

    def get_data(self):
        """
        Get all data from the buffer for training.

        Returns:
            dict: Dictionary containing all data needed for training
        """
        # Convert all lists to PyTorch tensors
        data = {}
        for agent_idx in range(self.num_agents):
            # Process observations correctly - convert visual_obs to tensors
            agent_obs = []
            for obs in self.observations[agent_idx]:
                if 'visual_obs' in obs:
                    # Make a copy and convert to tensor
                    visual_obs = th.tensor(np.ascontiguousarray(obs['visual_obs']), device=self.device).float()
                    # Flatten for passing to network
                    flattened_obs = visual_obs.reshape(1, -1).squeeze(0)
                    agent_obs.append(flattened_obs)

            data[f'agent_{agent_idx}_obs'] = th.stack(agent_obs) if agent_obs else []

            # Process action masks correctly
            agent_masks = []
            for mask in self.action_masks[agent_idx]:
                mask_tensor = th.tensor(mask, device=self.device).bool()
                agent_masks.append(mask_tensor)

            data[f'agent_{agent_idx}_action_masks'] = agent_masks if agent_masks else []

            # Process other data
            data[f'agent_{agent_idx}_actions'] = th.tensor(self.actions[agent_idx], device=self.device)
            data[f'agent_{agent_idx}_log_probs'] = th.tensor(self.log_probs[agent_idx], device=self.device)
            values_array = np.array(self.values[agent_idx])
            data[f'agent_{agent_idx}_values'] = th.tensor(values_array, device=self.device)
            data[f'agent_{agent_idx}_returns'] = th.tensor(np.array(self.returns[agent_idx]), device=self.device)
            data[f'agent_{agent_idx}_advantages'] = th.tensor(self.advantages[agent_idx], device=self.device)

        # Process centralized observations for critic
        cent_obs = []
        for obs in self.centralized_observations:
            if isinstance(obs, th.Tensor):
                cent_obs.append(obs.clone().detach().to(device=self.device).float())
            else:
                cent_obs.append(th.tensor(obs, device=self.device).float())

        # Always return a tensor (with appropriate dimensions if empty)
        if cent_obs:
            data['centralized_obs'] = th.cat(cent_obs, dim=0)
        else:
            # Create an empty tensor with proper shape
            # Adjust the shape based on your observation dimensions
            print("should be using obs dim here TODO FIX")
            data['centralized_obs'] = th.zeros((0, 0), device=self.device)

        return data

    def clear(self):
        """Clear the buffer."""
        self.observations = [[] for _ in range(self.num_agents)]
        self.action_masks = [[] for _ in range(self.num_agents)]
        self.actions = [[] for _ in range(self.num_agents)]
        self.rewards = [[] for _ in range(self.num_agents)]
        self.returns = [[] for _ in range(self.num_agents)]
        self.values = [[] for _ in range(self.num_agents)]
        self.advantages = [[] for _ in range(self.num_agents)]
        self.log_probs = [[] for _ in range(self.num_agents)]
        self.dones = [[] for _ in range(self.num_agents)]
        self.centralized_observations = []
        self.pos = 0
        self.full = False
