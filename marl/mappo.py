"""
MAPPO Trainer for HAHA Manager Policy Training

This module implements Multi-Agent Proximal Policy Optimization (MAPPO) for training
HAHA managers with a centralized critic and decentralized actors.
"""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv

from oai_agents.agents.agent_utils import load_agent
from oai_agents.common.subtasks import Subtasks


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

    def add(self, obs, centralized_obs, action_masks, actions, log_probs, values, rewards, dones):
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
                self.observations[i].append(obs[i])
                self.action_masks[i].append(action_masks[i])
                self.actions[i].append(actions[i])
                self.log_probs[i].append(log_probs[i])
                self.values[i].append(values[i])
                self.rewards[i].append(rewards[i])
                self.dones[i].append(dones[i])
            else:
                # Replace oldest entry (circular buffer)
                idx = self.pos % self.buffer_size
                self.observations[i][idx] = obs[i]
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
            data[f'agent_{agent_idx}_obs'] = self.observations[agent_idx]
            data[f'agent_{agent_idx}_action_masks'] = self.action_masks[agent_idx]
            data[f'agent_{agent_idx}_actions'] = th.tensor(self.actions[agent_idx], device=self.device)
            data[f'agent_{agent_idx}_log_probs'] = th.tensor(self.log_probs[agent_idx], device=self.device)
            data[f'agent_{agent_idx}_values'] = th.tensor(self.values[agent_idx], device=self.device)
            data[f'agent_{agent_idx}_returns'] = th.tensor(self.returns[agent_idx], device=self.device)
            data[f'agent_{agent_idx}_advantages'] = th.tensor(self.advantages[agent_idx], device=self.device)

        data['centralized_obs'] = self.centralized_observations
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


class MLPBase(nn.Module):
    """
    MLP Base network for actors and critic
    """
    def __init__(self, input_dim, hidden_size=64, use_orthogonal=True):
        super(MLPBase, self).__init__()
        self.hidden_size = hidden_size

        # Initialization function
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
        gain = 2 ** 0.5

        def init_(m):
            if isinstance(m, nn.Linear):
                init_method(m.weight, gain)
                nn.init.constant_(m.bias, 0)
            return m

        self.net = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class ActorNetwork(nn.Module):
    """
    Actor network for MAPPO that outputs subtask selection probabilities
    """
    def __init__(self, obs_dim, action_dim, hidden_size=64, use_orthogonal=True):
        super(ActorNetwork, self).__init__()
        self.base = MLPBase(obs_dim, hidden_size, use_orthogonal)

        # Initialization function for output layer
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        def init_(m):
            if isinstance(m, nn.Linear):
                init_method(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)
            return m

        self.action_head = init_(nn.Linear(hidden_size, action_dim))

    def forward(self, obs, action_masks=None):
        """
        Forward pass through actor network

        Args:
            obs (torch.Tensor): Observation tensor
            action_masks (torch.Tensor, optional): Boolean mask for valid actions

        Returns:
            torch.distributions.Categorical: Action distribution
        """
        features = self.base(obs)
        action_logits = self.action_head(features)

        # Apply action mask
        if action_masks is not None:
            action_logits[~action_masks] = -1e10

        return th.distributions.Categorical(logits=action_logits)


class CriticNetwork(nn.Module):
    """
    Centralized critic network for MAPPO
    """
    def __init__(self, obs_dim, hidden_size=64, use_orthogonal=True):
        super(CriticNetwork, self).__init__()
        self.base = MLPBase(obs_dim, hidden_size, use_orthogonal)

        # Initialization function for output layer
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        def init_(m):
            if isinstance(m, nn.Linear):
                init_method(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            return m

        self.value_head = init_(nn.Linear(hidden_size, 1))

    def forward(self, obs):
        """
        Forward pass through critic network

        Args:
            obs (torch.Tensor): Centralized observation tensor

        Returns:
            torch.Tensor: Value estimate
        """
        features = self.base(obs)
        return self.value_head(features)


class MAPPOPolicy:
    """
    MAPPO Policy class that handles the actor and critic networks
    """
    def __init__(self, agent_idx, obs_space, action_space, hidden_size=64,
                 lr_actor=3e-4, lr_critic=3e-4, use_orthogonal=True, device=th.device("cpu")):
        """
        Initialize the MAPPO policy

        Args:
            agent_idx: Index of the agent (0 or 1)
            obs_space: Observation space
            action_space: Action space
            hidden_size: Hidden layer size
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            use_orthogonal: Whether to use orthogonal initialization
            device: Device to use
        """
        self.agent_idx = agent_idx
        self.device = device

        # Determine observation dimension
        if 'visual_obs' in obs_space.spaces:
            obs_shape = obs_space.spaces['visual_obs'].shape
            self.obs_dim = int(np.prod(obs_shape))
        else:
            raise ValueError("Unsupported observation space")

        # Determine action dimension
        self.action_dim = action_space.n

        # Initialize actor
        self.actor = ActorNetwork(
            self.obs_dim,
            self.action_dim,
            hidden_size,
            use_orthogonal
        ).to(device)

        # If this is agent 0, also initialize the centralized critic
        # The critic is only initialized for agent 0 to avoid duplication
        if agent_idx == 0:
            # For centralized critic, we need observations from both agents + current subtasks
            centralized_obs_dim = self.obs_dim * 2 + Subtasks.NUM_SUBTASKS * 2
            self.critic = CriticNetwork(
                centralized_obs_dim,
                hidden_size,
                use_orthogonal
            ).to(device)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        else:
            self.critic = None

        # Actor optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

    def act(self, obs, action_mask=None, deterministic=False):
        """
        Sample an action from the policy given an observation

        Args:
            obs: Observation tensor
            action_mask: Boolean mask for valid actions
            deterministic: Whether to return the mode of the distribution

        Returns:
            action: Selected action
            log_prob: Log probability of the action
            entropy: Entropy of the distribution
        """
        # Flatten observation if necessary
        if isinstance(obs, dict):
            if 'visual_obs' in obs:
                obs_tensor = th.tensor(obs['visual_obs'], device=self.device).float().view(1, -1)
            else:
                raise ValueError("Unsupported observation format")
        else:
            obs_tensor = th.tensor(obs, device=self.device).float().view(1, -1)

        # Convert action mask to tensor if provided
        if action_mask is not None:
            action_mask = th.tensor(action_mask, device=self.device).bool()

        # Get action distribution
        with th.no_grad():
            dist = self.actor(obs_tensor, action_mask)

            # Sample action or take mode
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action.cpu().numpy(), log_prob.cpu().numpy(), entropy.cpu().numpy()

    def evaluate_actions(self, obs, actions, action_masks=None):
        """
        Evaluate log probability and entropy of given actions

        Args:
            obs: Observation tensor
            actions: Actions to evaluate
            action_masks: Boolean masks for valid actions

        Returns:
            log_probs: Log probabilities of actions
            entropy: Entropy of the distribution
        """
        # Get action distribution
        dist = self.actor(obs, action_masks)

        # Calculate log probabilities and entropy
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        return log_probs, entropy

    def get_value(self, centralized_obs):
        """
        Get value estimate from critic

        Args:
            centralized_obs: Centralized observation tensor

        Returns:
            values: Value estimates
        """
        assert self.critic is not None, "Critic is not initialized for this agent"

        with th.no_grad():
            values = self.critic(centralized_obs)

        return values.cpu().numpy()

    def update(self, rollout_data, clip_range=0.2, value_loss_coef=0.5, entropy_coef=0.01):
        """
        Update actor network using PPO

        Args:
            rollout_data: Data from rollout storage
            clip_range: PPO clip range parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient

        Returns:
            actor_loss: Actor loss value
            entropy: Entropy value
        """
        # Extract agent-specific data
        agent_key = f'agent_{self.agent_idx}'
        obs = rollout_data[f'{agent_key}_obs']
        action_masks = rollout_data[f'{agent_key}_action_masks']
        actions = rollout_data[f'{agent_key}_actions']
        old_log_probs = rollout_data[f'{agent_key}_log_probs']
        advantages = rollout_data[f'{agent_key}_advantages']

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten everything for mini-batch updates
        # (This could be modified to process in mini-batches)

        # Re-evaluate actions
        log_probs, entropy = self.evaluate_actions(obs, actions, action_masks)

        # Calculate PPO loss
        ratio = th.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages

        # Actor loss (negative because we're maximizing)
        actor_loss = -th.min(surr1, surr2).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        total_loss = actor_loss - entropy_coef * entropy
        total_loss.backward()
        # Clip gradient norm
        th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        return actor_loss.item(), entropy.item()

    def update_critic(self, centralized_obs, returns, clip_range=0.2, value_loss_coef=0.5):
        """
        Update critic network

        Args:
            centralized_obs: Centralized observations
            returns: Expected returns
            clip_range: PPO clip range parameter
            value_loss_coef: Value loss coefficient

        Returns:
            value_loss: Value loss
        """
        assert self.critic is not None, "Critic is not initialized for this agent"

        # Get value predictions
        values = self.critic(centralized_obs)

        # Calculate value loss (using Huber loss for stability)
        value_loss = F.mse_loss(values, returns)

        # Update critic
        self.critic_optimizer.zero_grad()
        (value_loss * value_loss_coef).backward()
        # Clip gradient norm
        th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        return value_loss.item()


class MAPPOTrainer:
    """
    MAPPO Trainer for HAHA managers
    """
    def __init__(self, env, worker_a, worker_b, args, hidden_size=64, lr_actor=3e-4,
                 lr_critic=3e-4, buffer_size=2048, gamma=0.99, gae_lambda=0.95):
        """
        Initialize the MAPPO trainer

        Args:
            env: Multi-agent environment
            worker_a: Worker agent for player 0
            worker_b: Worker agent for player 1
            args: Arguments
            hidden_size: Hidden layer size for networks
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            buffer_size: Size of the rollout buffer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.env = env
        self.worker_a = worker_a
        self.worker_b = worker_b
        self.args = args
        self.device = args.device

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.buffer_size = buffer_size
        self.n_epochs = 10
        self.clip_range = 0.2
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5

        # Initialize policies for both agents
        self.policies = [
            MAPPOPolicy(0, env.observation_space, env.action_space, hidden_size,
                       lr_actor, lr_critic, device=self.device),
            MAPPOPolicy(1, env.observation_space, env.action_space, hidden_size,
                       lr_actor, lr_critic, device=self.device)
        ]

        # Initialize rollout storage
        self.rollout_storage = RolloutStorage(
            num_agents=2,
            buffer_size=buffer_size,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=self.device,
            gae_lambda=gae_lambda,
            gamma=gamma
        )

        # Metrics for tracking
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def collect_rollouts(self, n_steps=None):
        """
        Collect rollouts by interacting with the environment

        Args:
            n_steps: Number of steps to collect (defaults to buffer_size)

        Returns:
            mean_reward: Mean episode reward during collection
        """
        n_steps = n_steps or self.buffer_size

        # Reset environment
        obs = self.env.reset()

        # Extract observations for each agent
        obs_a = obs['agent_0']
        obs_b = obs['agent_1']
        cent_obs = obs['centralized']

        # Track episode metrics
        episode_rewards = [0, 0]
        episode_length = 0
        completed_episodes = 0

        for step in range(n_steps):
            # Get action masks
            action_masks = self.env.action_masks()

            # Prepare observations for each agent
            obs_list = [obs_a, obs_b]
            action_masks_list = action_masks

            # Sample actions
            actions = []
            log_probs = []
            for i, policy in enumerate(self.policies):
                action, log_prob, _ = policy.act(obs_list[i], action_masks_list[i])
                actions.append(action[0])  # Remove extra dimension
                log_probs.append(log_prob[0])  # Remove extra dimension

            # Prepare centralized observations for value estimation
            # Flatten and concatenate observations and subtasks
            if isinstance(cent_obs['agent_0_obs'], dict) and 'visual_obs' in cent_obs['agent_0_obs']:
                flat_obs_a = cent_obs['agent_0_obs']['visual_obs'].reshape(-1)
                flat_obs_b = cent_obs['agent_1_obs']['visual_obs'].reshape(-1)
            else:
                raise ValueError("Unsupported observation format")

            # Create one-hot encodings for subtasks
            subtask_a_onehot = np.zeros(Subtasks.NUM_SUBTASKS)
            subtask_a_onehot[cent_obs['agent_0_subtask']] = 1

            subtask_b_onehot = np.zeros(Subtasks.NUM_SUBTASKS)
            subtask_b_onehot[cent_obs['agent_1_subtask']] = 1

            # Concatenate all centralized data
            centralized_features = np.concatenate([
                flat_obs_a, flat_obs_b, subtask_a_onehot, subtask_b_onehot
            ])

            # Convert to tensor
            centralized_tensor = th.tensor(centralized_features, dtype=th.float32, device=self.device).unsqueeze(0)

            # Get value estimates
            values = [
                self.policies[0].get_value(centralized_tensor)[0],  # Agent 0 estimates
                self.policies[0].get_value(centralized_tensor)[0]   # Use same value for agent 1 (centralized critic)
            ]

            # Execute actions in environment
            next_obs, rewards, done, info = self.env.step((actions[0], actions[1]))

            # Update episode metrics
            episode_rewards[0] += rewards[0]
            episode_rewards[1] += rewards[1]
            episode_length += 1

            # Add to rollout storage
            self.rollout_storage.add(
                obs=[obs_a, obs_b],
                centralized_obs=centralized_tensor,
                action_masks=action_masks,
                actions=actions,
                log_probs=log_probs,
                values=values,
                rewards=rewards,
                dones=[done, done]
            )

            # Update observations
            obs_a = next_obs['agent_0']
            obs_b = next_obs['agent_1']
            cent_obs = next_obs['centralized']

            # If episode finished, reset environment
            if done:
                self.episode_rewards.append(sum(episode_rewards) / 2)  # Average team reward
                self.episode_lengths.append(episode_length)

                # Reset environment
                obs = self.env.reset()
                obs_a = obs['agent_0']
                obs_b = obs['agent_1']
                cent_obs = obs['centralized']

                # Reset episode metrics
                episode_rewards = [0, 0]
                episode_length = 0
                completed_episodes += 1

        # After collecting rollouts, compute returns and advantages
        # Get final value estimates for bootstrapping
        if isinstance(cent_obs['agent_0_obs'], dict) and 'visual_obs' in cent_obs['agent_0_obs']:
            flat_obs_a = cent_obs['agent_0_obs']['visual_obs'].reshape(-1)
            flat_obs_b = cent_obs['agent_1_obs']['visual_obs'].reshape(-1)
        else:
            raise ValueError("Unsupported observation format")

        # Create one-hot encodings for subtasks
        subtask_a_onehot = np.zeros(Subtasks.NUM_SUBTASKS)
        subtask_a_onehot[cent_obs['agent_0_subtask']] = 1

        subtask_b_onehot = np.zeros(Subtasks.NUM_SUBTASKS)
        subtask_b_onehot[cent_obs['agent_1_subtask']] = 1

        # Concatenate all centralized data
        centralized_features = np.concatenate([
            flat_obs_a, flat_obs_b, subtask_a_onehot, subtask_b_onehot
        ])

        # Convert to tensor
        centralized_tensor = th.tensor(centralized_features, dtype=th.float32, device=self.device).unsqueeze(0)

        # Get final value estimates
        final_values = [
            self.policies[0].get_value(centralized_tensor)[0],
            self.policies[0].get_value(centralized_tensor)[0]
        ]

        # Compute returns and advantages
        self.rollout_storage.compute_returns_and_advantages(final_values)

        # Update total steps
        self.total_steps += n_steps

        # Return average episode reward if any episodes completed
        if completed_episodes > 0:
            return sum(self.episode_rewards[-completed_episodes:]) / completed_episodes
        else:
            return 0

    def update(self):
        """
        Update policies using PPO

        Returns:
            dict: Dictionary of training metrics
        """
        # Get all data from rollout storage
        rollout_data = self.rollout_storage.get_data()

        # Process centralized observations for critic update
        centralized_obs = th.cat(rollout_data['centralized_obs'], dim=0)

        # Get returns for both agents
        returns_a = rollout_data['agent_0_returns']
        returns_b = rollout_data['agent_1_returns']

        # Update metrics
        metrics = {
            'actor_loss_a': 0,
            'actor_loss_b': 0,
            'critic_loss': 0,
            'entropy_a': 0,
            'entropy_b': 0
        }

        # Perform multiple epochs of PPO updates
        for epoch in range(self.n_epochs):
            # Update actor for agent 0
            actor_loss_a, entropy_a = self.policies[0].update(
                rollout_data,
                self.clip_range,
                self.value_loss_coef,
                self.entropy_coef
            )

            # Update actor for agent 1
            actor_loss_b, entropy_b = self.policies[1].update(
                rollout_data,
                self.clip_range,
                self.value_loss_coef,
                self.entropy_coef
            )

            # Update centralized critic (only using agent 0's critic)
            # We use agent 0's returns, but could use an average or other combination
            critic_loss = self.policies[0].update_critic(
                centralized_obs,
                returns_a,
                self.clip_range,
                self.value_loss_coef
            )

            # Update metrics
            metrics['actor_loss_a'] += actor_loss_a / self.n_epochs
            metrics['actor_loss_b'] += actor_loss_b / self.n_epochs
            metrics['critic_loss'] += critic_loss / self.n_epochs
            metrics['entropy_a'] += entropy_a / self.n_epochs
            metrics['entropy_b'] += entropy_b / self.n_epochs

        # Clear rollout storage
        self.rollout_storage.clear()

        return metrics

    def train(self, total_timesteps, log_interval=100, eval_interval=1000):
        """
        Train the MAPPO policies

        Args:
            total_timesteps: Total number of timesteps to train for
            log_interval: Interval for logging metrics
            eval_interval: Interval for evaluation

        Returns:
            policies: Trained policies
        """
        timesteps_so_far = 0
        iterations = 0

        while timesteps_so_far < total_timesteps:
            # Collect rollouts
            mean_reward = self.collect_rollouts()
            timesteps_so_far += self.buffer_size

            # Update policies
            metrics = self.update()

            # Log metrics
            if iterations % log_interval == 0:
                print(f"Iteration {iterations}, Steps: {timesteps_so_far}/{total_timesteps}")
                print(f"Mean reward: {mean_reward:.2f}")
                print(f"Actor loss (A): {metrics['actor_loss_a']:.4f}, Actor loss (B): {metrics['actor_loss_b']:.4f}")
                print(f"Critic loss: {metrics['critic_loss']:.4f}")
                print(f"Entropy (A): {metrics['entropy_a']:.4f}, Entropy (B): {metrics['entropy_b']:.4f}")
                print("-" * 50)

            # Evaluate if needed
            if iterations % eval_interval == 0:
                # Implement evaluation if needed
                pass

            iterations += 1

        return self.policies

    def save(self, path):
        """
        Save the policies to disk

        Args:
            path: Path to save the policies to
        """
        import os
        os.makedirs(path, exist_ok=True)

        # Save actors
        th.save(self.policies[0].actor.state_dict(), os.path.join(path, "actor_a.pt"))
        th.save(self.policies[1].actor.state_dict(), os.path.join(path, "actor_b.pt"))

        # Save centralized critic
        th.save(self.policies[0].critic.state_dict(), os.path.join(path, "critic.pt"))

    def load(self, path):
        """
        Load the policies from disk

        Args:
            path: Path to load the policies from
        """
        import os

        # Load actors
        self.policies[0].actor.load_state_dict(th.load(os.path.join(path, "actor_a.pt"), map_location=self.device))
        self.policies[1].actor.load_state_dict(th.load(os.path.join(path, "actor_b.pt"), map_location=self.device))

        # Load centralized critic
        self.policies[0].critic.load_state_dict(th.load(os.path.join(path, "critic.pt"), map_location=self.device))