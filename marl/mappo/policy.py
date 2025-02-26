import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim

from marl.mappo.actor_network import ActorNetwork
from marl.mappo.critic_network import CriticNetwork
from oai_agents.common.subtasks import Subtasks


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
            obs: Observation dictionary from environment
            action_mask: Boolean mask for valid actions
            deterministic: Whether to return the mode of the distribution

        Returns:
            action: Selected action
            log_prob: Log probability of the action
            entropy: Entropy of the distribution
        """
        # Convert action mask to tensor if provided
        if action_mask is not None:
            action_mask = th.tensor(action_mask, device=self.device).bool()
            # Add batch dimension if needed
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
        elif 'subtask_mask' in obs:
            action_mask = th.tensor(obs['subtask_mask'], device=self.device).bool()
            # Add batch dimension if needed
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

        # Get visual observation and convert to tensor
        if 'visual_obs' in obs:
            # Create a copy to avoid stride issues
            visual_obs = np.ascontiguousarray(obs['visual_obs'])
            obs_tensor = th.tensor(visual_obs, device=self.device).float()
            # Flatten for feeding into the network
            obs_tensor = obs_tensor.reshape(1, -1) if len(obs_tensor.shape) == 3 else obs_tensor
        else:
            raise ValueError("Unsupported observation format")

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
        # Process action masks
        processed_masks = None
        if action_masks and len(action_masks) > 0:
            if len(action_masks) == len(obs):
                processed_masks = action_masks
            else:
                # Handle mismatch in batch sizes
                processed_masks = action_masks[:len(obs)]

        # Get action distribution
        dist = self.actor(obs, processed_masks)

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

        # Re-evaluate actions - make sure obs and action_masks are properly processed
        if len(obs) > 0:  # Make sure there are observations
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
        else:
            # Return zeros if no data
            return 0.0, 0.0

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