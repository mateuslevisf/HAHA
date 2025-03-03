"""
MAPPO Trainer for HAHA Manager Policy Training

This module implements Multi-Agent Proximal Policy Optimization (MAPPO) for training
HAHA managers with a centralized critic and decentralized actors.
"""

import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch as th
from oai_agents.common.subtasks import Subtasks
from marl.mappo.rollout_storage import RolloutStorage
from marl.mappo.policy import MAPPOPolicy

class MAPPOTrainer:
    """
    MAPPO Trainer for HAHA managers with improved monitoring
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
        self.entropy_coef = 0.05
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

        # New monitoring attributes
        self.training_metrics = {
            'iterations': [],
            'mean_rewards': [],
            'actor_loss_a': [],
            'actor_loss_b': [],
            'critic_loss': [],
            'entropy_a': [],
            'entropy_b': []
        }

        # Create logs directory
        self.logs_dir = os.path.join('marl', 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)

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
        episode_reward_history = []

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
            if isinstance(obs_a, dict) and 'visual_obs' in obs_a:
                flat_obs_a = obs_a['visual_obs'].reshape(-1)
                flat_obs_b = obs_b['visual_obs'].reshape(-1)
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
                obss=[obs_a, obs_b],
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
                team_reward = sum(episode_rewards) / 2  # Average team reward
                self.episode_rewards.append(team_reward)
                self.episode_lengths.append(episode_length)
                episode_reward_history.append(team_reward)

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
        if isinstance(obs_a, dict) and 'visual_obs' in obs_a:
            flat_obs_a = obs_a['visual_obs'].reshape(-1)
            flat_obs_b = obs_b['visual_obs'].reshape(-1)
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
            return sum(episode_reward_history) / completed_episodes
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
        centralized_obs = rollout_data['centralized_obs']

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

    def save_training_metrics(self):
        """Save training metrics as plots, including evaluation rewards if available"""
        # Create figure for rewards
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_metrics['iterations'], self.training_metrics['mean_rewards'], 'b-', label='Training Rewards')

        # Add evaluation rewards if available
        if 'eval_rewards' in self.training_metrics and len(self.training_metrics['eval_rewards']) > 0:
            plt.plot(self.training_metrics['eval_iterations'], self.training_metrics['eval_rewards'], 'r-',
                    marker='o', markersize=4, label='Evaluation Rewards')

        plt.title('Rewards per Iteration')
        plt.xlabel('Iterations')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.logs_dir, 'rewards.png'))
        plt.close()

        # Create figure for losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_metrics['iterations'], self.training_metrics['actor_loss_a'], 'r-', label='Actor A Loss')
        plt.plot(self.training_metrics['iterations'], self.training_metrics['actor_loss_b'], 'g-', label='Actor B Loss')
        plt.plot(self.training_metrics['iterations'], self.training_metrics['critic_loss'], 'b-', label='Critic Loss')
        plt.title('Training Losses')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.logs_dir, 'training_losses.png'))
        plt.close()

        # Create figure for entropy
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_metrics['iterations'], self.training_metrics['entropy_a'], 'r-', label='Entropy A')
        plt.plot(self.training_metrics['iterations'], self.training_metrics['entropy_b'], 'g-', label='Entropy B')
        plt.title('Policy Entropy')
        plt.xlabel('Iterations')
        plt.ylabel('Entropy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.logs_dir, 'policy_entropy.png'))
        plt.close()

        # Save raw data as numpy arrays for later analysis
        np.save(os.path.join(self.logs_dir, 'training_metrics.npy'), self.training_metrics)

    def train(self, total_timesteps, log_interval=100, eval_interval=1000):
        """
        Train the MAPPO policies with evaluation and best model saving

        Args:
            total_timesteps: Total number of timesteps to train for
            log_interval: Interval for logging metrics
            eval_interval: Interval for evaluation and potential model saving

        Returns:
            policies: Trained policies
        """
        timesteps_so_far = 0
        iterations = 0

        # Set up progress bar for total training
        pbar = tqdm(total=total_timesteps, desc="Training Progress",
                    unit="steps", ncols=100)

        # Track start time
        start_time = time.time()

        # Track best evaluation score for model saving
        best_eval_reward = float('-inf')
        best_model_path = os.path.join(self.args.save_dir, "best_model")

        while timesteps_so_far < total_timesteps:
            # Collect rollouts
            mean_reward = self.collect_rollouts()
            timesteps_so_far += self.buffer_size

            # Update policies
            metrics = self.update()

            # Store metrics
            self.training_metrics['iterations'].append(iterations)
            self.training_metrics['mean_rewards'].append(mean_reward)
            self.training_metrics['actor_loss_a'].append(metrics['actor_loss_a'])
            self.training_metrics['actor_loss_b'].append(metrics['actor_loss_b'])
            self.training_metrics['critic_loss'].append(metrics['critic_loss'])
            self.training_metrics['entropy_a'].append(metrics['entropy_a'])
            self.training_metrics['entropy_b'].append(metrics['entropy_b'])

            # Update progress bar
            pbar.update(self.buffer_size)
            pbar.set_postfix({
                'reward': f"{mean_reward:.2f}",
                'a_loss': f"{metrics['actor_loss_a']:.4f}",
                'c_loss': f"{metrics['critic_loss']:.4f}"
            })

            # Log metrics
            if iterations % log_interval == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = timesteps_so_far / elapsed_time

                print(f"\nIteration {iterations}, Steps: {timesteps_so_far}/{total_timesteps}")
                print(f"Mean reward: {mean_reward:.2f}")
                print(f"Actor loss (A): {metrics['actor_loss_a']:.4f}, Actor loss (B): {metrics['actor_loss_b']:.4f}")
                print(f"Critic loss: {metrics['critic_loss']:.4f}")
                print(f"Entropy (A): {metrics['entropy_a']:.4f}, Entropy (B): {metrics['entropy_b']:.4f}")
                print(f"Steps/sec: {steps_per_sec:.2f}, Estimated time remaining: {(total_timesteps - timesteps_so_far) / steps_per_sec / 60:.2f} minutes")
                print("-" * 50)

                # Save intermediate plots
                if iterations > 0:
                    self.save_training_metrics()

            # Evaluate and save best model
            if iterations % eval_interval == 0:
                # Run evaluation with deterministic policy
                eval_reward = self.evaluate_policies(n_episodes=5)
                print(f"\nEvaluation at iteration {iterations}: {eval_reward:.2f}")

                # Track evaluation performance
                if 'eval_rewards' not in self.training_metrics:
                    self.training_metrics['eval_rewards'] = []
                    self.training_metrics['eval_iterations'] = []

                self.training_metrics['eval_rewards'].append(eval_reward)
                self.training_metrics['eval_iterations'].append(iterations)

                # Save best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    print(f"New best model with reward: {best_eval_reward:.2f}, saving...")

                    # Save best model
                    os.makedirs(best_model_path, exist_ok=True)
                    self.save(best_model_path)

                    # Save best score info
                    with open(os.path.join(best_model_path, "best_score.txt"), "w") as f:
                        f.write(f"Iteration: {iterations}\n")
                        f.write(f"Timesteps: {timesteps_so_far}\n")
                        f.write(f"Eval reward: {best_eval_reward}\n")

            iterations += 1

        # Close progress bar
        pbar.close()

        # Final evaluation
        final_eval_reward = self.evaluate_policies(n_episodes=10)
        print(f"\nFinal evaluation: {final_eval_reward:.2f}")

        # Final save of training metrics
        self.save_training_metrics()

        # Final model save (if it's better than previous best)
        if final_eval_reward > best_eval_reward:
            self.save(best_model_path)

        print(f"\nTraining completed in {(time.time() - start_time) / 60:.2f} minutes")
        print(f"Best evaluation reward: {best_eval_reward:.2f}")
        print(f"Final evaluation reward: {final_eval_reward:.2f}")

        return self.policies

    def evaluate_policies(self, n_episodes=5):
        """
        Evaluate policies using deterministic action selection

        Args:
            n_episodes: Number of episodes to evaluate over

        Returns:
            float: Mean reward across evaluation episodes
        """
        # Store original state to restore after evaluation
        eval_obs = self.env.reset()

        total_rewards = []

        for _ in range(n_episodes):
            obs_a = eval_obs['agent_0']
            obs_b = eval_obs['agent_1']

            episode_reward = 0
            done = False

            while not done:
                # Get action masks
                action_masks = self.env.action_masks()

                # Sample actions deterministically
                action_a, _, _ = self.policies[0].act(obs_a, action_masks[0], deterministic=True)
                action_b, _, _ = self.policies[1].act(obs_b, action_masks[1], deterministic=True)

                # Execute actions in environment
                next_obs, rewards, done, _ = self.env.step((action_a[0], action_b[0]))

                # Update episode reward (average of both agents since it's cooperative)
                episode_reward += sum(rewards) / 2

                # Update observations
                obs_a = next_obs['agent_0']
                obs_b = next_obs['agent_1']

            # Record episode reward
            total_rewards.append(episode_reward)

            # Reset for next episode
            eval_obs = self.env.reset()

        # Calculate mean reward
        mean_eval_reward = sum(total_rewards) / len(total_rewards)

        return mean_eval_reward

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

        # Save training metrics plots in the same directory
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_metrics['iterations'], self.training_metrics['mean_rewards'], 'b-')
        plt.title('Mean Reward per Iteration')
        plt.xlabel('Iterations')
        plt.ylabel('Mean Reward')
        plt.grid(True)
        plt.savefig(os.path.join(path, 'mean_rewards.png'))
        plt.close()

        # Save raw metrics data
        np.save(os.path.join(path, 'training_metrics.npy'), self.training_metrics)

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

        # Try to load training metrics if they exist
        metrics_path = os.path.join(path, 'training_metrics.npy')
        if os.path.exists(metrics_path):
            self.training_metrics = np.load(metrics_path, allow_pickle=True).item()