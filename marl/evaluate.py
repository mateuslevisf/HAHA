#!/usr/bin/env python
"""
Evaluation script for MAPPO-trained HAHA agents.
This script loads trained HAHA agents and evaluates their performance in the environment.
"""

import os
import torch as th
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json

from oai_agents.common.arguments import get_arguments
from oai_agents.agents.agent_utils import load_agent
from oai_agents.agents.hrl import HierarchicalRL
from oai_agents.common.subtasks import Subtasks

from marl.env import MAHAHAEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate HAHA agents trained with MAPPO')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing the trained HAHA agents')
    parser.add_argument('--worker-a', type=str, required=True,
                        help='Path to worker model A')
    parser.add_argument('--worker-b', type=str, required=True,
                        help='Path to worker model B')
    parser.add_argument('--layout', type=str, default='cramped_room',
                        help='Overcooked layout name')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to evaluate')
    parser.add_argument('--horizon', type=int, default=400,
                        help='Maximum steps per episode')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic action selection')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment (not implemented)')
    parser.add_argument('--output-dir', type=str, default='marl/eval_results',
                        help='Directory to save evaluation results')

    return parser.parse_args()


def evaluate_agents(haha_a, haha_b, env, num_episodes=100, deterministic=False, render=False):
    """
    Evaluate the performance of two HAHA agents in the environment

    Args:
        haha_a: First HAHA agent
        haha_b: Second HAHA agent
        env: Environment to evaluate in
        num_episodes: Number of episodes to run
        deterministic: Whether to use deterministic action selection
        render: Whether to render the environment

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    completed_subtasks_a = []
    completed_subtasks_b = []
    subtask_choices_a = np.zeros(Subtasks.NUM_SUBTASKS)
    subtask_choices_b = np.zeros(Subtasks.NUM_SUBTASKS)

    # Run evaluation episodes
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        # Reset environment
        obs = env.reset()

        # Extract observations for each agent
        obs_a = obs['agent_0']
        obs_b = obs['agent_1']

        # Track episode metrics
        episode_reward = 0
        episode_length = 0
        episode_subtasks_a = 0
        episode_subtasks_b = 0

        done = False
        while not done:
            # Get agent A's action
            action_a, _ = haha_a.predict(obs_a, deterministic=deterministic)

            # Get agent B's action
            action_b, _ = haha_b.predict(obs_b, deterministic=deterministic)

            # Execute joint action in environment
            next_obs, rewards, done, info = env.step((action_a[0], action_b[0]))

            # Track subtask choices
            subtask_choices_a[info['agent_0_subtask']] += 1
            subtask_choices_b[info['agent_1_subtask']] += 1

            # Track completed subtasks
            if info['agent_0_subtask_completed']:
                episode_subtasks_a += 1
            if info['agent_1_subtask_completed']:
                episode_subtasks_b += 1

            # Update observations
            obs_a = next_obs['agent_0']
            obs_b = next_obs['agent_1']

            # Update episode metrics
            episode_reward += sum(rewards) / 2.0  # Average team reward
            episode_length += 1

            # Optional rendering
            if render:
                # Not implemented yet
                pass

            # Break if maximum episode length reached
            if episode_length >= env.horizon:
                done = True

        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        completed_subtasks_a.append(episode_subtasks_a)
        completed_subtasks_b.append(episode_subtasks_b)

    # Calculate evaluation metrics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    mean_subtasks_a = np.mean(completed_subtasks_a)
    mean_subtasks_b = np.mean(completed_subtasks_b)

    # Normalize subtask distributions
    subtask_dist_a = subtask_choices_a / np.sum(subtask_choices_a)
    subtask_dist_b = subtask_choices_b / np.sum(subtask_choices_b)

    # Return metrics
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'completed_subtasks_a': completed_subtasks_a,
        'completed_subtasks_b': completed_subtasks_b,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'mean_subtasks_a': mean_subtasks_a,
        'mean_subtasks_b': mean_subtasks_b,
        'subtask_dist_a': subtask_dist_a.tolist(),
        'subtask_dist_b': subtask_dist_b.tolist()
    }


def create_mappo_policy_wrapper(model_path, device):
    """
    Create a wrapper for the trained MAPPO policy

    Args:
        model_path: Path to the model file
        device: Device to load the model on

    Returns:
        wrapper: Policy wrapper object
    """
    # Define wrapper class that matches the interface expected by HierarchicalRL
    class MAPPOPolicyWrapper:
        def __init__(self, model_path, device):
            self.device = device
            self.actor = self.load_actor(model_path)

        def load_actor(self, path):
            # Import necessary classes
            from marl.mappo import ActorNetwork, MLPBase

            # Create actor network with same architecture as during training
            actor = ActorNetwork(
                obs_dim=1323,  # 7x7x27 flattened
                action_dim=Subtasks.NUM_SUBTASKS,
                hidden_size=64
            ).to(device)

            # Load weights
            actor.load_state_dict(th.load(path, map_location=device))
            actor.eval()

            return actor

        def predict(self, obs, deterministic=False):
            if 'subtask_mask' in obs:
                action_mask = th.tensor(obs['subtask_mask'], device=self.device).bool()
            else:
                action_mask = None

            # Prepare observation
            if 'visual_obs' in obs:
                # Create a copy to avoid stride issues
                visual_obs = np.ascontiguousarray(obs['visual_obs'])
                obs_tensor = th.tensor(visual_obs, device=self.device).float()
                # Flatten for the network
                obs_tensor = obs_tensor.reshape(1, -1)
            else:
                raise ValueError("Unsupported observation format")

            # Get action distribution
            with th.no_grad():
                # Add batch dimension to action mask if needed
                if action_mask is not None and action_mask.dim() == 1:
                    action_mask = action_mask.unsqueeze(0)

                dist = self.actor(obs_tensor, action_mask)

                # Sample action or take mode
                if deterministic:
                    action = dist.probs.argmax(dim=-1)
                else:
                    action = dist.sample()

            return action.cpu().numpy()

    # Create and return the wrapper
    return MAPPOPolicyWrapper(model_path, device)


def main():
    # Parse command line arguments
    args = parse_args()

    # Set device
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Get default arguments from the codebase
    common_args = get_arguments()
    common_args.horizon = args.horizon
    common_args.device = device

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load worker models
    print(f"Loading worker model A from: {args.worker_a}")
    worker_a = load_agent(Path(args.worker_a), common_args)

    print(f"Loading worker model B from: {args.worker_b}")
    worker_b = load_agent(Path(args.worker_b), common_args)

    # Create policy wrappers for the MAPPO-trained policies
    print(f"Loading MAPPO-trained policies from: {args.model_dir}")
    manager_a = create_mappo_policy_wrapper(os.path.join(args.model_dir, 'actor_a.pt'), device)
    manager_b = create_mappo_policy_wrapper(os.path.join(args.model_dir, 'actor_b.pt'), device)

    # Create HAHA agents
    haha_a = HierarchicalRL(worker_a, manager_a, common_args, name="haha_mappo_a")
    haha_b = HierarchicalRL(worker_b, manager_b, common_args, name="haha_mappo_b")

    # Create multi-agent environment
    print(f"Creating environment with layout: {args.layout}")
    env = MAHAHAEnv(
        worker_a=worker_a,
        worker_b=worker_b,
        args=common_args,
        shape_rewards=False,
        stack_frames=False,
        is_eval_env=True,
        horizon=args.horizon,
        layout_name=args.layout
    )

    # Run evaluation
    print(f"Starting evaluation for {args.episodes} episodes...")
    results = evaluate_agents(
        haha_a=haha_a,
        haha_b=haha_b,
        env=env,
        num_episodes=args.episodes,
        deterministic=args.deterministic,
        render=args.render
    )

    # Print summary statistics
    print("\nEvaluation Results:")
    print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean episode length: {results['mean_length']:.2f}")
    print(f"Mean completed subtasks (Agent A): {results['mean_subtasks_a']:.2f}")
    print(f"Mean completed subtasks (Agent B): {results['mean_subtasks_b']:.2f}")

    # Save results to file
    results_file = os.path.join(args.output_dir, 'eval_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Create visualization of results
    create_evaluation_plots(results, args.output_dir, args.layout)

    print(f"Results saved to {args.output_dir}")


def create_evaluation_plots(results, output_dir, layout_name):
    """Create visualizations of the evaluation results"""
    # Create figure for reward distribution
    plt.figure(figsize=(12, 10))

    # Plot reward distribution
    plt.subplot(2, 2, 1)
    plt.hist(results['episode_rewards'], bins=20, alpha=0.7)
    plt.axvline(results['mean_reward'], color='r', linestyle='dashed', linewidth=2)
    plt.title(f'Reward Distribution (Mean: {results["mean_reward"]:.2f})')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')

    # Plot episode length distribution
    plt.subplot(2, 2, 2)
    plt.hist(results['episode_lengths'], bins=20, alpha=0.7)
    plt.axvline(results['mean_length'], color='r', linestyle='dashed', linewidth=2)
    plt.title(f'Episode Length Distribution (Mean: {results["mean_length"]:.2f})')
    plt.xlabel('Episode Length')
    plt.ylabel('Frequency')

    # Plot subtask distribution for agent A
    plt.subplot(2, 2, 3)
    subtask_names = [Subtasks.IDS_TO_SUBTASKS[i] for i in range(Subtasks.NUM_SUBTASKS)]
    plt.bar(range(len(subtask_names)), results['subtask_dist_a'], alpha=0.7)
    plt.title('Subtask Distribution - Agent A')
    plt.xticks(range(len(subtask_names)), subtask_names, rotation=90)
    plt.ylabel('Frequency')
    plt.tight_layout()

    # Plot subtask distribution for agent B
    plt.subplot(2, 2, 4)
    plt.bar(range(len(subtask_names)), results['subtask_dist_b'], alpha=0.7)
    plt.title('Subtask Distribution - Agent B')
    plt.xticks(range(len(subtask_names)), subtask_names, rotation=90)
    plt.ylabel('Frequency')
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir, f'eval_results_{layout_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create comparison of subtask completions
    plt.figure(figsize=(10, 6))
    episodes = range(len(results['completed_subtasks_a']))
    plt.plot(episodes, results['completed_subtasks_a'], 'b-', label='Agent A')
    plt.plot(episodes, results['completed_subtasks_b'], 'r-', label='Agent B')
    plt.title('Completed Subtasks per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Completed Subtasks')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'subtask_completions_{layout_name}.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()