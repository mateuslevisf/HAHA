#!/usr/bin/env python
"""
Main script for training HAHA managers using MAPPO.
This script loads pre-trained worker models and trains manager policies with a centralized critic.
Uses hardcoded parameters rather than command line arguments.
"""

import os
import torch as th
import numpy as np
from pathlib import Path

from oai_agents.common.arguments import get_arguments
from oai_agents.agents.agent_utils import load_agent
from oai_agents.agents.hrl import HierarchicalRL
from oai_agents.common.subtasks import Subtasks

from marl.env import MAHAHAEnv
from marl.mappo import MAPPOTrainer


# ===== HARDCODED PARAMETERS =====
# Paths for pre-trained worker models
WORKER_A_PATH = 'agent_models_ICML/fcp_61/worker' # REPLACE WITH ACTUAL PATH
WORKER_B_PATH = 'agent_models_ICML/fcp_61/worker'   # REPLACE WITH ACTUAL PATH

# Training parameters
HIDDEN_SIZE = 64
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
BUFFER_SIZE = 2048
GAMMA = 0.99
GAE_LAMBDA = 0.95
TOTAL_TIMESTEPS = 1_000_000

# Logging and evaluation
LOG_INTERVAL = 10
EVAL_INTERVAL = 50
SAVE_DIR = "./mappo_models"
# ================================


def create_haha_from_mappo_policy(worker, policy, args, name="haha_mappo"):
    """
    Create a HAHA agent using a trained MAPPO policy as the manager

    Args:
        worker: Pre-trained worker agent
        policy: Trained MAPPO policy
        args: Arguments
        name: Name for the HAHA agent

    Returns:
        HierarchicalRL: HAHA agent with trained manager
    """
    # Create a wrapper for the MAPPO policy that matches the HierarchicalRL manager interface
    class MAPPOPolicyWrapper:
        def __init__(self, policy):
            self.policy = policy
            self.actor = policy.actor

        def predict(self, obs, deterministic=False):
            if 'subtask_mask' in obs:
                action_mask = th.tensor(obs['subtask_mask'], device=policy.device).bool()
            else:
                action_mask = None

            # Prepare observation
            if 'visual_obs' in obs:
                obs_tensor = th.tensor(obs['visual_obs'], device=policy.device).float().view(1, -1)
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

            return action.cpu().numpy()

    # Create the wrapper
    manager = MAPPOPolicyWrapper(policy)

    # Create the HAHA agent
    haha = HierarchicalRL(worker, manager, args, name=name)

    return haha


def main():
    print("Starting HAHA Manager MAPPO training...")

    # Get default arguments from the codebase
    args = get_arguments()

    # Update args with our hardcoded values
    args.hidden_size = HIDDEN_SIZE
    args.lr_actor = LR_ACTOR
    args.lr_critic = LR_CRITIC
    args.buffer_size = BUFFER_SIZE
    args.gamma = GAMMA
    args.gae_lambda = GAE_LAMBDA
    args.total_timesteps = TOTAL_TIMESTEPS
    args.log_interval = LOG_INTERVAL
    args.eval_interval = EVAL_INTERVAL
    args.save_dir = SAVE_DIR

    # Load pre-trained worker models
    print(f"Loading worker model A from: {WORKER_A_PATH}")
    worker_a = load_agent(Path(WORKER_A_PATH), args)

    print(f"Loading worker model B from: {WORKER_B_PATH}")
    worker_b = load_agent(Path(WORKER_B_PATH), args)

    # Create multi-agent environment
    print("Creating multi-agent environment...")
    env = MAHAHAEnv(
        worker_a=worker_a,
        worker_b=worker_b,
        args=args,
        shape_rewards=False,
        stack_frames=False,
        is_eval_env=False,
        horizon=args.horizon
    )

    # Create MAPPO trainer
    print("Setting up MAPPO trainer...")
    trainer = MAPPOTrainer(
        env=env,
        worker_a=worker_a,
        worker_b=worker_b,
        args=args,
        hidden_size=args.hidden_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda
    )

    # Train policies
    print(f"Starting training for {args.total_timesteps} timesteps...")
    trainer.train(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval
    )

    # Save the trained policies
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving trained policies to {save_path}...")
    trainer.save(save_path)

    # Create HAHA agents with trained managers
    haha_a = create_haha_from_mappo_policy(worker_a, trainer.policies[0], args, name="haha_mappo_a")
    haha_b = create_haha_from_mappo_policy(worker_b, trainer.policies[1], args, name="haha_mappo_b")

    # Save the HAHA agents
    haha_a.save(save_path / "haha_a")
    haha_b.save(save_path / "haha_b")

    print("Training complete!")
    return haha_a, haha_b


if __name__ == "__main__":
    main()