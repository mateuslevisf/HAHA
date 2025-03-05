#!/usr/bin/env python
"""
Main script for training HAHA managers using MAPPO.
This script loads pre-trained worker models and trains manager policies with a centralized critic.
Uses hardcoded parameters rather than command line arguments.
"""

import os
from pathlib import Path, PosixPath
import datetime
import torch as th
import torch.serialization
import matplotlib.pyplot as plt

from oai_agents.common.arguments import get_arguments
from oai_agents.agents.agent_utils import load_agent
from oai_agents.agents.hrl import HierarchicalRL

from marl.env import MAHAHAEnv
from marl.mappo.trainer import MAPPOTrainer

from oai_agents.agents.base_agent import SB3Wrapper
from stable_baselines3.ppo.ppo import PPO
from argparse import Namespace

# Add just the needed class to safe globals
torch.serialization.add_safe_globals([SB3Wrapper, PPO, Namespace, PosixPath])

import warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                       message="You are using `torch.load` with `weights_only=False`")


# ===== HARDCODED PARAMETERS =====
# Paths for pre-trained worker models
WORKER_A_PATH = 'agent_models_ICML/HAHA_fcp_61/worker' # REPLACE WITH ACTUAL PATH
WORKER_B_PATH = 'agent_models_ICML/HAHA_fcp_61/worker'   # REPLACE WITH ACTUAL PATH

# Training parameters
HIDDEN_SIZE = 256
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
BUFFER_SIZE = 2048
GAMMA = 0.99
GAE_LAMBDA = 0.95
TOTAL_TIMESTEPS = 1_000_000
LAYOUT_NAME = "cramped_room"

# Logging and evaluation
LOG_INTERVAL = 10
EVAL_INTERVAL = 50
SAVE_DIR = "./mappo_models"
# ================================

# Define a function to get the appropriate device
def get_optimal_device():
    """
    Get the optimal available device in this priority order:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU (fallback)
    """
    if th.cuda.is_available():
        device = th.device("cuda")
        print(f"CUDA GPU is available! Using {th.cuda.get_device_name(0)} for training.")
        print(f"CUDA device count: {th.cuda.device_count()}")
        # Set seeds for reproducibility
        th.cuda.manual_seed(42)
        # Optional: Print memory info
        print(f"GPU memory allocated: {th.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {th.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        return device
    elif hasattr(th, 'backends') and hasattr(th.backends, 'mps') and th.backends.mps.is_available():
        device = th.device("mps")
        print("MPS (Apple Silicon GPU) is available! Using M4 MAX for training.")
        # Set environment variables for better MPS performance
        import os
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        # Set seeds for reproducibility
        if hasattr(th.mps, 'manual_seed'):
            th.mps.manual_seed(42)
        return device
    else:
        print("No GPU acceleration available. Falling back to CPU.")
        return th.device("cpu")

# Tune performance based on selected device
def tune_performance(device):
    """
    Apply device-specific performance optimizations
    """
    if device.type == "cuda":
        # CUDA-specific optimizations
        th.backends.cudnn.benchmark = True  # Can speed up training if input sizes don't change
        # Use TF32 precision on Ampere or newer GPUs (faster with minimal precision loss)
        if hasattr(th.backends.cudnn, 'allow_tf32'):
            th.backends.cudnn.allow_tf32 = True
        if hasattr(th, 'set_float32_matmul_precision'):
            th.set_float32_matmul_precision('high')  # Options: 'highest', 'high', 'medium'

    elif device.type == "mps":
        # MPS-specific optimizations for Apple Silicon
        # Currently limited options, but this function can be expanded as MPS support improves
        pass

    # Set global precision if needed
    # th.set_default_dtype(th.float32)  # Use float32 for better numerical stability

    print(f"Performance tuning applied for {device.type}")
    return True

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
        """
        A wrapper class for the MAPPO policy to handle prediction with optional action masks.
        Attributes:
            policy: The policy object containing the actor network.
            actor: The actor network extracted from the policy.
        Methods:
            __init__(policy):
                Initializes the MAPPOPolicyWrapper with the given policy.
            predict(obs, deterministic=False):
                Predicts an action based on the given observation.
                Args:
                    obs (dict): A dictionary containing the observation data. Must contain 'visual_obs' key.
                                Optionally, it can contain 'subtask_mask' key for action masking.
                    deterministic (bool): If True, selects the action with the highest probability.
                                          If False, samples an action from the distribution.
                Returns:
                    numpy.ndarray: The predicted action.
                Raises:
                    ValueError: If the observation format is unsupported.
        """

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

        def save(self, path):
            """Save the policy to the specified path"""
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            # Save actor network
            th.save(self.policy.actor.state_dict(), os.path.join(path, "actor.pt"))

    # Create the wrapper
    manager = MAPPOPolicyWrapper(policy)

    # Create the HAHA agent
    haha = HierarchicalRL(worker, manager, args, name=name)

    return haha


def main():
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(SAVE_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Create logs directory
    logs_dir = os.path.join('marl', 'logs', timestamp)
    os.makedirs(logs_dir, exist_ok=True)

    print("Starting HAHA Manager MAPPO training...")
    print(f"Logs will be saved to {logs_dir}")
    print(f"Model checkpoints will be saved to {run_dir}")

    # Get default arguments from the codebase
    args = get_arguments()

    # Set the device to the optimal available option (CUDA > MPS > CPU)
    args.device = get_optimal_device()
    tune_performance(args.device)
    print(f"Using device: {args.device}")

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
    args.save_dir = run_dir

    # Save configuration
    with open(os.path.join(logs_dir, 'config.txt'), 'w', encoding='utf-8') as f:
        f.write("Training configuration:\n")
        f.write(f"- Worker A path: {WORKER_A_PATH}\n")
        f.write(f"- Worker B path: {WORKER_B_PATH}\n")
        f.write(f"- Hidden size: {HIDDEN_SIZE}\n")
        f.write(f"- Actor learning rate: {LR_ACTOR}\n")
        f.write(f"- Critic learning rate: {LR_CRITIC}\n")
        f.write(f"- Buffer size: {BUFFER_SIZE}\n")
        f.write(f"- Gamma: {GAMMA}\n")
        f.write(f"- GAE Lambda: {GAE_LAMBDA}\n")
        f.write(f"- Total timesteps: {TOTAL_TIMESTEPS}\n")
        f.write(f"- Layout: {LAYOUT_NAME}\n")

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
        horizon=args.horizon,
        layout_name=LAYOUT_NAME
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
    print(f"Saving trained policies to {run_dir}...")
    trainer.save(run_dir)

    # Create HAHA agents with trained managers
    haha_a = create_haha_from_mappo_policy(worker_a, trainer.policies[0], args, name="haha_mappo_a")
    haha_b = create_haha_from_mappo_policy(worker_b, trainer.policies[1], args, name="haha_mappo_b")

    # Save the HAHA agents
    haha_a.save(Path(run_dir) / "haha_a")
    haha_b.save(Path(run_dir) / "haha_b")

    # Create final summary plot
    plt.figure(figsize=(15, 10))

    # Mean rewards subplot
    plt.subplot(2, 2, 1)
    plt.plot(trainer.training_metrics['iterations'], trainer.training_metrics['mean_rewards'], 'b-')
    plt.title('Mean Reward per Iteration')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Reward')
    plt.grid(True)

    # Actor loss subplot
    plt.subplot(2, 2, 2)
    plt.plot(trainer.training_metrics['iterations'],
             trainer.training_metrics['actor_loss_a'],
             'r-',
             label='Actor A')
    plt.plot(trainer.training_metrics['iterations'],
             trainer.training_metrics['actor_loss_b'],
             'g-',
             label='Actor B')
    plt.title('Actor Losses')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Critic loss subplot
    plt.subplot(2, 2, 3)
    plt.plot(trainer.training_metrics['iterations'], trainer.training_metrics['critic_loss'], 'b-')
    plt.title('Critic Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)

    # Entropy subplot
    plt.subplot(2, 2, 4)
    plt.plot(trainer.training_metrics['iterations'],
             trainer.training_metrics['entropy_a'],
             'r-',
             label='Agent A')
    plt.plot(trainer.training_metrics['iterations'],
             trainer.training_metrics['entropy_b'],
             'g-',
             label='Agent B')
    plt.title('Policy Entropy')
    plt.xlabel('Iterations')
    plt.ylabel('Entropy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_summary.png'))
    plt.savefig(os.path.join(logs_dir, 'training_summary.png'))
    plt.close()

    print("Training complete!")
    print(f"All logs and models saved to {run_dir}")
    return haha_a, haha_b


if __name__ == "__main__":
    main()
