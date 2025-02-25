"""
Multi-Agent environment for training HAHA managers using MAPPO.
This environment adapts OvercookedGymEnv to simultaneously train two manager agents.
"""
from copy import deepcopy
from gym import spaces
import torch as th

from overcooked_ai_py.mdp.overcooked_mdp import Action

from oai_agents.gym_environments.base_overcooked_env import USEABLE_COUNTERS, OvercookedGymEnv
from oai_agents.common.subtasks import Subtasks, get_doable_subtasks


class MAHAHAEnv(OvercookedGymEnv):
    """
    Multi-Agent environment for training HAHA managers using MAPPO.

    This environment adapts OvercookedGymEnv to handle two agents with their own
    worker models simultaneously, enabling MAPPO training of HAHA managers.
    """

    def __init__(self, worker_a, worker_b, args, shape_rewards=False, stack_frames=False,
                 is_eval_env=False, horizon=None, **kwargs):
        """
        Initialize the multi-agent environment.

        Args:
            worker_a: Worker agent for the first manager
            worker_b: Worker agent for the second manager
            args: Common arguments
            shape_rewards: Whether to shape rewards
            stack_frames: Whether to stack frames in observations
            is_eval_env: Whether this is an evaluation environment
            horizon: Maximum number of steps per episode
        """
        super(MAHAHAEnv, self).__init__(
            shape_rewards=shape_rewards,
            stack_frames=stack_frames,
            is_eval_env=is_eval_env,
            full_init=True,
            args=args,
            **kwargs
        )

        # Store worker agents
        self.worker_a = worker_a
        self.worker_b = worker_b

        # Set up action spaces for subtasks
        self.action_space = spaces.Discrete(Subtasks.NUM_SUBTASKS)

        # Current subtasks being executed by each agent
        self.curr_subtasks = [Subtasks.SUBTASKS_TO_IDS['unknown'], Subtasks.SUBTASKS_TO_IDS['unknown']]

        # Store previous step's subtasks for action masking
        self.prev_subtasks = [Subtasks.SUBTASKS_TO_IDS['unknown'], Subtasks.SUBTASKS_TO_IDS['unknown']]

        # Add an observation field for goal objects
        if 'visual_obs' in self.obs_dict:
            # Add a layer for goal objects/markers
            self.obs_dict['goal_objects'] = spaces.Box(0, 1,
                shape=(self.num_enc_channels, *self.grid_shape), dtype=int)

        # Track rewards for each agent
        self.agent_rewards = [0, 0]

        # Separate current steps for worker execution
        self.worker_steps = [0, 0]
        self.max_worker_steps = 10  # Maximum steps a worker can take before manager selects a new subtask

        # Initialize centralized observation space
        self._setup_centralized_spaces()

    def _setup_centralized_spaces(self):
        """
        Set up observation and action spaces for centralized training.
        The centralized critic will need observations from both agents.
        """
        # Create a centralized observation space that combines both agents' observations
        self.centralized_observation_space = deepcopy(self.observation_space)

        # Action space for joint actions (both agents' subtasks)
        self.joint_action_space = spaces.MultiDiscrete([Subtasks.NUM_SUBTASKS, Subtasks.NUM_SUBTASKS])

    def action_masks(self, p_idx=None):
        """Get action masks for valid subtasks for a specific player."""
        if p_idx is None:
            # Return masks for both agents
            return [self._get_agent_action_mask(0), self._get_agent_action_mask(1)]
        else:
            return self._get_agent_action_mask(p_idx)

    def _get_agent_action_mask(self, p_idx):
        """Get action mask for valid subtasks for a specific player."""
        return get_doable_subtasks(
            self.state,
            self.prev_subtasks[p_idx],
            self.layout_name,
            self.terrain,
            p_idx,
            self.valid_counters,
            USEABLE_COUNTERS.get(self.layout_name, 5)
        ).astype(bool)

    def get_obs_for_agent(self, p_idx, for_worker=False):
        """Get observations for a specific agent."""
        goal_objects = None
        if for_worker and self.curr_subtasks[p_idx] != Subtasks.SUBTASKS_TO_IDS['unknown']:
            goal_objects = Subtasks.IDS_TO_GOAL_MARKERS[self.curr_subtasks[p_idx]]

        # Get base observation
        obs = self.get_obs(p_idx=p_idx, goal_objects=goal_objects)

        # Add action mask for managers
        if not for_worker:
            obs['subtask_mask'] = self._get_agent_action_mask(p_idx)

        return obs

    def get_centralized_obs(self):
        """
        Get a centralized observation that combines information from both agents.
        Used by the centralized critic in MAPPO.
        """
        obs_a = self.get_obs_for_agent(0)
        obs_b = self.get_obs_for_agent(1)

        # Centralized observation contains both agents' observations and their current subtasks
        cent_obs = {
            'agent_0_obs': obs_a,
            'agent_1_obs': obs_b,
            'agent_0_subtask': self.curr_subtasks[0],
            'agent_1_subtask': self.curr_subtasks[1],
            'state': self.state  # Include full state information
        }

        return cent_obs

    def step(self, joint_subtasks):
        """
        Take a step in the environment given both agents' subtask selections.

        Args:
            joint_subtasks: Tuple of (agent_a_subtask, agent_b_subtask)

        Returns:
            tuple: (observations, rewards, done, info)
        """
        # Unpack joint subtasks
        subtask_a, subtask_b = joint_subtasks
        self.curr_subtasks = [subtask_a, subtask_b]

        # Execute worker actions based on selected subtasks
        joint_action = [Action.STAY, Action.STAY]
        done = False
        cumulative_reward = 0
        self.agent_rewards = [0, 0]

        # Track if subtasks were completed
        subtask_completed = [False, False]

        # Execute workers for up to max_worker_steps or until both subtasks are completed
        for step in range(self.max_worker_steps):
            # Only execute worker for agents with non-completed subtasks
            for p_idx in [0, 1]:
                if subtask_completed[p_idx]:
                    continue

                if self.curr_subtasks[p_idx] != Subtasks.SUBTASKS_TO_IDS['unknown']:
                    # Get worker observation with goal markers
                    worker_obs = self.get_obs_for_agent(p_idx, for_worker=True)

                    # Get worker action
                    worker = self.worker_a if p_idx == 0 else self.worker_b
                    with th.no_grad():
                        joint_action[p_idx] = Action.INDEX_TO_ACTION[worker.predict(worker_obs)[0]]
                else:
                    # Unknown subtask, just stay
                    joint_action[p_idx] = Action.STAY

            # Execute joint action in environment
            prev_state = deepcopy(self.state)
            self.state, reward, done, info = self.env.step(joint_action)

            # Add to cumulative reward
            cumulative_reward += reward

            # Check if subtasks were completed
            for p_idx in [0, 1]:
                if subtask_completed[p_idx]:
                    continue

                # Check if agent completed its subtask
                if joint_action[p_idx] == Action.INTERACT:
                    # Getting agent's previous and current objects and the tile in front
                    tile_in_front = self.mdp.terrain_mtx[self.state.players[p_idx].position[1] +
                                                      self.state.players[p_idx].orientation[1]][
                                                      self.state.players[p_idx].position[0] +
                                                      self.state.players[p_idx].orientation[0]]

                    prev_obj = prev_state.players[p_idx].held_object.name if prev_state.players[p_idx].held_object else None
                    curr_obj = self.state.players[p_idx].held_object.name if self.state.players[p_idx].held_object else None

                    from oai_agents.common.subtasks import calculate_completed_subtask
                    completed = calculate_completed_subtask(prev_obj, curr_obj, tile_in_front)

                    # If agent completed its assigned subtask, mark as completed
                    if completed == self.curr_subtasks[p_idx]:
                        subtask_completed[p_idx] = True
                        self.agent_rewards[p_idx] += 1  # Reward for completing subtask

                # Also check if the subtask is no longer possible
                doable = self._get_agent_action_mask(p_idx)[self.curr_subtasks[p_idx]]
                if not doable:
                    subtask_completed[p_idx] = True
                    self.agent_rewards[p_idx] -= 0.5  # Penalty for selecting a subtask that became invalid

            # If all subtasks completed or environment is done, break
            if all(subtask_completed) or done:
                break

        # Record previous subtasks for action masking
        self.prev_subtasks = deepcopy(self.curr_subtasks)

        # Game-level reward (e.g., soup delivered) is shared equally
        for p_idx in [0, 1]:
            self.agent_rewards[p_idx] += cumulative_reward / 2

            # Small penalty for selecting unknown subtask
            if self.curr_subtasks[p_idx] == Subtasks.SUBTASKS_TO_IDS['unknown']:
                self.agent_rewards[p_idx] -= 0.1

        # Get observations for next step
        obs = {
            'agent_0': self.get_obs_for_agent(0),
            'agent_1': self.get_obs_for_agent(1),
            'centralized': self.get_centralized_obs()
        }

        # Additional info
        info = {
            'agent_0_subtask': self.curr_subtasks[0],
            'agent_1_subtask': self.curr_subtasks[1],
            'agent_0_subtask_completed': subtask_completed[0],
            'agent_1_subtask_completed': subtask_completed[1],
            'sparse_reward': cumulative_reward,
            'shaped_rewards': self.agent_rewards
        }

        return obs, self.agent_rewards, done, info

    def reset(self, p_idx=None):
        """Reset the environment."""
        # Reset the base environment
        super().reset()

        # Reset subtasks
        self.curr_subtasks = [Subtasks.SUBTASKS_TO_IDS['unknown'], Subtasks.SUBTASKS_TO_IDS['unknown']]
        self.prev_subtasks = [Subtasks.SUBTASKS_TO_IDS['unknown'], Subtasks.SUBTASKS_TO_IDS['unknown']]

        # Reset rewards
        self.agent_rewards = [0, 0]

        # Get observations
        obs = {
            'agent_0': self.get_obs_for_agent(0),
            'agent_1': self.get_obs_for_agent(1),
            'centralized': self.get_centralized_obs()
        }

        return obs