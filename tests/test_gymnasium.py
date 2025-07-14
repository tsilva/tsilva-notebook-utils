# tests/test_collect_rollouts.py
import unittest
import numpy as np
import torch

import pytest
from typing import Optional
from tsilva_notebook_utils.gymnasium import collect_rollouts, group_trajectories_by_episode, build_env        # ← adjust to your package
import multiprocessing
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------- #
#  Minimal but versatile dummy VecEnv                                    #
# ---------------------------------------------------------------------- #
class DummyVecEnv:
    """
    A mock vector-env whose 'episode length' can differ per env
    so we can trigger dones at different times.
    """
    def __init__(self, lens, obs_dim=4):
        """
        lens : Sequence[int] – episode length for each env
        """
        self.ep_lens   = np.asarray(lens)
        self.num_envs  = len(self.ep_lens)
        self.obs_dim   = obs_dim
        self._obs      = np.ones((self.num_envs, obs_dim), dtype=np.float32)
        self.counters  = np.zeros(self.num_envs, dtype=np.int32)

    # Gymnasium-style API ------------------------------------------------
    def reset(self):
        self.counters[:] = 0
        return self._obs.copy()

    def step(self, actions):
        self.counters += 1
        obs    = self._obs.copy()
        reward = np.ones(self.num_envs, dtype=np.float32)
        done   = self.counters >= self.ep_lens
        infos  = [{} for _ in range(self.num_envs)]
        return obs, reward, done, infos

    def get_images(self):
        #  tiny dummy RGB frames
        return [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(self.num_envs)]


# ---------------------------------------------------------------------- #
#  Dummy models that actually have parameters so .parameters() is safe   #
# ---------------------------------------------------------------------- #
class ConstPolicy(torch.nn.Module):
    """Returns constant logits for 3 discrete actions."""
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.))   # single param

    def forward(self, x):
        # shape: (batch, 3) – all zeros → uniform categorical
        return torch.zeros(x.shape[0], 3, device=x.device)


class ConstValue(torch.nn.Module):
    """Predicts value = 1 for every state (so GAE is non-trivial)."""
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return torch.ones(x.shape[0], 1, device=x.device)


# ---------------------------------------------------------------------- #
#  Test-case mixins / helpers                                            #
# ---------------------------------------------------------------------- #
def output_lengths(matchers, env, T):
    """Assert that every returned tensor has length len(matchers)."""
    expected = env.num_envs * T
    for tensor in matchers:
        assert len(tensor) == expected


# ---------------------------------------------------------------------- #
#  Actual tests                                                          #
# ---------------------------------------------------------------------- #
class TestCollectRollouts(unittest.TestCase):

    # -------------------- Sanity / assertion branch ------------------- #
    def test_raises_without_limits(self):
        env = DummyVecEnv([1])
        with self.assertRaises(AssertionError):
            collect_rollouts(env, ConstPolicy(), n_steps=None, n_episodes=None)

    # --------------------- Stop after n_steps only -------------------- #
    def test_stop_by_n_steps(self):
        env   = DummyVecEnv([10, 10])   # long episodes
        steps = 4
        res   = collect_rollouts(
            env, ConstPolicy(), ConstValue(),
            n_steps=steps, n_episodes=None,
            deterministic=True, collect_frames=False
        )
        states, actions, rewards, dones, *_ = res
        output_lengths((states, actions, rewards, dones), env, T=steps)
        # exactly 'steps' timesteps collected
        self.assertEqual(dones.view(-1).shape[0], env.num_envs * steps)

    # -------------------- Stop after n_episodes only ------------------ #
    def test_stop_by_n_episodes(self):
        env = DummyVecEnv([3, 5])   # 1st env ends at t=3, 2nd at t=5
        res = collect_rollouts(
            env, ConstPolicy(), ConstValue(),
            n_steps=None, n_episodes=1,   # first episode across ANY env
            deterministic=True
        )
        # first env finishes in 3 steps → T=3
        T = 3
        states, actions, *_ = res[:4]
        output_lengths((states, actions), env, T)

    # ----------------- Both limits – earlier one wins ----------------- #
    def test_both_limits(self):
        env = DummyVecEnv([6, 6])
        res = collect_rollouts(
            env, ConstPolicy(), ConstValue(),
            n_steps=4, n_episodes=10,    # n_steps triggers first
        )
        states = res[0]
        self.assertEqual(states.shape[0], env.num_envs * 4)

    # --------------- No value net & no advantage normalisation -------- #
    def test_value_none_no_norm(self):
        env = DummyVecEnv([2])
        (states, actions, rewards, dones,
         logps, values, advs, returns, _) = collect_rollouts(
            env, ConstPolicy(), value_model=None,
            n_steps=2, normalize_advantage=False
        )
        # Values should all be 0; therefore returns==advs
        self.assertTrue(torch.allclose(advs, returns))
        # logps produced by uniform categorical (all zeros logits) → -log(3)
        self.assertTrue(torch.allclose(
            logps,
            torch.full_like(logps, fill_value=-np.log(3.0))
        ))

    # ------------------- Frame collection branch on ------------------ #
    def test_collect_frames_on(self):
        env = DummyVecEnv([1, 1])
        *_, frames = collect_rollouts(
            env, ConstPolicy(), n_steps=1, collect_frames=True
        )
        # should have placeholder RGB frames, not zeros
        self.assertIsInstance(frames[0], np.ndarray)
        self.assertEqual(frames[0].shape[-1], 3)

    # ------------------ Frame collection branch off ------------------ #
    def test_collect_frames_off(self):
        env = DummyVecEnv([1])
        *_, frames = collect_rollouts(
            env, ConstPolicy(), n_steps=1, collect_frames=False
        )
        # should be list of zeros
        self.assertEqual(frames, [0])

    # ---------------------- Stochastic actions path ------------------ #
    def test_sampling_mode(self):
        env = DummyVecEnv([2])
        _, actions1, *_ = collect_rollouts(
            env, ConstPolicy(), n_steps=2, deterministic=False
        )
        _, actions2, *_ = collect_rollouts(
            env, ConstPolicy(), n_steps=2, deterministic=False
        )
        # Very small chance they match exactly – use it as a heuristic
        self.assertFalse(torch.equal(actions1, actions2))

    # ------------------------------------------------------------------ #
    #  1. n_episodes wins over n_steps (opposite ordering)               #
    # ------------------------------------------------------------------ #
    def test_n_episodes_wins(self):
        env = DummyVecEnv([2, 2])           # every env ends at t = 2
        # Give a very *large* step budget – episode limit should stop first
        res = collect_rollouts(
            env, ConstPolicy(), ConstValue(),
            n_steps=999, n_episodes=1
        )
        states = res[0]
        # Only 2 timesteps (t=0,1) collected from each env → 4 transitions
        assert states.shape[0] == 4


    # ------------------------------------------------------------------ #
    # 2. Advantage normalisation – compare with population std            #
    # ------------------------------------------------------------------ #
    def test_advantage_standardisation(self):
        env = DummyVecEnv([5, 5])           # 10 samples for stability
        _, _, _, _, _, _, advs, _, _ = collect_rollouts(
            env, ConstPolicy(), ConstValue(),
            n_steps=5, normalize_advantage=True
        )
        mu  = advs.mean().item()
        # unbiased=False → population standard deviation
        std = advs.std(unbiased=False).item()
        assert abs(mu) < 1e-6
        # 0.5 % tolerance is plenty (handles fp noise)
        assert abs(std - 1.0) < 5e-3

    # ------------------------------------------------------------------ #
    #  3. Zero-variance advantage → no NaNs thanks to adv_norm_eps       #
    # ------------------------------------------------------------------ #
    def test_zero_variance_advantage(self):
        class ZeroRewardEnv(DummyVecEnv):
            def step(self, actions):
                obs, reward, done, infos = super().step(actions)
                reward[:] = 0.0            # make returns == values
                return obs, reward, done, infos

        env = ZeroRewardEnv([2])
        # ValueNet predicts 0 → perfect fit → advantages all zero
        class ZeroValue(ConstValue):
            def forward(self, x): return torch.zeros(x.shape[0], 1)

        *_, advs, _, _ = collect_rollouts(
            env, ConstPolicy(), ZeroValue(),
            n_steps=2, normalize_advantage=True
        )
        # Should all be 0, and crucially not NaN / inf
        assert torch.allclose(advs, torch.zeros_like(advs), atol=0, rtol=0)
        assert not torch.isnan(advs).any()


    # ------------------------------------------------------------------ #
    # 4. Done masking – allow tiny fp noise                               #
    # ------------------------------------------------------------------ #
    def test_gae_resets_after_done(self):
        env = DummyVecEnv([1, 3])           # env-0 terminates immediately
        *_ , advs, _, _ = collect_rollouts(
            env, ConstPolicy(), ConstValue(),
            n_steps=3, normalize_advantage=False
        )
        advs = advs.reshape(2, 3)           # (E, T)
        # post-done advantages should be ~0 up to 1e-6
        assert torch.all(advs[0, 1:].abs() < 1e-6)


    # ------------------------------------------------------------------ #
    #  5. Helper _flat_env_major preserves dtype                         #
    # ------------------------------------------------------------------ #
    def test_flatten_dtypes(self):
        env = DummyVecEnv([1])
        outs = collect_rollouts(env, ConstPolicy(), n_steps=1)
        states, actions, rewards, dones, logps, values, *_ = outs
        assert actions.dtype   == torch.int64
        assert rewards.dtype   == torch.float32
        assert dones.dtype     == torch.bool
        assert values.dtype    == torch.float32
        assert states.dtype    == torch.float32


    # ------------------------------------------------------------------ #
    #  6. Single-env corner case                                         #
    # ------------------------------------------------------------------ #
    def test_single_env(self):
        env = DummyVecEnv([4])               # num_envs == 1
        states, _, _, _, _, _, _, _, _ = collect_rollouts(
            env, ConstPolicy(), n_steps=4
        )
        # Shape should be (T, obs_dim) = (4, 4)
        assert states.shape == (4, env.obs_dim)


    # ------------------------------------------------------------------ #
    #  7. Frame order is env-major                                        #
    # ------------------------------------------------------------------ #
    def test_frame_order_env_major(self):
        class CountingEnv(DummyVecEnv):
            def __init__(self):
                super().__init__([2, 2])
                self.frame_counter = 0

            def get_images(self):
                # return distinct numbers so we can spot ordering
                out = []
                for e in range(self.num_envs):
                    out.append(np.full((1,), self.frame_counter + e, dtype=np.int32))
                self.frame_counter += self.num_envs
                return out

        env = CountingEnv()
        *_, frames = collect_rollouts(
            env, ConstPolicy(), n_steps=2, collect_frames=True
        )
        # After env-major flattening we expect [0, 2, 1, 3]
        #   Explanation: frame 0 & 1 were t=0, env-0/env-1
        #                frame 2 & 3 were t=1, env-0/env-1
        expected = [0, 2, 1, 3]
        assert [int(f[0]) for f in frames] == expected


# ---------------------------------------------------------------------- #
#  Tests for group_trajectories_by_episode                               #
# ---------------------------------------------------------------------- #
class TestGroupTrajectoriesByEpisode(unittest.TestCase):
    
    def _create_dummy_trajectories(self, done_pattern, obs_dim=4, num_actions=3):
        """Helper to create dummy trajectory data with specified done pattern."""
        T = len(done_pattern)
        
        # Create dummy data
        states = torch.randn(T, obs_dim)
        actions = torch.randint(0, num_actions, (T,))
        rewards = torch.randn(T)
        dones = torch.tensor(done_pattern, dtype=torch.bool)
        logps = torch.randn(T)
        values = torch.randn(T)
        advs = torch.randn(T)
        returns = torch.randn(T)
        
        return (states, actions, rewards, dones, logps, values, advs, returns)
    
    def test_single_episode_complete(self):
        """Test grouping a single complete episode."""
        # Single episode that ends
        done_pattern = [False, False, False, True]
        trajectories = self._create_dummy_trajectories(done_pattern)
        
        episodes = group_trajectories_by_episode(trajectories)
        
        self.assertEqual(len(episodes), 1)
        self.assertEqual(len(episodes[0]), 4)  # 4 steps in the episode
        
        # Verify structure - each step should be a tuple of (state, action, reward, done, ...)
        for i, step in enumerate(episodes[0]):
            self.assertEqual(len(step), 8)  # 8 trajectory components
            self.assertEqual(step[3], done_pattern[i])  # done flag at index 3
    
    def test_single_episode_incomplete(self):
        """Test grouping a single incomplete episode (no terminal done)."""
        # Episode that doesn't end
        done_pattern = [False, False, False, False]
        trajectories = self._create_dummy_trajectories(done_pattern)
        
        episodes = group_trajectories_by_episode(trajectories)
        
        # Should return empty list since no episode is completed
        self.assertEqual(len(episodes), 0)
    
    def test_multiple_complete_episodes(self):
        """Test grouping multiple complete episodes."""
        # Two complete episodes
        done_pattern = [False, False, True, False, True]
        trajectories = self._create_dummy_trajectories(done_pattern)
        
        episodes = group_trajectories_by_episode(trajectories)
        
        self.assertEqual(len(episodes), 2)
        self.assertEqual(len(episodes[0]), 3)  # First episode: steps 0,1,2
        self.assertEqual(len(episodes[1]), 2)  # Second episode: steps 3,4
        
        # Verify done flags
        self.assertFalse(episodes[0][0][3])  # step 0
        self.assertFalse(episodes[0][1][3])  # step 1
        self.assertTrue(episodes[0][2][3])   # step 2 (terminal)
        self.assertFalse(episodes[1][0][3])  # step 3
        self.assertTrue(episodes[1][1][3])   # step 4 (terminal)
    
    def test_episodes_with_incomplete_final(self):
        """Test episodes where the last episode is incomplete."""
        # One complete episode followed by incomplete one
        done_pattern = [False, True, False, False]
        trajectories = self._create_dummy_trajectories(done_pattern)
        
        episodes = group_trajectories_by_episode(trajectories)
        
        self.assertEqual(len(episodes), 1)  # Only the complete episode
        self.assertEqual(len(episodes[0]), 2)  # Steps 0,1
        
        # Verify the complete episode
        self.assertFalse(episodes[0][0][3])  # step 0
        self.assertTrue(episodes[0][1][3])   # step 1 (terminal)
    
    def test_immediate_termination(self):
        """Test episode that terminates immediately (done=True at t=0)."""
        done_pattern = [True, False, False]
        trajectories = self._create_dummy_trajectories(done_pattern)
        
        episodes = group_trajectories_by_episode(trajectories)
        
        self.assertEqual(len(episodes), 1)
        self.assertEqual(len(episodes[0]), 1)  # Single step episode
        self.assertTrue(episodes[0][0][3])     # done=True
    
    def test_consecutive_terminations(self):
        """Test consecutive terminations (multiple single-step episodes)."""
        done_pattern = [True, True, True]
        trajectories = self._create_dummy_trajectories(done_pattern)
        
        episodes = group_trajectories_by_episode(trajectories)
        
        self.assertEqual(len(episodes), 3)
        for i, episode in enumerate(episodes):
            self.assertEqual(len(episode), 1)  # Each episode has 1 step
            self.assertTrue(episode[0][3])     # All are terminal
    
    def test_empty_trajectories(self):
        """Test with empty trajectory data."""
        # Create empty trajectories
        states = torch.empty(0, 4)
        actions = torch.empty(0, dtype=torch.long)
        rewards = torch.empty(0)
        dones = torch.empty(0, dtype=torch.bool)
        logps = torch.empty(0)
        values = torch.empty(0)
        advs = torch.empty(0)
        returns = torch.empty(0)
        
        trajectories = (states, actions, rewards, dones, logps, values, advs, returns)
        episodes = group_trajectories_by_episode(trajectories)
        
        self.assertEqual(len(episodes), 0)
    
    def test_data_integrity(self):
        """Test that the original data is preserved correctly in episodes."""
        done_pattern = [False, True, False, True]
        trajectories = self._create_dummy_trajectories(done_pattern)
        
        # Store original values for verification
        orig_states = trajectories[0].clone()
        orig_actions = trajectories[1].clone()
        orig_rewards = trajectories[2].clone()
        
        episodes = group_trajectories_by_episode(trajectories)
        
        # Verify first episode (steps 0,1)
        self.assertTrue(torch.equal(episodes[0][0][0], orig_states[0]))  # state at t=0
        self.assertTrue(torch.equal(episodes[0][1][0], orig_states[1]))  # state at t=1
        self.assertEqual(episodes[0][0][1], orig_actions[0])             # action at t=0
        self.assertEqual(episodes[0][1][1], orig_actions[1])             # action at t=1
        self.assertEqual(episodes[0][0][2], orig_rewards[0])             # reward at t=0
        self.assertEqual(episodes[0][1][2], orig_rewards[1])             # reward at t=1
        
        # Verify second episode (steps 2,3)
        self.assertTrue(torch.equal(episodes[1][0][0], orig_states[2]))  # state at t=2
        self.assertTrue(torch.equal(episodes[1][1][0], orig_states[3]))  # state at t=3
        self.assertEqual(episodes[1][0][1], orig_actions[2])             # action at t=2
        self.assertEqual(episodes[1][1][1], orig_actions[3])             # action at t=3
    
    def test_mixed_tensor_types(self):
        """Test with different tensor types and devices."""
        done_pattern = [False, True]
        
        # Create trajectories with different dtypes
        states = torch.randn(2, 4, dtype=torch.float32)
        actions = torch.tensor([0, 1], dtype=torch.int64)
        rewards = torch.tensor([1.5, -0.5], dtype=torch.float32)
        dones = torch.tensor(done_pattern, dtype=torch.bool)
        logps = torch.tensor([-1.1, -2.3], dtype=torch.float64)  # Different dtype
        values = torch.tensor([0.8, 1.2], dtype=torch.float32)
        advs = torch.tensor([0.1, -0.2], dtype=torch.float32)
        returns = torch.tensor([1.6, 1.0], dtype=torch.float32)
        
        trajectories = (states, actions, rewards, dones, logps, values, advs, returns)
        episodes = group_trajectories_by_episode(trajectories)
        
        self.assertEqual(len(episodes), 1)
        self.assertEqual(len(episodes[0]), 2)
        
        # Verify dtypes are preserved
        self.assertEqual(episodes[0][0][0].dtype, torch.float32)  # states
        self.assertEqual(episodes[0][0][1].dtype, torch.int64)    # actions
        self.assertEqual(episodes[0][0][4].dtype, torch.float64)  # logps
    
    def test_large_trajectory(self):
        """Test with a larger trajectory to ensure performance is reasonable."""
        # Create a pattern with multiple episodes of varying lengths
        done_pattern = ([False] * 10 + [True] +      # Episode 1: 11 steps
                       [False] * 5 + [True] +        # Episode 2: 6 steps  
                       [False] * 15 + [True] +       # Episode 3: 16 steps
                       [False] * 3)                  # Incomplete episode: 3 steps
        
        trajectories = self._create_dummy_trajectories(done_pattern)
        episodes = group_trajectories_by_episode(trajectories)
        
        self.assertEqual(len(episodes), 3)  # 3 complete episodes
        self.assertEqual(len(episodes[0]), 11)  # First episode length
        self.assertEqual(len(episodes[1]), 6)   # Second episode length  
        self.assertEqual(len(episodes[2]), 16)  # Third episode length
        
        # Verify terminal flags
        self.assertTrue(episodes[0][-1][3])   # Last step of first episode
        self.assertTrue(episodes[1][-1][3])   # Last step of second episode
        self.assertTrue(episodes[2][-1][3])   # Last step of third episode
        
        # Verify non-terminal flags
        for episode in episodes:
            for step in episode[:-1]:  # All but last step
                self.assertFalse(step[3])
    
    def test_scalar_vs_tensor_done_values(self):
        """Test that done values work correctly whether scalar or tensor."""
        done_pattern = [False, True]
        trajectories = self._create_dummy_trajectories(done_pattern)
        
        # The function should handle .item() call on tensor boolean values
        episodes = group_trajectories_by_episode(trajectories)
        
        self.assertEqual(len(episodes), 1)
        self.assertEqual(len(episodes[0]), 2)
        self.assertFalse(episodes[0][0][3])  # First step: not done
        self.assertTrue(episodes[0][1][3])   # Second step: done
    
    def test_wrong_trajectory_format_handling(self):
        """Test behavior with malformed input (for robustness)."""
        # Test with non-tensor done values (should raise AttributeError on .item())
        states = torch.randn(2, 4)
        actions = torch.tensor([0, 1])
        rewards = torch.tensor([1.0, 2.0])
        dones = [False, True]  # List instead of tensor
        
        trajectories = (states, actions, rewards, dones)
        
        with self.assertRaises(AttributeError):
            group_trajectories_by_episode(trajectories)


# ---------------------------------------------------------------------- #
#  Tests for build_env function                                           #
# ---------------------------------------------------------------------- #
class TestBuildEnv(unittest.TestCase):
    """Test cases for the build_env function."""

    @patch('stable_baselines3.common.env_util.make_vec_env')
    def test_build_env_single_env_default_params(self, mock_make_vec_env):
        """Test build_env with default parameters (single environment)."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        # Mock the return value
        mock_env = MagicMock()
        mock_make_vec_env.return_value = mock_env
        
        # Call build_env with minimal parameters
        result = build_env('CartPole-v1')
        
        # Verify make_vec_env was called with correct parameters
        mock_make_vec_env.assert_called_once_with(
            'CartPole-v1', 
            n_envs=1, 
            seed=None, 
            vec_env_cls=DummyVecEnv
        )
        
        # Verify the result is the mock environment (no normalization)
        self.assertEqual(result, mock_env)

    @patch('stable_baselines3.common.env_util.make_vec_env')
    def test_build_env_multiple_envs(self, mock_make_vec_env):
        """Test build_env with multiple environments (should use SubprocVecEnv)."""
        from stable_baselines3.common.vec_env import SubprocVecEnv
        
        mock_env = MagicMock()
        mock_make_vec_env.return_value = mock_env
        
        # Call with multiple environments
        result = build_env('CartPole-v1', n_envs=4)
        
        # Should use SubprocVecEnv for multiple environments
        mock_make_vec_env.assert_called_once_with(
            'CartPole-v1',
            n_envs=4,
            seed=None,
            vec_env_cls=SubprocVecEnv
        )
        
        self.assertEqual(result, mock_env)

    @patch('stable_baselines3.common.env_util.make_vec_env')
    @patch('multiprocessing.cpu_count')
    def test_build_env_auto_n_envs(self, mock_cpu_count, mock_make_vec_env):
        """Test build_env with n_envs='auto' (should use CPU count)."""
        from stable_baselines3.common.vec_env import SubprocVecEnv
        
        # Mock CPU count
        mock_cpu_count.return_value = 8
        mock_env = MagicMock()
        mock_make_vec_env.return_value = mock_env
        
        # Call with auto n_envs
        result = build_env('CartPole-v1', n_envs='auto')
        
        # Should use CPU count and SubprocVecEnv
        mock_make_vec_env.assert_called_once_with(
            'CartPole-v1',
            n_envs=8,
            seed=None,
            vec_env_cls=SubprocVecEnv
        )
        
        self.assertEqual(result, mock_env)

    @patch('stable_baselines3.common.env_util.make_vec_env')
    def test_build_env_with_seed(self, mock_make_vec_env):
        """Test build_env with seed parameter."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        mock_env = MagicMock()
        mock_make_vec_env.return_value = mock_env
        
        # Call with seed
        result = build_env('CartPole-v1', seed=42)
        
        # Verify seed is passed through
        mock_make_vec_env.assert_called_once_with(
            'CartPole-v1',
            n_envs=1,
            seed=42,
            vec_env_cls=DummyVecEnv
        )
        
        self.assertEqual(result, mock_env)

    @patch('stable_baselines3.common.vec_env.VecNormalize')
    @patch('stable_baselines3.common.env_util.make_vec_env')
    def test_build_env_with_obs_normalization(self, mock_make_vec_env, mock_vec_normalize):
        """Test build_env with observation normalization."""
        mock_base_env = MagicMock()
        mock_normalized_env = MagicMock()
        mock_make_vec_env.return_value = mock_base_env
        mock_vec_normalize.return_value = mock_normalized_env
        
        # Call with observation normalization
        result = build_env('CartPole-v1', norm_obs=True)
        
        # Verify VecNormalize was called with correct parameters
        mock_vec_normalize.assert_called_once_with(
            mock_base_env,
            norm_obs=True,
            norm_reward=False
        )
        
        # Should return the normalized environment
        self.assertEqual(result, mock_normalized_env)

    @patch('stable_baselines3.common.vec_env.VecNormalize')
    @patch('stable_baselines3.common.env_util.make_vec_env')
    def test_build_env_with_reward_normalization(self, mock_make_vec_env, mock_vec_normalize):
        """Test build_env with reward normalization."""
        mock_base_env = MagicMock()
        mock_normalized_env = MagicMock()
        mock_make_vec_env.return_value = mock_base_env
        mock_vec_normalize.return_value = mock_normalized_env
        
        # Call with reward normalization
        result = build_env('CartPole-v1', norm_reward=True)
        
        # Verify VecNormalize was called with correct parameters
        mock_vec_normalize.assert_called_once_with(
            mock_base_env,
            norm_obs=False,
            norm_reward=True
        )
        
        # Should return the normalized environment
        self.assertEqual(result, mock_normalized_env)

    @patch('stable_baselines3.common.vec_env.VecNormalize')
    @patch('stable_baselines3.common.env_util.make_vec_env')
    def test_build_env_with_both_normalizations(self, mock_make_vec_env, mock_vec_normalize):
        """Test build_env with both observation and reward normalization."""
        mock_base_env = MagicMock()
        mock_normalized_env = MagicMock()
        mock_make_vec_env.return_value = mock_base_env
        mock_vec_normalize.return_value = mock_normalized_env
        
        # Call with both normalizations
        result = build_env('CartPole-v1', norm_obs=True, norm_reward=True)
        
        # Verify VecNormalize was called with both flags
        mock_vec_normalize.assert_called_once_with(
            mock_base_env,
            norm_obs=True,
            norm_reward=True
        )
        
        # Should return the normalized environment
        self.assertEqual(result, mock_normalized_env)

    @patch('stable_baselines3.common.env_util.make_vec_env')
    def test_build_env_no_normalization(self, mock_make_vec_env):
        """Test build_env without normalization (norm_obs=False, norm_reward=False)."""
        mock_env = MagicMock()
        mock_make_vec_env.return_value = mock_env
        
        # Call without normalization (explicit False)
        result = build_env('CartPole-v1', norm_obs=False, norm_reward=False)
        
        # Should return the original environment without wrapping
        self.assertEqual(result, mock_env)

    @patch('stable_baselines3.common.env_util.make_vec_env')
    def test_build_env_single_env_boundary(self, mock_make_vec_env):
        """Test build_env with n_envs=1 (boundary case for DummyVecEnv)."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        mock_env = MagicMock()
        mock_make_vec_env.return_value = mock_env
        
        # Call with exactly 1 environment
        result = build_env('CartPole-v1', n_envs=1)
        
        # Should use DummyVecEnv (not SubprocVecEnv)
        mock_make_vec_env.assert_called_once_with(
            'CartPole-v1',
            n_envs=1,
            seed=None,
            vec_env_cls=DummyVecEnv
        )

    @patch('stable_baselines3.common.env_util.make_vec_env')
    def test_build_env_multiple_envs_boundary(self, mock_make_vec_env):
        """Test build_env with n_envs=2 (boundary case for SubprocVecEnv)."""
        from stable_baselines3.common.vec_env import SubprocVecEnv
        
        mock_env = MagicMock()
        mock_make_vec_env.return_value = mock_env
        
        # Call with 2 environments (> 1)
        result = build_env('CartPole-v1', n_envs=2)
        
        # Should use SubprocVecEnv
        mock_make_vec_env.assert_called_once_with(
            'CartPole-v1',
            n_envs=2,
            seed=None,
            vec_env_cls=SubprocVecEnv
        )

    @patch('stable_baselines3.common.vec_env.VecNormalize')
    @patch('stable_baselines3.common.env_util.make_vec_env')
    def test_build_env_all_parameters(self, mock_make_vec_env, mock_vec_normalize):
        """Test build_env with all parameters specified."""
        from stable_baselines3.common.vec_env import SubprocVecEnv
        
        mock_base_env = MagicMock()
        mock_normalized_env = MagicMock()
        mock_make_vec_env.return_value = mock_base_env
        mock_vec_normalize.return_value = mock_normalized_env
        
        # Call with all parameters
        result = build_env(
            env_id='Pendulum-v1',
            n_envs=4,
            seed=123,
            norm_obs=True,
            norm_reward=True
        )
        
        # Verify make_vec_env call
        mock_make_vec_env.assert_called_once_with(
            'Pendulum-v1',
            n_envs=4,
            seed=123,
            vec_env_cls=SubprocVecEnv
        )
        
        # Verify VecNormalize call
        mock_vec_normalize.assert_called_once_with(
            mock_base_env,
            norm_obs=True,
            norm_reward=True
        )
        
        # Should return normalized environment
        self.assertEqual(result, mock_normalized_env)

    def test_build_env_env_id_parameter_types(self):
        """Test build_env with different env_id parameter types."""
        with patch('stable_baselines3.common.env_util.make_vec_env') as mock_make_vec_env:
            mock_env = MagicMock()
            mock_make_vec_env.return_value = mock_env
            
            # Test with string
            build_env('CartPole-v1')
            mock_make_vec_env.assert_called_with(
                'CartPole-v1',
                n_envs=1,
                seed=None,
                vec_env_cls=unittest.mock.ANY
            )
            
            # Reset mock
            mock_make_vec_env.reset_mock()
            
            # Test with different string
            build_env('LunarLander-v2')
            mock_make_vec_env.assert_called_with(
                'LunarLander-v2',
                n_envs=1,
                seed=None,
                vec_env_cls=unittest.mock.ANY
            )

    @patch('stable_baselines3.common.env_util.make_vec_env')
    def test_build_env_seed_parameter_types(self, mock_make_vec_env):
        """Test build_env with different seed parameter types."""
        mock_env = MagicMock()
        mock_make_vec_env.return_value = mock_env
        
        # Test with integer seed
        build_env('CartPole-v1', seed=42)
        mock_make_vec_env.assert_called_with(
            'CartPole-v1',
            n_envs=1,
            seed=42,
            vec_env_cls=unittest.mock.ANY
        )
        
        # Reset mock
        mock_make_vec_env.reset_mock()
        
        # Test with None seed
        build_env('CartPole-v1', seed=None)
        mock_make_vec_env.assert_called_with(
            'CartPole-v1',
            n_envs=1,
            seed=None,
            vec_env_cls=unittest.mock.ANY
        )

    def test_build_env_integration_imports(self):
        """Test that build_env imports work correctly (integration test)."""
        # This test verifies that the function can import required modules
        try:
            from stable_baselines3.common.env_util import make_vec_env
            from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")
        
        # If imports work, the function should be callable (even if it fails on environment creation)
        # We just test that it doesn't fail on import
        from tsilva_notebook_utils.gymnasium import build_env
        self.assertTrue(callable(build_env))


# ---------------------------------------------------------------------- #
#  Tests for build_env function (alias for build_env)                    #
# ---------------------------------------------------------------------- #
class TestMakeEnv(unittest.TestCase):
    """Test cases for the build_env function (which is an alias for build_env)."""

    @patch('stable_baselines3.common.env_util.make_vec_env')
    def test_build_env_basic_functionality(self, mock_make_vec_env):
        """Test that build_env works as an alias for build_env."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        mock_env = MagicMock()
        mock_make_vec_env.return_value = mock_env
        
        # Call build_env with basic parameters
        result = build_env('CartPole-v1')
        
        # Should behave exactly like build_env
        mock_make_vec_env.assert_called_once_with(
            'CartPole-v1',
            n_envs=1,
            seed=None,
            vec_env_cls=DummyVecEnv
        )
        
        self.assertEqual(result, mock_env)

    @patch('stable_baselines3.common.vec_env.VecNormalize')
    @patch('stable_baselines3.common.env_util.make_vec_env')
    def test_build_env_all_parameters(self, mock_make_vec_env, mock_vec_normalize):
        """Test build_env with all parameters (should behave like build_env)."""
        from stable_baselines3.common.vec_env import SubprocVecEnv
        
        mock_base_env = MagicMock()
        mock_normalized_env = MagicMock()
        mock_make_vec_env.return_value = mock_base_env
        mock_vec_normalize.return_value = mock_normalized_env
        
        # Call build_env with all parameters
        result = build_env(
            env_id='Pendulum-v1',
            n_envs=4,
            seed=123,
            norm_obs=True,
            norm_reward=True
        )
        
        # Should call make_vec_env with correct parameters
        mock_make_vec_env.assert_called_once_with(
            'Pendulum-v1',
            n_envs=4,
            seed=123,
            vec_env_cls=SubprocVecEnv
        )
        
        # Should also apply VecNormalize
        mock_vec_normalize.assert_called_once_with(
            mock_base_env,
            norm_obs=True,
            norm_reward=True
        )
        
        self.assertEqual(result, mock_normalized_env)

    @patch('stable_baselines3.common.vec_env.VecNormalize')
    @patch('stable_baselines3.common.env_util.make_vec_env')
    def test_build_env_with_normalization(self, mock_make_vec_env, mock_vec_normalize):
        """Test build_env with normalization (should behave like build_env)."""
        mock_base_env = MagicMock()
        mock_normalized_env = MagicMock()
        mock_make_vec_env.return_value = mock_base_env
        mock_vec_normalize.return_value = mock_normalized_env
        
        # Call build_env with normalization
        result = build_env('CartPole-v1', norm_obs=True, norm_reward=True)
        
        # Should apply VecNormalize wrapper
        mock_vec_normalize.assert_called_once_with(
            mock_base_env,
            norm_obs=True,
            norm_reward=True
        )
        
        self.assertEqual(result, mock_normalized_env)

    def test_build_env_signature_compatibility(self):
        """Test that build_env has the same signature as build_env."""
        import inspect
        
        # Get function signatures
        build_env_sig = inspect.signature(build_env)
        build_env_sig = inspect.signature(build_env)
        
        # Parameters should be identical
        self.assertEqual(
            list(build_env_sig.parameters.keys()),
            list(build_env_sig.parameters.keys())
        )
        
        # Default values should be identical
        for param_name in build_env_sig.parameters:
            build_env_param = build_env_sig.parameters[param_name]
            build_env_param = build_env_sig.parameters[param_name]
            self.assertEqual(build_env_param.default, build_env_param.default)

    @patch('tsilva_notebook_utils.gymnasium.build_env')
    def test_build_env_calls_build_env(self, mock_build_env):
        """Test that build_env actually calls build_env internally."""
        mock_env = MagicMock()
        mock_build_env.return_value = mock_env
        
        # Call build_env
        result = build_env('CartPole-v1', n_envs=2, seed=42, norm_obs=True)
        
        # Should call build_env with same parameters
        mock_build_env.assert_called_once_with(
            env_id='CartPole-v1',
            n_envs=2,
            seed=42,
            norm_obs=True,
            norm_reward=False
        )
        
        self.assertEqual(result, mock_env)

    def test_build_env_exists_and_callable(self):
        """Test that build_env function exists and is callable."""
        from tsilva_notebook_utils.gymnasium import build_env
        self.assertTrue(callable(build_env))

    def test_build_env_docstring(self):
        """Test that build_env has proper documentation."""
        from tsilva_notebook_utils.gymnasium import build_env
        self.assertIsNotNone(build_env.__doc__)
        self.assertIn("alias for build_env", build_env.__doc__)


if __name__ == "__main__":
    unittest.main()