# tests/test_collect_rollouts.py
import unittest
import numpy as np
import torch

import pytest
from typing import Optional
from tsilva_notebook_utils.gymnasium import collect_rollouts        # ← adjust to your package


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

if __name__ == "__main__":
    unittest.main()