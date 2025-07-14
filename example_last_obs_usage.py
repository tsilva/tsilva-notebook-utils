#!/usr/bin/env python3
"""
Example demonstrating the new last_obs functionality in collect_rollouts.

This shows how you can now call collect_rollouts with n_steps and keep iterating
on the same stream instead of discarding unfinished episodes.
"""

import torch
import numpy as np
from tsilva_notebook_utils.gymnasium import collect_rollouts
from tests.test_gymnasium import DummyVecEnv, ConstPolicy, ConstValue


def demonstrate_continuous_rollout_collection():
    """Demonstrate collecting rollouts in chunks while maintaining episode continuity."""
    
    print("=== Continuous Rollout Collection Demo ===")
    print()
    
    # Create environment with long episodes
    env = DummyVecEnv([20, 25])  # Episodes of length 20 and 25
    print(f"Created environment with {env.num_envs} environments")
    print(f"Episode lengths: {env.ep_lens}")
    print()
    
    # Collect rollouts in chunks
    chunk_size = 3
    n_chunks = 5
    
    print(f"Collecting rollouts in {n_chunks} chunks of {chunk_size} steps each")
    print()
    
    last_obs = None  # Start with reset
    total_transitions = 0
    
    for chunk_idx in range(n_chunks):
        print(f"--- Chunk {chunk_idx + 1} ---")
        print(f"Environment counters before: {env.counters}")
        
        # Collect rollout chunk
        results = collect_rollouts(
            env=env,
            policy_model=ConstPolicy(),
            value_model=ConstValue(),
            n_steps=chunk_size,
            last_obs=last_obs
        )
        
        states, actions, rewards, dones, logps, values, advs, returns, frames = results
        
        print(f"Collected {len(states)} transitions")
        print(f"Environment counters after: {env.counters}")
        print(f"Any episodes done: {torch.any(dones).item()}")
        
        # Update for next iteration
        last_obs = env._obs.copy()  # Continue from current state
        total_transitions += len(states)
        
        print()
    
    print(f"Total transitions collected: {total_transitions}")
    print(f"Expected transitions: {n_chunks * chunk_size * env.num_envs}")
    print(f"Final environment counters: {env.counters}")
    print()
    
    # Demonstrate the difference with resetting behavior
    print("=== Comparison: With Reset (Old Behavior) ===")
    
    # Reset environment
    env = DummyVecEnv([20, 25])
    total_with_reset = 0
    
    for chunk_idx in range(n_chunks):
        print(f"Chunk {chunk_idx + 1} - counters before: {env.counters}")
        
        # Always reset (old behavior)
        results = collect_rollouts(
            env=env,
            policy_model=ConstPolicy(),
            value_model=ConstValue(),
            n_steps=chunk_size,
            last_obs=None  # This causes reset
        )
        
        states, actions, rewards, dones, *_ = results
        total_with_reset += len(states)
        
        print(f"Collected {len(states)} transitions - counters after: {env.counters}")
    
    print(f"Total with reset: {total_with_reset}")
    print()
    print("Notice how with reset, the environment counters restart each time,")
    print("losing the continuity of episode progress.")


if __name__ == "__main__":
    demonstrate_continuous_rollout_collection()
