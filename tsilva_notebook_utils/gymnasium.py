def build_env(
    env_id, 
    seed=None, 
    normalize_observation=False,
    env_kwargs={}
):
    import gymnasium as gym
    from gymnasium.wrappers import NormalizeObservation

    env = gym.make(env_id, **env_kwargs)

    if normalize_observation: 
        env = NormalizeObservation(env)

    state, info = env.reset(seed=seed)
    if seed is not None: 
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
    return env, state, info


def run_episode(env_id, model, seed=None):
    import torch
    from .torch import get_module_device

    device = get_module_device(model)

    env, state, _ = build_env(env_id, seed=seed, env_kwargs=dict(render_mode="rgb_array"))
    frames = []
    done = False
    total_reward = 0.0
    while not done:
        frames.append(env.render())
        state_t = torch.as_tensor(state, device=device).float().unsqueeze(0)
        with torch.no_grad(): action = model(state_t).argmax(dim=1).item()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    env.close()
    return frames, total_reward


def record_episode(env_id, model, seed=None, fps=30):
    import tempfile
    import imageio
    import numpy as np

    frames, _ = run_episode(env_id, model, seed=seed)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        imageio.mimsave(tmp.name, [np.array(f) for f in frames], macro_block_size=1, fps=fps)
        return tmp.name


def render_episode(env_id, model, seed=None, fps=30):
    from IPython.display import Video
    video_path = record_episode(env_id, model, seed=seed, fps=fps)
    return Video(video_path, embed=True)


def build_pl_callback(callback_id, *args, **kwargs):
    import pytorch_lightning as pl

    class EvalEpisodeAndRecordCallback(pl.Callback):
        def __init__(self, every_n_episodes=10, fps=30, seed=None):
            self.every_n_episodes = every_n_episodes
            self.fps = fps
            self.seed = seed
            super().__init__()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            import wandb
            from tsilva_notebook_utils.gymnasium import record_episode

            if getattr(pl_module, "episode_steps") != 0: return
            episode = getattr(pl_module, "episode")
            if episode == 0 or episode % self.every_n_episodes: return

            video_path = record_episode(
                env_id=pl_module.env_id,
                model=pl_module.q_model,
                seed=self.seed,
                fps=self.fps
            )
            logger_run = trainer.logger.experiment
            logger_run.log(
                {"eval/video": wandb.Video(video_path, format="mp4")},
                step=trainer.global_step
            )
            
    return {
        "EvalEpisodeAndRecordCallback": EvalEpisodeAndRecordCallback
    }[callback_id](*args, **kwargs)
