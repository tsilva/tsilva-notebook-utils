from __future__ import annotations

import torch
import numpy as np
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    from torch.utils.data import Dataset as TorchDataset
except Exception:
    TorchDataset = object  # type: ignore



from .torch import get_module_device


def run_episode(env, model, seed=None):

    import torch


    device = get_module_device(model)

    state, _ = env.reset(seed=seed)
    if seed is not None: 
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

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


def record_episode(env, model, seed=None, fps=30):

    import imageio
    import numpy as np


    if callable(env): env, _, _ = env(env_kwargs=dict(render_mode="rgb_array"))

    frames, _ = run_episode(env, model, seed=seed)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        imageio.mimsave(tmp.name, [np.array(f) for f in frames], macro_block_size=1, fps=fps)
        return tmp.name


def render_episode(env, model, seed=None, fps=30):
    from IPython.display import Video
    video_path = record_episode(env, model, seed=seed, fps=fps)
    return Video(video_path, embed=True)


def build_pl_callback(callback_id, *args, **kwargs):
    import pytorch_lightning as pl
    import wandb
    class EvalEpisodeAndRecordCallback(pl.Callback):
        def __init__(self, every_n_episodes=10, fps=30, seed=None):
            self.every_n_episodes = every_n_episodes
            self.fps = fps
            self.seed = seed
            super().__init__()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            

            if getattr(pl_module, "episode_steps") != 0: return
            episode = getattr(pl_module, "episode")
            if episode == 0 or episode % self.every_n_episodes: return

            video_path = record_episode(
                env=pl_module.build_env_fn,
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


def render_episode_frames(
    frames: Iterable[np.ndarray] | Iterable[Iterable[np.ndarray]],
    *,
    fps: int = 30,
    out_path: str | os.PathLike | None = None,
    out_dir: str | os.PathLike | None = None,
    codec: str = "libx264",
    crf: int = 23,
    preset: str = "medium",
    # ---------- label styling ----------------------------------------
    font: ImageFont.ImageFont | None = None,
    text_xy: Tuple[int, int] = (5, 5),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    stroke_color: Tuple[int, int, int] = (0, 0, 0),
    stroke_width: int = 1,
    # ---------- outer grid size --------------------------------------
    grid: Tuple[int, int] = (1, 1),
    # ---------- embed options ----------------------------------------
    width: int = 640,
) -> HTML:
    """
    Encode *frames* (flat list) or *episodes* (list of lists) into an MP4
    and return an embeddable HTML snippet.

    Notes
    -----
    * If ``out_dir`` is given and **lies outside** the current notebook
      directory, the video is silently **copied back** next to the
      notebook so the browser can reach it.
    """

    import itertools
    import numpy as np
    from IPython.display import HTML
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v3 as iio

    # ---------------- font --------------------------------------------------
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", size=24)
        except OSError:
            font = ImageFont.load_default()

    # ---------------- flat vs. nested detection ----------------------------
    fr_iter = iter(frames)
    try:
        first_item = next(fr_iter)
    except StopIteration as e:
        raise ValueError("No frames provided.") from e
    fr_iter = itertools.chain([first_item], fr_iter)

    nested = isinstance(first_item, (Sequence, Iterable)) and not (
        isinstance(first_item, np.ndarray) and first_item.ndim == 3
    )

    episodes = (
        [list(ep) for ep in fr_iter] if nested else [list(fr_iter)]
    )
    if not episodes or not episodes[0]:
        raise ValueError("No frames provided.")

    # ---------------- validate frames --------------------------------------
    ref_H, ref_W, _ = episodes[0][0].shape
    if not (
        isinstance(episodes[0][0], np.ndarray)
        and episodes[0][0].dtype == np.uint8
        and episodes[0][0].ndim == 3
    ):
        raise ValueError("Frames must be uint8 RGB arrays of shape (H,W,3).")

    for ep in episodes:
        for fr in ep:
            if fr.shape != (ref_H, ref_W, 3):
                raise ValueError("All frames must share the same (H,W,3).")

    # ---------------- outer grid geometry ----------------------------------
    rows, cols = grid
    if rows <= 0 or cols <= 0:
        raise ValueError("grid must contain positive integers.")
    n_cells = rows * cols

    # partition episodes -> cells, round-robin
    cell_to_eps = {c: [] for c in range(n_cells)}
    for ep_idx, cell in enumerate(
        itertools.islice(itertools.cycle(range(n_cells)), len(episodes))
    ):
        cell_to_eps[cell].append(ep_idx)

    # build per-cell flat playback sequence [(ep, step, frame), ...]
    cell_seq: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    for cell, eps in cell_to_eps.items():
        seq = []
        for ep_idx in eps:
            for st_idx, fr in enumerate(episodes[ep_idx]):
                seq.append((ep_idx, st_idx, fr))
        if not seq:  # empty slot → black frame
            black = np.zeros_like(episodes[0][0])
            seq = [(-1, -1, black)]
        cell_seq[cell] = seq

    max_len = max(len(s) for s in cell_seq.values())

    # ---------------- helper: stamp label ----------------------------------
    def stamp(img: Image.Image, label: str):
        draw = ImageDraw.Draw(img, "RGB")
        draw.text(
            text_xy,
            label,
            font=font,
            fill=text_color,
            stroke_fill=stroke_color,
            stroke_width=stroke_width,
        )

    # ---------------- temp dir for PNGs ------------------------------------
    with tempfile.TemporaryDirectory() as tmp_root:
        png_dir = Path(tmp_root) / "frames"
        png_dir.mkdir()

        frame_id = 0
        for t in range(max_len):
            canvas = np.zeros((rows * ref_H, cols * ref_W, 3), dtype=np.uint8)
            for r in range(rows):
                for c in range(cols):
                    cell = r * cols + c
                    seq = cell_seq[cell]
                    ep_idx, st_idx, frame = (
                        seq[t] if t < len(seq) else seq[-1]
                    )
                    img = Image.fromarray(frame)
                    if ep_idx >= 0:
                        stamp(img, f"Episode: {ep_idx + 1}  Step: {st_idx + 1}")
                    y0, x0 = r * ref_H, c * ref_W
                    canvas[y0 : y0 + ref_H, x0 : x0 + ref_W] = np.asarray(img)

            iio.imwrite(
                png_dir / f"frame_{frame_id:06d}.png",
                canvas,
                plugin="pillow",
            )
            frame_id += 1

        # ---------------- encode with FFmpeg -------------------------------
        if out_path is None:
            fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            out_path_ = Path(tmp_name)
        else:
            out_path_ = Path(out_path)

        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(png_dir / "frame_%06d.png"),
            "-c:v",
            codec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(out_path_),
        ]
        subprocess.run(cmd, check=True)

    # ---------------- move / copy video for the notebook -------------------
    if out_dir is None:
        out_dir = Path.cwd()
    else:
        out_dir = Path(out_dir)

    dest = out_dir / f"video_{uuid.uuid4().hex}.mp4"
    if out_path_ != dest:
        shutil.move(out_path_, dest)

    # Ensure the browser can reach the file
    nb_root = Path.cwd().resolve()
    try:
        # Jupyter can serve anything under the notebook root
        src_path = dest.resolve().relative_to(nb_root).as_posix()
    except ValueError:
        # File lies outside → copy it next to the notebook
        safe_dest = nb_root / dest.name
        shutil.copy(dest, safe_dest)
        src_path = safe_dest.name

    # ---------------- build HTML snippet -----------------------------------
    vid_id = f"vid_{uuid.uuid4().hex}"
    html = f"""
<div style="max-width:{width}px">
    <video id="{vid_id}" src="{src_path}" width="100%" controls playsinline
            style="background:#000"></video>

    <div style="font-size:0.9em;margin-top:4px">
    <a href="{src_path}" target="_blank">🔗 open in new tab (fullscreen ✓)</a>
    &nbsp;•&nbsp; double-click video to toggle fullscreen
    </div>
</div>

<script>
const v = document.getElementById('{vid_id}');
v.addEventListener('dblclick', () => {{
    if (!document.fullscreenElement) {{
        (v.requestFullscreen || v.webkitRequestFullscreen ||
            v.msRequestFullscreen).call(v);
    }} else {{
        (document.exitFullscreen || document.webkitExitFullscreen ||
            document.msExitFullscreen).call(document);
    }}
}});
</script>
""".strip()
    
    return HTML(html)

def build_env(env_id, n_envs=1, seed=None, norm_obs=False, norm_reward=False):

    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    
    if n_envs == 'auto': n_envs = multiprocessing.cpu_count()
    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)
    if norm_obs or norm_reward: env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward)
    return env

def set_random_seed(seed):
    from stable_baselines3.common.utils import set_random_seed as _set_random_seed

    _set_random_seed(seed)


def log_env_info(env) -> None:
    """Print key attributes of an environment or a vec-env.

    Handles:
      • Plain or wrapped Gym/Gymnasium envs
      • DummyVecEnv  (stores sub-envs locally)
      • SubprocVecEnv (sub-envs live in worker processes, accessed via RPC)

    Fields printed:
      Env ID, observation/action space, (reward range if available),
      max episode steps.
    """
    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    # ─────────────────────────────── helpers ──────────────────────────────────────
    def _compact(arr: np.ndarray) -> str:
        """Return a 1-D array as '[a, b, c]' with exactly one space after commas."""
        def fmt(x):
            if np.isposinf(x):
                return "inf"
            if np.isneginf(x):
                return "-inf"
            return f"{x:.3g}"          # 3 significant digits, no extra padding
        return "[" + ", ".join(fmt(v) for v in arr.ravel()) + "]"


    def _fmt_space(space) -> str:
        """Pretty-print Box / Discrete / etc. with compact low/high arrays."""
        if hasattr(space, "low") and hasattr(space, "high"):
            low  = _compact(space.low)
            high = _compact(space.high)
            return (f"{space.__class__.__name__}"
                    f"(low={low}, high={high}, shape={space.shape}, dtype={space.dtype})")
        return str(space)

    # 1) Detect vec-env type and pick a sub-env handle when possible
    if isinstance(env, DummyVecEnv):
        vec_kind, n = "DummyVecEnv", len(env.envs)
        base_env    = env.envs[0]                  # local object
    elif isinstance(env, SubprocVecEnv):
        vec_kind, n = "SubprocVecEnv", env.num_envs
        base_env    = None                         # must query via RPC
    else:
        vec_kind, n = None, 1                      # single or wrapped env
        base_env    = env

    # 2) Safe remote attribute fetcher for SubprocVecEnv
    def remote_attr(name):
        if not isinstance(env, SubprocVecEnv):
            return None
        try:
            return env.get_attr(name, indices=0)[0]
        except Exception:
            return None                            # Attribute missing ➞ None

    # 3) Gather ID, max-steps, reward range
    if base_env is not None:                       # DummyVecEnv or plain env
        spec       = getattr(base_env, "spec", None)
        env_id     = getattr(spec, "id", None) or getattr(base_env, "id", "Unknown")
        max_steps  = getattr(spec, "max_episode_steps", None) \
                     or getattr(base_env, "_max_episode_steps", "Unknown")
        reward_rng = getattr(base_env, "reward_range", None)  # Gym only
    else:                                          # SubprocVecEnv
        spec       = remote_attr("spec")
        env_id     = getattr(spec, "id", None) or remote_attr("id") or "Unknown"
        max_steps  = getattr(spec, "max_episode_steps", None) \
                     or remote_attr("_max_episode_steps") or "Unknown"
        reward_rng = None                          # don’t fetch reward_range remotely

    # 4) Observation / action spaces are exposed on the vec-env itself
    obs_space = env.observation_space
    act_space = env.action_space

    # 5) Print results
    header = f"Environment Info ({vec_kind} with {n} envs)" if vec_kind else "Environment Info"
    print(header)
    print(f"  Env ID: {env_id}")
    print(f"  Observation space: {_fmt_space(obs_space)}")
    print(f"  Action space: {_fmt_space(act_space)}")
    if reward_rng is not None:
        print(f"  Reward range: {reward_rng}")
    print(f"  Max episode steps: {max_steps}")


def collect_rollouts(
    env,
    policy_model: torch.nn.Module,
    value_model: Optional[torch.nn.Module] = None,
    n_steps: Optional[int] = None,
    n_episodes: Optional[int] = None,
    *,
    deterministic: bool = False,
    gamma: float = 0.99,
    lam: float = 0.95,
    normalize_advantage: bool = True,
    adv_norm_eps: float = 1e-8,
    collect_frames: bool = False,
    last_obs: Optional[np.ndarray] = None
) -> Tuple[torch.Tensor, ...]:
    """Collect transitions from *env* until *n_steps* or *n_episodes* (whichever
    comes first) are reached.

    Parameters
    ----------
    env : VecEnv-like
        Vectorised environment exposing ``reset()``, ``step(actions)`` and
        ``get_images()`` (if *collect_frames* is ``True``).
    policy_model : nn.Module
        Model that maps observations to *logits* (action probabilities).
    value_model : nn.Module | None
        Optional value network; if ``None``, value estimates are *zero*.
    n_steps : int | None, default 1024
        Maximum number of *timesteps* (across all environments) to collect.
    n_episodes : int | None, default None
        Maximum number of *episodes* (across all environments) to collect.
        Either *n_steps*, *n_episodes* or both **must** be provided.
    deterministic : bool, default False
        Whether to act greedily (``argmax``) instead of sampling.
    gamma, lam : float
        Discount and GAE-λ parameters.
    normalize_advantage : bool, default True
        Whether to standardise advantages.
    adv_norm_eps : float, default 1e-8
        Numerical stability epsilon for advantage normalisation.
    collect_frames : bool, default False
        If ``True``, return RGB frames alongside transition tensors.
    last_obs : np.ndarray | None, default None
        If provided, use these observations to continue collection without
        resetting the environment. If ``None``, reset the environment first.

    Returns
    -------
    Tuple[Tensor, ...]
        ``(states, actions, rewards, dones, logps, values, advs, returns, frames)``
        in *Stable-Baselines3*-compatible, env-major flattened order.
    """

    from torch.distributions import Categorical
    # ------------------------------------------------------------------
    # 0. Sanity checks & helpers
    # ------------------------------------------------------------------
    assert (n_steps is not None and n_steps > 0) or (
        n_episodes is not None and n_episodes > 0
    ), "Provide *n_steps*, *n_episodes*, or both (> 0)."

    def _device_of(module: torch.nn.Module) -> torch.device:
        """Infer the device of *module*'s first parameter."""
        return next(module.parameters()).device

    device: torch.device = _device_of(policy_model)
    n_envs: int = env.num_envs

    # ------------------------------------------------------------------
    # 1. Buffers (dynamic lists — we'll stack/concat later)
    # ------------------------------------------------------------------
    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    rew_buf: list[np.ndarray] = []
    done_buf: list[np.ndarray] = []
    logp_buf: list[np.ndarray] = []
    val_buf: list[np.ndarray] = []
    frame_buf: list[Sequence[np.ndarray]] | None = [] if collect_frames else None

    step_count = 0
    episode_count = 0

    # ------------------------------------------------------------------
    # 2. Rollout
    # ------------------------------------------------------------------
    obs = env.reset() if last_obs is None else last_obs
    while True:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = policy_model(obs_t)
        dist = Categorical(logits=logits)

        act_t = logits.argmax(-1) if deterministic else dist.sample()
        logp_t = dist.log_prob(act_t)
        val_t = (
            value_model(obs_t).squeeze(-1)
            if value_model is not None
            else torch.zeros(n_envs, device=device)
        )

        next_obs, reward, done, infos = env.step(act_t.cpu().numpy())

        # store step
        obs_buf.append(obs.copy())
        act_buf.append(act_t.cpu().numpy())
        rew_buf.append(reward)
        done_buf.append(done)
        logp_buf.append(logp_t.detach().cpu().numpy())
        val_buf.append(val_t.detach().cpu().numpy())

        if collect_frames and frame_buf is not None:
            frame_buf.append(env.get_images())

        step_count += 1
        episode_count += done.sum()

        # termination condition
        if (n_steps is not None and step_count >= n_steps) or (
            n_episodes is not None and episode_count >= n_episodes
        ):
            obs = next_obs  # needed for bootstrap
            break

        obs = next_obs

    T = step_count  # actual collected timesteps

    # ------------------------------------------------------------------
    # 3. Bootstrap value for the next state of each env
    # ------------------------------------------------------------------
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        next_values = (
            value_model(obs_t).squeeze(-1).cpu().numpy()
            if value_model is not None
            else np.zeros(n_envs, dtype=np.float32)
        )

    # ------------------------------------------------------------------
    # 4. Stack buffers to (T, E) arrays for GAE
    # ------------------------------------------------------------------
    act_arr = np.stack(act_buf)  # (T, E)
    rew_arr = np.stack(rew_buf)
    done_arr = np.stack(done_buf)
    logp_arr = np.stack(logp_buf)
    val_arr = np.stack(val_buf)

    # ------------------------------------------------------------------
    # 5. GAE-λ advantage / return (with masking)
    # ------------------------------------------------------------------
    adv_arr = np.zeros_like(rew_arr, dtype=np.float32)

    gae = np.zeros(n_envs, dtype=np.float32)
    next_non_terminal = 1.0 - done_arr[-1].astype(np.float32)
    next_value = next_values

    for t in reversed(range(T)):
        delta = rew_arr[t] + gamma * next_value * next_non_terminal - val_arr[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        adv_arr[t] = gae

        next_non_terminal = 1.0 - done_arr[t].astype(np.float32)
        next_value = val_arr[t]

    ret_arr = adv_arr + val_arr

    if normalize_advantage:
        adv_flat = adv_arr.reshape(-1)
        adv_arr = (adv_arr - adv_flat.mean()) / (adv_flat.std() + adv_norm_eps)

    # ------------------------------------------------------------------
    # 6. Env-major flattening: (T, E, …) -> (E, T, …) -> (E*T, …)
    # ------------------------------------------------------------------
    obs_arr = np.stack(obs_buf)  # (T, E, obs)
    obs_env_major = obs_arr.transpose(1, 0, 2)  # (E, T, obs)
    states = torch.as_tensor(obs_env_major.reshape(n_envs * T, -1), dtype=torch.float32)

    def _flat_env_major(arr: np.ndarray, dtype: torch.dtype):
        return torch.as_tensor(arr.transpose(1, 0).reshape(-1), dtype=dtype)

    actions = _flat_env_major(act_arr, torch.int64)
    rewards = _flat_env_major(rew_arr, torch.float32)
    dones = _flat_env_major(done_arr, torch.bool)
    logps = _flat_env_major(logp_arr, torch.float32)
    values = _flat_env_major(val_arr, torch.float32)
    advs = _flat_env_major(adv_arr, torch.float32)
    returns = _flat_env_major(ret_arr, torch.float32)

    if collect_frames and frame_buf is not None:
        frames_env_major: list[np.ndarray] = []
        for e in range(n_envs):
            for t in range(T):
                frames_env_major.append(frame_buf[t][e])
    else:
        frames_env_major = [0] * (n_envs * T)

    # ------------------------------------------------------------------
    # 7. Return (SB3 order)
    # ------------------------------------------------------------------
    return (
        states,
        actions,
        rewards,
        dones,
        logps,
        values,
        advs,
        returns,
        frames_env_major,
    )


def group_trajectories_by_episode(trajectories):
    episodes = []
    episode = []

    T = trajectories[0].shape[0]  # number of time steps

    for t in range(T):
        step = tuple(x[t] for x in trajectories)  # (state, action, reward, done, ...)
        episode.append(step)
        done = step[3]
        if done.item():  # convert tensor to bool
            episodes.append(episode)
            episode = []

    return episodes



# TODO: should dataloader move to gpu?
class RolloutDataset(TorchDataset):
    """Holds PPO roll-out tensors and lets them be swapped in-place."""
    def __init__(self):
        self.trajectories = None 

    def update(self, *trajectories):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories[0])

    def __getitem__(self, idx):
        return tuple(t[idx] for t in self.trajectories)
