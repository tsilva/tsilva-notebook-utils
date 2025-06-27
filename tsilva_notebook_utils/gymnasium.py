from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
from IPython.display import HTML
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.distributions import Categorical


def run_episode(env, model, seed=None):
    import torch
    from .torch import get_module_device

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
    import tempfile
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


def collect_rollouts(
    env,
    policy_model: torch.nn.Module,
    value_model: Optional[torch.nn.Module] = None,
    *,
    n_episodes: int = 1,
    deterministic: bool = False,
    #render: bool = False,
    collect_frames: bool = False,
    compute_advantages: bool = True,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[
    List[float],                                   # total episode returns
    Tuple[List, List, List, List, List, List, List],  # trajectory buffers (+ advantages)
    List[List[np.ndarray]] | None,                 # optional frames by episode
]:
    """Light-weight rollout collector that works with Gymnasium *and* SB3
    vector environments.

    Enhancements over the original version
    --------------------------------------
    1. **Optional value function** – ``value_model`` can be *None*. When
       omitted, the value estimates default to *zero*, and advantages
       become equal to discounted returns.
    2. **GAE computation** – When *compute_advantages* is ``True`` the
       routine computes Generalised Advantage Estimates (GAE-λ).

    Parameters
    ----------
    env : Union[gym.Env, VecEnv]
        Environment—either a single Gymnasium env or a vectorised env
        (e.g. SB3 ``VecEnv``).
    policy_model : torch.nn.Module
        Model taking a batch of observations and returning *logits*.
    value_model : Optional[torch.nn.Module], default=None
        Model returning state-values. If ``None`` no value prediction is
        performed; advantages fall back to Monte-Carlo returns.
    n_episodes : int, default=1  (must be ≥ 1)
        Number of **complete** episodes to collect across *all* workers.
    deterministic : bool, default=False
        If *True*, take the **argmax** action instead of sampling.
    render : bool, default=False
        Call ``env.render()`` after every step.
    collect_frames : bool, default=False
        Save raw frames via ``env.get_images()``.
    gamma : float, default=0.99
        Discount factor for returns / GAE.
    gae_lambda : float, default=0.95
        λ parameter for GAE.
    compute_advantages : bool, default=True
        Toggle the (potentially expensive) GAE computation.

    Returns
    -------
    total_ep_returns : List[float]
        One scalar per episode.
    trajectories : Tuple[List, …]
        ``(obs, actions, rewards, dones, logps, values, advantages)`` –
        flattened across workers and time.
    frames_by_ep : Optional[List[List[np.ndarray]]]
        Raw RGB frames grouped by episode when *collect_frames* is *True*.
    """
    if n_episodes < 1:
        raise ValueError("n_episodes must be ≥ 1")

    # Local helper ------------------------------------------------------
    def _get_device(module: torch.nn.Module) -> torch.device:  # type: ignore
        """Return the device of *module*'s first parameter."""
        return next(module.parameters()).device

    device   = _get_device(policy_model)
    obs      = env.reset()
    n_envs   = env.num_envs

    # Per-env episode buffers ------------------------------------------
    ep_obs, ep_actions, ep_rewards = [[] for _ in range(n_envs)], [[] for _ in range(n_envs)], [[] for _ in range(n_envs)]
    ep_dones, ep_logps             = [[] for _ in range(n_envs)], [[] for _ in range(n_envs)]
    ep_values                      = [[] for _ in range(n_envs)] if value_model or compute_advantages else None
    ep_frames                      = [[] for _ in range(n_envs)] if collect_frames else None

    # Global rollout buffers -------------------------------------------
    obs_buf, act_buf, rew_buf, done_buf, logp_buf, val_buf, adv_buf = ([] for _ in range(7))
    frames_buf: List[List[np.ndarray]] = [] if collect_frames else None
    total_ep_returns: List[float] = []
    episodes_collected            = 0

    with torch.no_grad():
        while episodes_collected < n_episodes:
            # ------------------------------------------------ policy forward
            obs_t  = torch.as_tensor(obs, dtype=torch.float32, device=device)
            logits = policy_model(obs_t)
            dist   = Categorical(logits=logits)

            action_tensor = (
                dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            )                           # shape (n_envs,)
            logp_tensor   = dist.log_prob(action_tensor)

            action_np = np.atleast_1d(action_tensor.cpu().numpy())
            logp_np   = np.atleast_1d(logp_tensor.cpu().numpy())

            # value prediction is optional -----------------------------
            if value_model is not None:
                value_np = np.atleast_1d(value_model(obs_t).squeeze(-1).cpu().numpy())
            else:
                value_np = np.zeros(n_envs, dtype=np.float32)

            # ------------------------------------------------ env step
            step_out = env.step(action_np)
            if len(step_out) == 4:
                next_obs, reward, done, _ = step_out
            else:
                next_obs, reward, done, trunc, _ = step_out   # Gymnasium
                done = np.logical_or(done, trunc)

            reward = np.atleast_1d(reward)        # ensure 1-D

            # -------- explicit reset for Gymnasium VecEnv -------------
            if done.any() and hasattr(env, "reset_done"):
                reset_obs, *_ = env.reset_done()
                next_obs[done] = reset_obs
            # ----------------------------------------------------------

            # ------------------------------------------------ frame capture
            if collect_frames:
                frame_batch = env.get_images()
                if len(frame_batch) != n_envs:
                    raise RuntimeError("get_images() must return a list of length n_envs")
                for i in range(n_envs):
                    ep_frames[i].append(frame_batch[i])  # type: ignore[arg-type]

            # ------------------------------------------------ store transition
            for i in range(n_envs):
                if episodes_collected >= n_episodes:
                    break  # quota reached

                ep_obs[i].append(obs[i])
                ep_actions[i].append(int(action_np[i]))
                ep_rewards[i].append(float(reward[i]))
                ep_dones[i].append(bool(done[i]))
                ep_logps[i].append(float(logp_np[i]))
                if ep_values is not None:
                    ep_values[i].append(float(value_np[i]))

                if done[i]:
                    # ------- episode i terminates --------------------
                    total_ep_returns.append(float(np.sum(ep_rewards[i])))

                    # Compute advantages / returns --------------------
                    if compute_advantages:
                        t_rewards = ep_rewards[i]
                        t_values  = ep_values[i] if ep_values is not None else [0.0] * len(t_rewards)
                        T = len(t_rewards)
                        adv = [0.0] * T
                        ret = [0.0] * T
                        last_gae = 0.0
                        next_value = 0.0  # because episode ended
                        for t in reversed(range(T)):
                            delta = t_rewards[t] + gamma * next_value - t_values[t]
                            last_gae = delta + gamma * gae_lambda * last_gae
                            adv[t] = last_gae
                            next_value = t_values[t]
                        # returns-to-go (Monte Carlo)
                        running_ret = 0.0
                        for t in reversed(range(T)):
                            running_ret = t_rewards[t] + gamma * running_ret
                            ret[t] = running_ret
                    else:
                        adv = [0.0] * len(ep_rewards[i])
                        ret = None  # unused

                    # ------- flush episode i -------------------------
                    obs_buf.extend(ep_obs[i]);       ep_obs[i].clear()
                    act_buf.extend(ep_actions[i]);   ep_actions[i].clear()
                    rew_buf.extend(ep_rewards[i]);   ep_rewards[i].clear()
                    done_buf.extend(ep_dones[i]);    ep_dones[i].clear()
                    logp_buf.extend(ep_logps[i]);    ep_logps[i].clear()
                    if ep_values is not None:
                        val_buf.extend(ep_values[i]); ep_values[i].clear()
                    else:
                        val_buf.extend([0.0] * len(adv))
                    adv_buf.extend(adv)

                    if collect_frames:
                        frames_buf.append(ep_frames[i].copy())  # type: ignore[arg-type]
                        ep_frames[i].clear()  # type: ignore[union-attr]

                    episodes_collected += 1

            obs = next_obs  # iterate

    # ------------------------------ return buffers --------------------
    traj = (obs_buf, act_buf, rew_buf, done_buf, logp_buf, val_buf, adv_buf)
    if collect_frames:
        return total_ep_returns, traj, frames_buf  # type: ignore[return-value]
    return total_ep_returns, traj  # type: ignore[return-value]


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
    import multiprocessing
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    if n_envs == 'auto': n_envs = multiprocessing.cpu_count()
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)
    if norm_obs or norm_reward: env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward)
    return env


def set_random_seed(seed):
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(seed)

