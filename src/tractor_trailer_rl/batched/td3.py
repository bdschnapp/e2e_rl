"""Compact GPU-resident TD3 for the batched env.

SB3's TD3 drives one (CPU-vectorised) env per process and cannot consume a single
batched-GPU env, so the CuPy port needs its own trainer. This is a minimal but
faithful TD3 (twin critics, delayed actor updates, target-policy smoothing,
clipped double-Q) that keeps the replay buffer and networks on the GPU and bridges
to the CuPy env via DLPack (zero-copy) so no per-step host transfer occurs.

It is intentionally small (~1 file). The goal is an end-to-end on-GPU training loop
that (a) proves the batched env trains a policy and (b) is the vehicle for the
multi-seed ablation reruns. Observation/action normalisation matches the scalar
setup's spaces; reward scaling is left to the caller.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backend import get_backend


def _to_torch(arr, device):
    """Zero-copy (cupy) / cheap (numpy) bridge from a backend array to torch."""
    if get_backend() == "cupy":
        return torch.from_dlpack(arr.toDlpack() if hasattr(arr, "toDlpack") else arr)
    return torch.as_tensor(np.asarray(arr), device=device)


def _to_backend(t):
    """torch tensor -> active backend array (cupy reads torch's CUDA array iface)."""
    if get_backend() == "cupy":
        import cupy as cp
        return cp.asarray(t.detach())
    return t.detach().cpu().numpy()


class RunningMeanStd:
    """Welford running mean/var of observations (the VecNormalize equivalent).

    TD3 is sensitive to input scale and the raw observation spans very different
    magnitudes (cross-track error up to ~100, curvature ~0.3, lidar 0-1). The SB3
    baseline handled this with VecNormalize; this reproduces it on the GPU.
    """

    def __init__(self, dim, device, clip=10.0, eps=1e-8):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = eps
        self.clip = clip
        self.eps = eps

    def update(self, x):
        bmean = x.mean(0); bvar = x.var(0, unbiased=False); bn = x.shape[0]
        delta = bmean - self.mean
        tot = self.count + bn
        self.mean = self.mean + delta * bn / tot
        m_a = self.var * self.count
        m_b = bvar * bn
        self.var = (m_a + m_b + delta ** 2 * self.count * bn / tot) / tot
        self.count = tot

    def norm(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + self.eps),
                           -self.clip, self.clip)


def mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]),
                   act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden=(400, 300)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim], out_act=nn.Tanh)
        self.register_buffer("act_limit", torch.as_tensor(act_limit, dtype=torch.float32))

    def forward(self, o):
        return self.net(o) * self.act_limit


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(400, 300)):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, o, a):
        return self.q(torch.cat([o, a], dim=-1)).squeeze(-1)


class ReplayBuffer:
    """Preallocated GPU ring buffer."""

    def __init__(self, cap, obs_dim, act_dim, device):
        self.cap = cap; self.ptr = 0; self.full = False; self.device = device
        self.o = torch.zeros((cap, obs_dim), device=device)
        self.a = torch.zeros((cap, act_dim), device=device)
        self.r = torch.zeros(cap, device=device)
        self.o2 = torch.zeros((cap, obs_dim), device=device)
        self.d = torch.zeros(cap, device=device)

    def add(self, o, a, r, o2, d):
        n = o.shape[0]
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.cap
        self.o[idx] = o; self.a[idx] = a; self.r[idx] = r; self.o2[idx] = o2; self.d[idx] = d
        self.ptr = (self.ptr + n) % self.cap
        self.full = self.full or self.ptr < n
        # (approximate wrap flag; fine for large buffers)

    def sample(self, bs):
        hi = self.cap if self.full else max(1, self.ptr)
        idx = torch.randint(0, hi, (bs,), device=self.device)
        return self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx]


class TD3:
    def __init__(self, env, device="cuda", gamma=0.99, tau=0.005, actor_lr=1e-3,
                 critic_lr=1e-3, policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                 expl_noise=0.1, buffer_cap=1_000_000, batch_size=256, seed=0):
        torch.manual_seed(seed)
        self.env = env
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.obs_dim = env.single_observation_space.shape[0]
        self.act_dim = env.action_dim
        self.act_limit = np.abs(env.single_action_space.high).astype(np.float32)
        self.gamma, self.tau = gamma, tau
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.policy_delay, self.expl_noise, self.batch_size = policy_delay, expl_noise, batch_size

        self.actor = Actor(self.obs_dim, self.act_dim, self.act_limit).to(self.device)
        self.actor_t = Actor(self.obs_dim, self.act_dim, self.act_limit).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.q1 = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.q2 = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.q1_t = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.q2_t = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.qopt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr)
        self.buf = ReplayBuffer(buffer_cap, self.obs_dim, self.act_dim, self.device)
        self._alimit_t = torch.as_tensor(self.act_limit, device=self.device)
        self.obs_rms = RunningMeanStd(self.obs_dim, self.device)

    def _act(self, o, explore=True):
        with torch.no_grad():
            a = self.actor(self.obs_rms.norm(o))
            if explore:
                a = a + torch.randn_like(a) * self.expl_noise * self._alimit_t
            return torch.clamp(a, -self._alimit_t, self._alimit_t)

    def _update(self, step):
        o_raw, a, r, o2_raw, d = self.buf.sample(self.batch_size)
        o = self.obs_rms.norm(o_raw)
        o2 = self.obs_rms.norm(o2_raw)
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_t(o2) + noise * self._alimit_t).clamp(-self._alimit_t, self._alimit_t)
            q_targ = torch.min(self.q1_t(o2, a2), self.q2_t(o2, a2))
            y = r + self.gamma * (1 - d) * q_targ
        q_loss = F.mse_loss(self.q1(o, a), y) + F.mse_loss(self.q2(o, a), y)
        self.qopt.zero_grad(set_to_none=True); q_loss.backward(); self.qopt.step()

        if step % self.policy_delay == 0:
            a_loss = -self.q1(o, self.actor(o)).mean()
            self.aopt.zero_grad(set_to_none=True); a_loss.backward(); self.aopt.step()
            for net, net_t in ((self.actor, self.actor_t), (self.q1, self.q1_t), (self.q2, self.q2_t)):
                with torch.no_grad():
                    for p, pt in zip(net.parameters(), net_t.parameters()):
                        pt.mul_(1 - self.tau).add_(self.tau * p)
        return float(q_loss.detach())

    def train(self, total_env_steps, start_steps=5000, updates_per_step=1, log_every=20):
        """total_env_steps counts env *ticks* (each tick = num_envs transitions)."""
        N = self.env.num_envs
        obs = _to_torch(self.env.reset(seed=0), self.device).float()
        ep_ret = torch.zeros(N, device=self.device)
        returns_log = []
        collected = 0
        tick = 0
        while collected < total_env_steps:
            if collected < start_steps:
                a = (torch.rand((N, self.act_dim), device=self.device) * 2 - 1) * self._alimit_t
            else:
                a = self._act(obs, explore=True)
            no, r, term, trunc, info = self.env.step(_to_backend(a))
            no_t = _to_torch(no, self.device).float()
            r_t = _to_torch(r, self.device).float()
            term_t = _to_torch(term, self.device).float()
            done_t = _to_torch((term | trunc), self.device).float() if hasattr(term, "__or__") else term_t
            self.buf.add(obs, a, r_t, no_t, term_t)  # bootstrap masks on termination only
            ep_ret += r_t
            fin = done_t.bool()
            if fin.any():
                returns_log.extend(ep_ret[fin].tolist())
                ep_ret = ep_ret * (~fin)
            obs = no_t
            collected += N
            tick += 1
            if collected >= start_steps:
                for _ in range(updates_per_step):
                    self._update(tick)
            if tick % log_every == 0 and returns_log:
                recent = np.mean(returns_log[-200:])
                print(f"  steps={collected:>9,d}  ep_return(mean last200)={recent:8.2f}  "
                      f"buffer={min(self.buf.ptr if not self.buf.full else self.buf.cap, self.buf.cap):,d}")
        return returns_log
