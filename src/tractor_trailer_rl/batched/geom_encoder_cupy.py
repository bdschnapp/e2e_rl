"""CuPy/numpy forward pass of the frozen geometry-distillation encoder (Phase-2B; NEW,
additive). Runs entirely on the batched backend (xp = cupy on GPU) so the encoder can
live INSIDE the env's observation pipeline with NO torch in the env loop.

Mirrors scripts/geom_encoder.py::GeomEncoder exactly:
  CNN [Conv(3x3,s2,p1)+ReLU] x3 (in->16->32->32) -> Flatten
  trunk: Linear -> ReLU -> LayerNorm
  head:  Linear -> standardized geometry; de-standardized with saved label mean/std.

The im2col buffer is CACHED per shape (self._colbuf) to avoid a fresh allocation every
step — a mitigation for the stochastic native segfault seen under the heavy per-step
encoder workload (see PHASE2_GEOMETRY_ENCODER.md §6). The math is unchanged (every
element of the buffer is overwritten each call).
"""

from __future__ import annotations

from .backend import xp


def _relu(x):
    return xp.maximum(x, xp.asarray(0, dtype=x.dtype))


def _layernorm(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / xp.sqrt(var + eps) * g + b


class GeomEncoderCupy:
    """Loads torch weights (from geom_pretrain checkpoint) and runs the forward in xp."""

    def __init__(self, ckpt):
        f32 = xp.float32
        sd = ckpt["state_dict"]

        def a(k):
            v = sd[k]
            v = v.detach().cpu().numpy() if hasattr(v, "detach") else v
            return xp.asarray(v, dtype=f32)

        # conv layers live at cnn.0 / cnn.2 / cnn.4 (indices 1,3,5 are ReLU)
        self.cw = [a("cnn.0.weight"), a("cnn.2.weight"), a("cnn.4.weight")]
        self.cb = [a("cnn.0.bias"), a("cnn.2.bias"), a("cnn.4.bias")]
        self.lw = a("trunk.0.weight"); self.lb = a("trunk.0.bias")      # Linear
        self.lng = a("trunk.2.weight"); self.lnb = a("trunk.2.bias")    # LayerNorm
        self.hw = a("head.weight"); self.hb = a("head.bias")            # Linear head
        self.mean = xp.asarray(ckpt["label_mean"], dtype=f32)
        self.std = xp.asarray(ckpt["label_std"], dtype=f32)
        self.out_dim = int(ckpt["out_dim"])
        self._colbuf = {}   # cache: (N,Cin,kh,kw,Hout,Wout) -> preallocated im2col buffer

    def _conv(self, x, W, b, stride, pad):
        """Cross-correlation matching torch.nn.Conv2d, reusing a cached im2col buffer."""
        N, Cin, H, Wd = x.shape
        Cout, _, kh, kw = W.shape
        Hout = (H + 2 * pad - kh) // stride + 1
        Wout = (Wd + 2 * pad - kw) // stride + 1
        xp_pad = xp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        key = (N, Cin, kh, kw, Hout, Wout)
        cols = self._colbuf.get(key)
        if cols is None or cols.dtype != x.dtype:
            cols = xp.empty(key, dtype=x.dtype)
            self._colbuf[key] = cols
        for i in range(kh):
            for j in range(kw):
                cols[:, :, i, j, :, :] = xp_pad[:, :, i:i + stride * Hout:stride, j:j + stride * Wout:stride]
        c2 = cols.reshape(N, Cin * kh * kw, Hout * Wout)
        Wf = W.reshape(Cout, Cin * kh * kw)
        out = xp.einsum("oc,ncp->nop", Wf, c2) + b[None, :, None]
        return out.reshape(N, Cout, Hout, Wout)

    def features(self, img):
        x = img.astype(xp.float32)
        for W, b in zip(self.cw, self.cb):
            x = _relu(self._conv(x, W, b, stride=2, pad=1))
        x = x.reshape(x.shape[0], -1)
        x = _relu(x @ self.lw.T + self.lb)                # trunk Linear + ReLU
        x = _layernorm(x, self.lng, self.lnb)             # trunk LayerNorm
        return x

    def predict(self, img):
        """Return the DE-STANDARDIZED predicted geometry (N, out_dim)."""
        z = self.features(img) @ self.hw.T + self.hb      # standardized prediction
        return z * self.std + self.mean
