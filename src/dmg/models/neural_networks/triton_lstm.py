"""Triton-accelerated LSTM cell for single-layer unidirectional LSTM.

Replaces CuDNN LSTM (and torch._VF.lstm) with a custom Triton kernel
that processes all batch elements in a single kernel call.

Key advantages vs CuDNN:
- No chunking: one kernel launch for all batch elements
- Memory-efficient: recomputes gates during backward (no reserve space)
- Fused gate activations (sigmoid/tanh) within the kernel

Backward uses a two-phase approach:
  Phase 1 (Triton): Sequential BPTT --> grad_x, grad_h0, grad_c0, gate deltas
  Phase 2 (PyTorch): Weight grads via einsum on saved gate deltas
"""

import math

import torch
import triton
import triton.language as tl
from torch.autograd import Function
from triton.language.extra.cuda import libdevice


# ── Forward kernel ────────────────────────────────────────────────
@triton.jit
def _lstm_fwd_kernel(
    # Input: [nsteps, batch, in_size]
    x_ptr,
    # Weights (transposed for dot): w_ih.T [in_size, 4*H], w_hh.T [H, 4*H]
    w_ih_t_ptr,
    w_hh_t_ptr,
    # Combined bias: b_ih + b_hh, shape [4*H]
    bias_ptr,
    # Initial states: [batch, H]
    h0_ptr,
    c0_ptr,
    # Output: [nsteps, batch, H]
    out_ptr,
    # Final states: [batch, H]
    hn_ptr,
    cn_ptr,
    # Saved c for backward: [nsteps, batch, H]
    c_save_ptr,
    # Strides (in elements)
    x_stride_t,  # = batch * in_size
    x_stride_b,  # = in_size
    out_stride_t,  # = batch * H
    out_stride_b,  # = H
    # Dims
    nsteps,
    batch_size,
    IN_SIZE: tl.constexpr,
    H: tl.constexpr,
    H4: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Forward Triton kernel replacing HydroDL's cuDNN LSTM model.

    Dot products always use fp16 operands for tensor-core acceleration and to
    fit within A100 shared-memory limits.  tl.dot accumulates in fp32
    internally, so gate activations and state updates remain fp32-accurate.
    """
    pid = tl.program_id(0)
    b_start = pid * BLOCK_B
    b_idx = b_start + tl.arange(0, BLOCK_B)
    b_mask = b_idx < batch_size

    h_range = tl.arange(0, H)
    in_range = tl.arange(0, IN_SIZE)

    # Load initial states [BLOCK_B, H] in fp32
    state_offs = b_idx[:, None] * H + h_range[None, :]
    h = tl.load(h0_ptr + state_offs, mask=b_mask[:, None], other=0.0).to(tl.float32)
    c = tl.load(c0_ptr + state_offs, mask=b_mask[:, None], other=0.0).to(tl.float32)

    # Load combined bias per gate [H] in fp32
    b_i = tl.load(bias_ptr + h_range).to(tl.float32)
    b_f = tl.load(bias_ptr + H + h_range).to(tl.float32)
    b_g = tl.load(bias_ptr + 2 * H + h_range).to(tl.float32)
    b_o = tl.load(bias_ptr + 3 * H + h_range).to(tl.float32)

    for t in range(nsteps):
        # Load input tile [BLOCK_B, IN_SIZE]
        x_offs = t * x_stride_t + b_idx[:, None] * x_stride_b + in_range[None, :]
        x_tile = tl.load(x_ptr + x_offs, mask=b_mask[:, None], other=0.0).to(tl.float16)

        # Cast h for dot (fp16 for tensor cores, accumulates in fp32)
        h_half = h.to(tl.float16)

        # Input gates: x @ w_ih_T  (per gate to avoid slicing)
        w = tl.load(w_ih_t_ptr + in_range[:, None] * H4 + h_range[None, :])
        gi_i = tl.dot(x_tile, w)
        w = tl.load(w_ih_t_ptr + in_range[:, None] * H4 + H + h_range[None, :])
        gi_f = tl.dot(x_tile, w)
        w = tl.load(w_ih_t_ptr + in_range[:, None] * H4 + 2 * H + h_range[None, :])
        gi_g = tl.dot(x_tile, w)
        w = tl.load(w_ih_t_ptr + in_range[:, None] * H4 + 3 * H + h_range[None, :])
        gi_o = tl.dot(x_tile, w)

        # Hidden gates: h @ w_hh_T  (per gate)
        w = tl.load(w_hh_t_ptr + h_range[:, None] * H4 + h_range[None, :])
        gh_i = tl.dot(h_half, w)
        w = tl.load(w_hh_t_ptr + h_range[:, None] * H4 + H + h_range[None, :])
        gh_f = tl.dot(h_half, w)
        w = tl.load(w_hh_t_ptr + h_range[:, None] * H4 + 2 * H + h_range[None, :])
        gh_g = tl.dot(h_half, w)
        w = tl.load(w_hh_t_ptr + h_range[:, None] * H4 + 3 * H + h_range[None, :])
        gh_o = tl.dot(h_half, w)

        # Gate activations (fp32)
        i_gate = tl.sigmoid(gi_i + gh_i + b_i[None, :])
        f_gate = tl.sigmoid(gi_f + gh_f + b_f[None, :])
        g_gate = libdevice.tanh(gi_g + gh_g + b_g[None, :])
        o_gate = tl.sigmoid(gi_o + gh_o + b_o[None, :])

        # State update
        c = f_gate * c + i_gate * g_gate
        h = o_gate * libdevice.tanh(c)

        # Store output h and save c
        out_offs = t * out_stride_t + b_idx[:, None] * out_stride_b + h_range[None, :]
        tl.store(
            out_ptr + out_offs, h.to(out_ptr.dtype.element_ty), mask=b_mask[:, None]
        )
        tl.store(
            c_save_ptr + out_offs,
            c.to(c_save_ptr.dtype.element_ty),
            mask=b_mask[:, None],
        )

    # Store final states
    tl.store(hn_ptr + state_offs, h.to(hn_ptr.dtype.element_ty), mask=b_mask[:, None])
    tl.store(cn_ptr + state_offs, c.to(cn_ptr.dtype.element_ty), mask=b_mask[:, None])


@triton.jit
def _lstm_bwd_kernel(
    # Grad output: [nsteps, batch, H]
    grad_out_ptr,
    # Inputs (for recomputation): [nsteps, batch, in_size]
    x_ptr,
    # Weights (transposed for gate recomputation)
    w_ih_t_ptr,  # [in_size, 4*H]
    w_hh_t_ptr,  # [H, 4*H]
    # Weights (non-transposed, for grad_h/grad_x computation)
    w_ih_ptr,  # [4*H, in_size]
    w_hh_ptr,  # [4*H, H]
    # Combined bias [4*H]
    bias_ptr,
    # Saved from forward
    h0_ptr,  # [batch, H]
    c0_ptr,  # [batch, H]
    h_all_ptr,  # [nsteps, batch, H] (output from forward)
    c_all_ptr,  # [nsteps, batch, H]
    # Grad outputs
    grad_x_ptr,  # [nsteps, batch, in_size]
    grad_h0_ptr,  # [batch, H]
    grad_c0_ptr,  # [batch, H]
    # Gate deltas output: [nsteps, batch, 4*H]
    dp_ptr,
    # Terminal state grads
    grad_hn_ptr,  # [batch, H]
    grad_cn_ptr,  # [batch, H]
    # Strides
    x_stride_t,
    x_stride_b,
    out_stride_t,
    out_stride_b,
    dp_stride_t,  # = batch * 4*H
    dp_stride_b,  # = 4*H
    # Dims
    nsteps,
    batch_size,
    IN_SIZE: tl.constexpr,
    H: tl.constexpr,
    H4: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Backward kernel (phase 1 -- BPTT only).

    Computes grad_x, grad_h0, grad_c0 and saves gate deltas (dp) per timestep.
    Weight grads are computed in Phase 2 using PyTorch einsum.
    """
    pid = tl.program_id(0)
    b_start = pid * BLOCK_B
    b_idx = b_start + tl.arange(0, BLOCK_B)
    b_mask = b_idx < batch_size

    h_range = tl.arange(0, H)
    in_range = tl.arange(0, IN_SIZE)

    # Load combined bias per gate
    b_i = tl.load(bias_ptr + h_range).to(tl.float32)
    b_f = tl.load(bias_ptr + H + h_range).to(tl.float32)
    b_g = tl.load(bias_ptr + 2 * H + h_range).to(tl.float32)
    b_o = tl.load(bias_ptr + 3 * H + h_range).to(tl.float32)

    # Initialize running grad_h, grad_c from terminal grads [BLOCK_B, H]
    state_offs = b_idx[:, None] * H + h_range[None, :]
    grad_h = tl.load(grad_hn_ptr + state_offs, mask=b_mask[:, None], other=0.0).to(
        tl.float32
    )
    grad_c = tl.load(grad_cn_ptr + state_offs, mask=b_mask[:, None], other=0.0).to(
        tl.float32
    )

    for t_rev in range(nsteps):
        t = nsteps - 1 - t_rev

        # Add grad from output at this timestep
        go_offs = t * out_stride_t + b_idx[:, None] * out_stride_b + h_range[None, :]
        grad_h = grad_h + tl.load(
            grad_out_ptr + go_offs, mask=b_mask[:, None], other=0.0
        ).to(tl.float32)

        # Load saved states for recomputation
        c_t = tl.load(c_all_ptr + go_offs, mask=b_mask[:, None], other=0.0).to(
            tl.float32
        )
        if t > 0:
            prev_offs = (
                (t - 1) * out_stride_t
                + b_idx[:, None] * out_stride_b
                + h_range[None, :]
            )
            h_prev = tl.load(h_all_ptr + prev_offs, mask=b_mask[:, None], other=0.0).to(
                tl.float32
            )
            c_prev = tl.load(c_all_ptr + prev_offs, mask=b_mask[:, None], other=0.0).to(
                tl.float32
            )
        else:
            h_prev = tl.load(h0_ptr + state_offs, mask=b_mask[:, None], other=0.0).to(
                tl.float32
            )
            c_prev = tl.load(c0_ptr + state_offs, mask=b_mask[:, None], other=0.0).to(
                tl.float32
            )

        # Load input
        x_offs = t * x_stride_t + b_idx[:, None] * x_stride_b + in_range[None, :]
        x_tile = tl.load(x_ptr + x_offs, mask=b_mask[:, None], other=0.0).to(tl.float16)
        h_prev_half = h_prev.to(tl.float16)

        # ── Recompute gates ──
        w = tl.load(w_ih_t_ptr + in_range[:, None] * H4 + h_range[None, :])
        gi_i = tl.dot(x_tile, w)
        w = tl.load(w_ih_t_ptr + in_range[:, None] * H4 + H + h_range[None, :])
        gi_f = tl.dot(x_tile, w)
        w = tl.load(w_ih_t_ptr + in_range[:, None] * H4 + 2 * H + h_range[None, :])
        gi_g = tl.dot(x_tile, w)
        w = tl.load(w_ih_t_ptr + in_range[:, None] * H4 + 3 * H + h_range[None, :])
        gi_o = tl.dot(x_tile, w)

        w = tl.load(w_hh_t_ptr + h_range[:, None] * H4 + h_range[None, :])
        gh_i = tl.dot(h_prev_half, w)
        w = tl.load(w_hh_t_ptr + h_range[:, None] * H4 + H + h_range[None, :])
        gh_f = tl.dot(h_prev_half, w)
        w = tl.load(w_hh_t_ptr + h_range[:, None] * H4 + 2 * H + h_range[None, :])
        gh_g = tl.dot(h_prev_half, w)
        w = tl.load(w_hh_t_ptr + h_range[:, None] * H4 + 3 * H + h_range[None, :])
        gh_o = tl.dot(h_prev_half, w)

        i_gate = tl.sigmoid(gi_i + gh_i + b_i[None, :])
        f_gate = tl.sigmoid(gi_f + gh_f + b_f[None, :])
        g_gate = libdevice.tanh(gi_g + gh_g + b_g[None, :])
        o_gate = tl.sigmoid(gi_o + gh_o + b_o[None, :])

        # ── Backward through cell ──
        tanh_c = libdevice.tanh(c_t)
        d_o = grad_h * tanh_c
        d_tanh_c = grad_h * o_gate
        grad_c = grad_c + d_tanh_c * (1.0 - tanh_c * tanh_c)

        d_f = grad_c * c_prev
        d_i = grad_c * g_gate
        d_g = grad_c * i_gate
        grad_c_prev = grad_c * f_gate

        # Gate activation derivatives --> gate deltas
        dp_i = d_i * i_gate * (1.0 - i_gate)
        dp_f = d_f * f_gate * (1.0 - f_gate)
        dp_g = d_g * (1.0 - g_gate * g_gate)
        dp_o = d_o * o_gate * (1.0 - o_gate)

        # ── Store gate deltas dp [BLOCK_B, 4*H] ──
        dp_base = t * dp_stride_t + b_idx[:, None] * dp_stride_b
        tl.store(
            dp_ptr + dp_base + h_range[None, :],
            dp_i.to(dp_ptr.dtype.element_ty),
            mask=b_mask[:, None],
        )
        tl.store(
            dp_ptr + dp_base + H + h_range[None, :],
            dp_f.to(dp_ptr.dtype.element_ty),
            mask=b_mask[:, None],
        )
        tl.store(
            dp_ptr + dp_base + 2 * H + h_range[None, :],
            dp_g.to(dp_ptr.dtype.element_ty),
            mask=b_mask[:, None],
        )
        tl.store(
            dp_ptr + dp_base + 3 * H + h_range[None, :],
            dp_o.to(dp_ptr.dtype.element_ty),
            mask=b_mask[:, None],
        )

        # ── grad_h_prev = dp @ w_hh (4 dot products) ──
        dp_i_half = dp_i.to(tl.float16)
        dp_f_half = dp_f.to(tl.float16)
        dp_g_half = dp_g.to(tl.float16)
        dp_o_half = dp_o.to(tl.float16)

        # w_hh: [4*H, H], load per-gate sub-matrix [H, H]
        w = tl.load(w_hh_ptr + h_range[:, None] * H + h_range[None, :])
        grad_h_prev = tl.dot(dp_i_half, w)
        w = tl.load(w_hh_ptr + (H + h_range[:, None]) * H + h_range[None, :])
        grad_h_prev = grad_h_prev + tl.dot(dp_f_half, w)
        w = tl.load(w_hh_ptr + (2 * H + h_range[:, None]) * H + h_range[None, :])
        grad_h_prev = grad_h_prev + tl.dot(dp_g_half, w)
        w = tl.load(w_hh_ptr + (3 * H + h_range[:, None]) * H + h_range[None, :])
        grad_h_prev = grad_h_prev + tl.dot(dp_o_half, w)

        # ── grad_x = dp @ w_ih (4 dot products) ──
        w = tl.load(w_ih_ptr + h_range[:, None] * IN_SIZE + in_range[None, :])
        grad_x_t = tl.dot(dp_i_half, w)
        w = tl.load(w_ih_ptr + (H + h_range[:, None]) * IN_SIZE + in_range[None, :])
        grad_x_t = grad_x_t + tl.dot(dp_f_half, w)
        w = tl.load(w_ih_ptr + (2 * H + h_range[:, None]) * IN_SIZE + in_range[None, :])
        grad_x_t = grad_x_t + tl.dot(dp_g_half, w)
        w = tl.load(w_ih_ptr + (3 * H + h_range[:, None]) * IN_SIZE + in_range[None, :])
        grad_x_t = grad_x_t + tl.dot(dp_o_half, w)

        # Store grad_x
        tl.store(
            grad_x_ptr + x_offs,
            grad_x_t.to(grad_x_ptr.dtype.element_ty),
            mask=b_mask[:, None],
        )

        # Propagate grad_h and grad_c backward
        grad_h = grad_h_prev
        grad_c = grad_c_prev

    # Store grad_h0, grad_c0
    tl.store(
        grad_h0_ptr + state_offs,
        grad_h.to(grad_h0_ptr.dtype.element_ty),
        mask=b_mask[:, None],
    )
    tl.store(
        grad_c0_ptr + state_offs,
        grad_c.to(grad_c0_ptr.dtype.element_ty),
        mask=b_mask[:, None],
    )


class TritonLstmFunction(Function):
    """Wrapper for Pytorch autograd to call Triton LSTM kernels."""

    @staticmethod
    def forward(ctx, input, w_ih, w_hh, b_ih, b_hh, h0, c0):
        """Forward pass.

        Parameters
        ----------
        input
            [nsteps, batch, in_size]
        w_ih
            [4*H, in_size]
        w_hh
            [4*H, H]
        b_ih
            [4*H]
        b_hh
            [4*H]
        h0
            [1, batch, H]
        c0
            [1, batch, H]

        Returns
        -------
        output
            [nsteps, batch, H]
        hn
            [1, batch, H]
        cn
            [1, batch, H]
        """
        nsteps, batch, in_size = input.shape
        H = w_hh.shape[1]
        H4 = 4 * H
        device = input.device
        BLOCK_B = 64

        # Squeeze initial states: [1, batch, H] -> [batch, H]
        h0_2d = h0.squeeze(0).contiguous()
        c0_2d = c0.squeeze(0).contiguous()

        # Pre-transpose weights for efficient tl.dot, always fp16 for
        # tensor-core dot products (tl.dot accumulates in fp32 internally).
        # Also prepare non-transposed versions for backward.
        w_ih_t = w_ih.T.to(torch.float16).contiguous()  # [in_size, 4*H]
        w_hh_t = w_hh.T.to(torch.float16).contiguous()  # [H, 4*H]
        bias = (b_ih + b_hh).float().contiguous()  # [4*H] keep bias fp32
        w_ih_nt = w_ih.to(torch.float16).contiguous()  # [4*H, in_size]
        w_hh_nt = w_hh.to(torch.float16).contiguous()  # [4*H, H]

        # Allocate outputs
        output = torch.empty(nsteps, batch, H, device=device, dtype=input.dtype)
        hn = torch.empty(batch, H, device=device, dtype=input.dtype)
        cn = torch.empty(batch, H, device=device, dtype=input.dtype)
        c_save = torch.empty(nsteps, batch, H, device=device, dtype=input.dtype)

        # Launch kernel
        num_blocks = triton.cdiv(batch, BLOCK_B)
        _lstm_fwd_kernel[(num_blocks,)](
            input,
            w_ih_t,
            w_hh_t,
            bias,
            h0_2d,
            c0_2d,
            output,
            hn,
            cn,
            c_save,
            # Strides
            batch * in_size,
            in_size,
            batch * H,
            H,
            # Dims
            nsteps,
            batch,
            IN_SIZE=in_size,
            H=H,
            H4=H4,
            BLOCK_B=BLOCK_B,
        )

        ctx.save_for_backward(input, h0_2d, c0_2d, output, c_save)
        # Cache pre-converted weights on ctx to skip redundant dtype
        # conversions in backward. Must detach to avoid autograd conflicts
        ctx.w_ih_t = w_ih_t.detach()
        ctx.w_hh_t = w_hh_t.detach()
        ctx.w_ih_nt = w_ih_nt.detach()
        ctx.w_hh_nt = w_hh_nt.detach()
        ctx.bias = bias.detach()
        ctx.nsteps = nsteps
        ctx.batch = batch
        ctx.in_size = in_size
        ctx.H = H

        return output, hn.unsqueeze(0), cn.unsqueeze(0)

    @staticmethod
    def backward(ctx, grad_output, grad_hn, grad_cn):
        """Backward pass."""
        input, h0_2d, c0_2d, h_all, c_all = ctx.saved_tensors
        nsteps = ctx.nsteps
        batch = ctx.batch
        in_size = ctx.in_size
        H = ctx.H
        H4 = 4 * H
        device = input.device
        BLOCK_B = 32
        num_blocks = triton.cdiv(batch, BLOCK_B)

        # Ensure contiguous layout -- PyTorch may pass expanded tensors
        # (e.g. stride=(0,0,0) from sum().backward()) which break kernel
        # stride assumptions.
        grad_output = grad_output.contiguous()

        # Use pre-converted weights cached from forward (no aten::copy_)
        w_ih_t = ctx.w_ih_t
        w_hh_t = ctx.w_hh_t
        w_ih_nt = ctx.w_ih_nt
        w_hh_nt = ctx.w_hh_nt
        bias = ctx.bias

        # Allocate grad outputs
        grad_x = torch.empty_like(input)
        grad_h0 = torch.empty(batch, H, device=device, dtype=input.dtype)
        grad_c0 = torch.empty(batch, H, device=device, dtype=input.dtype)

        # Gate deltas buffer: [nsteps, batch, 4*H]
        dp = torch.empty(nsteps, batch, H4, device=device, dtype=input.dtype)

        grad_hn_2d = grad_hn.squeeze(0).contiguous()
        grad_cn_2d = grad_cn.squeeze(0).contiguous()

        # Phase 1: Triton kernel -- BPTT for grad_x, grad_h0, grad_c0, dp
        # num_stages=1 disables software pipelining to fit in shared memory
        _lstm_bwd_kernel[(num_blocks,)](
            grad_output,
            input,
            w_ih_t,
            w_hh_t,
            w_ih_nt,
            w_hh_nt,
            bias,
            h0_2d,
            c0_2d,
            h_all,
            c_all,
            grad_x,
            grad_h0,
            grad_c0,
            dp,
            grad_hn_2d,
            grad_cn_2d,
            # Strides
            batch * in_size,
            in_size,
            batch * H,
            H,
            batch * H4,
            H4,
            # Dims
            nsteps,
            batch,
            IN_SIZE=in_size,
            H=H,
            H4=H4,
            BLOCK_B=BLOCK_B,
            num_stages=1,
            num_warps=4,
        )

        # Phase 2: Weight grads via matmul on flattened tensors
        # dp: [T, B, 4H], input: [T, B, IN], h_prev: [T, B, H]
        dp_2d = dp.reshape(-1, H4)  # [T*B, 4H] fp16
        x_2d = input.reshape(-1, in_size)  # [T*B, IN] fp16
        hp_2d = torch.cat([h0_2d.unsqueeze(0), h_all[:-1]], dim=0).reshape(-1, H)

        # fp16 matmul uses tensor cores; accumulate result in fp32
        grad_w_ih = (dp_2d.T @ x_2d).float()  # [4H, IN]
        grad_w_hh = (dp_2d.T @ hp_2d).float()  # [4H, H]
        grad_bias = dp_2d.float().sum(dim=0)  # [4H]

        return (
            grad_x,
            grad_w_ih.to(input.dtype),
            grad_w_hh.to(input.dtype),
            grad_bias.to(input.dtype),  # grad_b_ih
            grad_bias.to(input.dtype),  # grad_b_hh
            grad_h0.unsqueeze(0),
            grad_c0.unsqueeze(0),
        )


class TritonLstm(torch.nn.Module):
    """Drop-in replacement for the Lstm class using Triton kernels.

    Same interface: forward(input, hx, cx, do_drop_mc, dr_false).
    """

    def __init__(self, nx: int, hidden_size: int, dr: float = 0.5):
        super().__init__()
        self.name = 'TritonLstm'
        self.nx = nx
        self.hidden_size = hidden_size
        self.dr = dr

        from dmg.core.calc.dropout import DropMask, createMask

        self.w_ih = torch.nn.Parameter(torch.Tensor(hidden_size * 4, nx))
        self.w_hh = torch.nn.Parameter(torch.Tensor(hidden_size * 4, hidden_size))
        self.b_ih = torch.nn.Parameter(torch.Tensor(hidden_size * 4))
        self.b_hh = torch.nn.Parameter(torch.Tensor(hidden_size * 4))

        self._createMask = createMask
        self._DropMask = DropMask
        self._init_mask()
        self._init_parameters()

    def _init_mask(self):
        """Initialize dropout mask."""
        self.mask_w_ih = self._createMask(self.w_ih, self.dr)
        self.mask_w_hh = self._createMask(self.w_hh, self.dr)

    def _init_parameters(self):
        """Initialize parameters."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None, cx=None, do_drop_mc=False, dr_false=False):
        """Forward pass.

        Parameters
        ----------
        input
            The input tensor.
        hx
            Hidden state tensor.
        cx
            Cell state tensor.
        do_drop_mc
            Flag for applying dropout.
        dr_false
            Flag for applying dropout.
        """
        if dr_false and not do_drop_mc:
            do_drop = False
        elif self.dr > 0 and (do_drop_mc or self.training):
            do_drop = True
        else:
            do_drop = False

        batch_size = input.size(1)
        if hx is None:
            hx = input.new_zeros(1, batch_size, self.hidden_size)
        if cx is None:
            cx = input.new_zeros(1, batch_size, self.hidden_size)

        if do_drop:
            self._init_mask()
            w_ih = self._DropMask.apply(self.w_ih, self.mask_w_ih, True)
            w_hh = self._DropMask.apply(self.w_hh, self.mask_w_hh, True)
        else:
            w_ih = self.w_ih
            w_hh = self.w_hh

        output, hy, cy = TritonLstmFunction.apply(
            input,
            w_ih,
            w_hh,
            self.b_ih,
            self.b_hh,
            hx,
            cx,
        )
        return output, (hy, cy)


class TritonLstmModel(torch.nn.Module):
    """LSTM model using Triton kernels (GPU only).

    Drop-in replacement for CudnnLstmModel with identical interface.
    """

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        hidden_size: int,
        dr: float = 0.5,
        dpl: bool = False,
        cache_states: bool = False,
    ) -> None:
        super().__init__()
        self.name = 'TritonLstmModel'
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.dr = dr
        self.dpl = dpl
        self.cache_states = cache_states

        self.hn, self._hn_cache = None, None
        self.cn, self._cn_cache = None, None

        self.linear_in = torch.nn.Linear(nx, hidden_size)
        self.lstm = TritonLstm(nx=hidden_size, hidden_size=hidden_size, dr=dr)
        self.linear_out = torch.nn.Linear(hidden_size, ny)

    def get_states(self):
        """Get hidden and cell states."""
        return self._hn_cache, self._cn_cache

    def load_states(self, states):
        """Load hidden and cell states."""
        device = next(self.parameters()).device
        self.hn = states[0].detach().to(device)
        self.cn = states[1].detach().to(device)

    def forward(self, x, do_drop_mc=False, dr_false=False):
        """Forward pass.

        NOTE (caching): Hidden states are always cached so that they can be
        accessed by `get_states`, but they are only available to the LSTM if
        `cache_states` is set to True.

        Parameters
        ----------
        x
            The input tensor.
        do_drop_mc
            Flag for applying mc dropout.
        dr_false
            Flag for applying dropout.
        """
        x0 = torch.nn.functional.relu(self.linear_in(x))
        lstm_out, (hn, cn) = self.lstm(
            x0,
            self.hn,
            self.cn,
            do_drop_mc=do_drop_mc,
            dr_false=dr_false,
        )

        self._hn_cache = hn.detach().cpu()
        self._cn_cache = cn.detach().cpu()

        if self.cache_states:
            self.hn = self._hn_cache.to(x.device)
            self.cn = self._cn_cache.to(x.device)

        out = self.linear_out(lstm_out)
        if self.dpl:
            return torch.sigmoid(out)
        else:
            return out
