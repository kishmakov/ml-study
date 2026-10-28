"""
Template for implementing and training an attention block.

The block takes two inputs:
  * x        -- embedding of the current token, shape (B, d_model)
  * context  -- sequence of vectors the token attends to, shape (B, T, d_ctx)
  * mask     -- optional bool mask over context, shape (B, T); True = keep

and returns the attended representation of shape (B, d_model) together with
the attention weights of shape (B, n_heads, T).

Synthetic task (key-value retrieval) that is solvable only by attending:
  context[i] = concat(key_i, value_i),  i = 0..T-1
  x          = key_j + noise           (for a random j)
  target     = value_j

Fill in the TODOs in `Attention`, then run:
  python attention/test.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

D_KEY = 16  # token embedding size (d_model)
D_VALUE = 8  # size of the value to retrieve
SEQ_LEN = 10  # T, number of context slots
N_HEADS = 4
QUERY_NOISE = 0.1

BATCH_SIZE = 256
STEPS = 3000
LR = 1e-3
EVAL_EVERY = 250

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Attention block -- implement me
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head attention of a single token over a context sequence."""

    def __init__(self, d_model: int, d_ctx: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.d_ctx = d_ctx
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # TODO: define projections
        #   query  : d_model -> d_model   (applied to x)
        #   key    : d_ctx   -> d_model   (applied to context)
        #   value  : d_ctx   -> d_model   (applied to context)
        #   output : d_model -> d_model   (applied to concatenated heads)
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,  # (B, d_model)
        context: torch.Tensor,  # (B, T, d_ctx)
        mask: torch.Tensor | None = None,  # (B, T) bool, True = attend
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO:
        #   1. q = W_q x            -> (B, H, d_head)
        #      k = W_k context      -> (B, H, T, d_head)
        #      v = W_v context      -> (B, H, T, d_head)
        #   2. scores = q . k / sqrt(d_head)          -> (B, H, T)
        #   3. scores[~mask] = -inf
        #   4. weights = softmax(scores, dim=-1)      -> (B, H, T)
        #   5. heads = sum_t weights * v              -> (B, H, d_head)
        #   6. out = W_o concat(heads)                -> (B, d_model)
        #   return out, weights
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Model wrapping the block for the retrieval task
# ---------------------------------------------------------------------------


class RetrievalModel(nn.Module):
    def __init__(self, d_key: int, d_value: int, n_heads: int):
        super().__init__()
        self.attn = Attention(d_model=d_key, d_ctx=d_key + d_value, n_heads=n_heads)
        self.norm = nn.LayerNorm(d_key)
        self.head = nn.Linear(d_key, d_value)

    def forward(self, x, context, mask=None):
        h, weights = self.attn(x, context, mask)
        h = self.norm(x + h)  # residual connection
        return self.head(h), weights


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def make_batch(batch_size: int, device: str = DEVICE):
    """Returns x (B, D_KEY), context (B, T, D_KEY + D_VALUE), mask (B, T),
    target (B, D_VALUE), index of the correct slot (B,)."""
    keys = F.normalize(torch.randn(batch_size, SEQ_LEN, D_KEY, device=device), dim=-1)
    values = torch.randn(batch_size, SEQ_LEN, D_VALUE, device=device)
    context = torch.cat([keys, values], dim=-1)

    # Random-length padding: slots past `length` are masked out.
    lengths = torch.randint(2, SEQ_LEN + 1, (batch_size,), device=device)
    mask = torch.arange(SEQ_LEN, device=device)[None, :] < lengths[:, None]

    idx = (torch.rand(batch_size, device=device) * lengths).long()
    rows = torch.arange(batch_size, device=device)
    x = keys[rows, idx] + QUERY_NOISE * torch.randn(batch_size, D_KEY, device=device)
    target = values[rows, idx]

    # Garbage in padded slots, so a correct mask actually matters.
    context = torch.where(mask[..., None], context, 100 * torch.randn_like(context))
    return x, context, mask, target, idx


# ---------------------------------------------------------------------------
# Sanity checks for the block (run before training)
# ---------------------------------------------------------------------------


def check_attention():
    torch.manual_seed(SEED)
    B, T, d_model, d_ctx, H = 3, 5, 8, 6, 2
    attn = Attention(d_model, d_ctx, H)
    x = torch.randn(B, d_model)
    context = torch.randn(B, T, d_ctx)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool
    )

    out, weights = attn(x, context, mask)
    assert out.shape == (B, d_model), f"bad output shape {tuple(out.shape)}"
    assert weights.shape == (B, H, T), f"bad weights shape {tuple(weights.shape)}"
    assert torch.allclose(weights.sum(-1), torch.ones(B, H), atol=1e-5), \
        "weights must sum to 1 over context"
    assert (weights[~mask[:, None, :].expand_as(weights)] == 0).all(), \
        "masked positions must get zero weight"

    # Masked context must not influence the output.
    context2 = context.clone()
    context2[~mask] = torch.randn_like(context2[~mask])
    out2, _ = attn(x, context2, mask)
    assert torch.allclose(out, out2, atol=1e-5), "output depends on masked slots"

    # Permuting the context must not change the output (no positional info).
    perm = torch.randperm(T)
    out3, _ = attn(x, context[:, perm], mask[:, perm])
    assert torch.allclose(out, out3, atol=1e-5), "output must be permutation invariant"

    # Gradients must flow to every parameter.
    out.sum().backward()
    for name, p in attn.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no gradient for {name}"

    print("check_attention: OK")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model: nn.Module, n_batches: int = 10):
    model.eval()
    mse, hits, total = 0.0, 0, 0
    for _ in range(n_batches):
        x, context, mask, target, idx = make_batch(BATCH_SIZE)
        pred, weights = model(x, context, mask)
        mse += F.mse_loss(pred, target, reduction="sum").item()
        # Does the head-averaged attention peak on the correct slot?
        hits += (weights.mean(1).argmax(-1) == idx).sum().item()
        total += x.shape[0]
    model.train()
    return mse / (total * D_VALUE), hits / total


def train():
    torch.manual_seed(SEED)
    model = RetrievalModel(D_KEY, D_VALUE, N_HEADS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    baseline_mse = 1.0  # always predicting 0 for N(0, 1) targets
    baseline_acc = sum(1 / n for n in range(2, SEQ_LEN + 1)) / (SEQ_LEN - 1)
    print(f"device={DEVICE}, params={sum(p.numel() for p in model.parameters())}")
    print(f"baseline MSE (predict zeros): {baseline_mse:.3f}, "
          f"random attention accuracy: {baseline_acc:.3f}")

    for step in range(1, STEPS + 1):
        x, context, mask, target, _ = make_batch(BATCH_SIZE)
        pred, _ = model(x, context, mask)
        loss = F.mse_loss(pred, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % EVAL_EVERY == 0:
            mse, acc = evaluate(model)
            print(f"step {step:5d}  train {loss.item():.4f}  "
                  f"eval MSE {mse:.4f}  attn acc {acc:.3f}")

    return model


if __name__ == "__main__":
    check_attention()
    train()
