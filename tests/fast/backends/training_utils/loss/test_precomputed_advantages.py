"""--advantage-estimator precomputed: the rollout side's per-token advantages become the advantages unchanged."""

import pytest
import torch

from miles.backends.training_utils.loss.objective import compute_advantages_and_returns, loss_function

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state, make_rollout_data

VOCAB_SIZE = 32
PROMPT_LENS = [6, 9, 4]
RESPONSE_LENS = [5, 7, 3]


def _inputs(args) -> dict:
    return make_inputs(7, len(PROMPT_LENS), PROMPT_LENS, RESPONSE_LENS, VOCAB_SIZE, args)


def _advantages(args, inputs: dict, token_advantages: list[torch.Tensor] | None = None) -> dict:
    rollout_data = make_rollout_data(inputs)
    if token_advantages is not None:
        rollout_data["token_advantages"] = deep_clone(token_advantages)
    compute_advantages_and_returns(args, rollout_data)
    return rollout_data


def _loss(args, inputs: dict, advantages: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    batch = make_batch(inputs, "policy_loss")
    batch["advantages"] = deep_clone(advantages)
    logits = deep_clone(inputs["policy_logits"]).requires_grad_(True)
    loss, _, _ = loss_function(args, batch, 1, logits)
    loss.backward()
    return loss.detach(), logits.grad


def test_token_advantages_become_the_advantages_and_returns() -> None:
    make_parallel_state()
    args = make_args(advantage_estimator="precomputed")
    inputs = _inputs(args)
    token_advantages = [torch.randn(n, dtype=torch.float64) for n in RESPONSE_LENS]

    rollout_data = _advantages(args, inputs, token_advantages)

    for got, want in zip(rollout_data["advantages"], token_advantages, strict=True):
        assert got.dtype == torch.float32
        torch.testing.assert_close(got, want.float())
    for got, want in zip(rollout_data["returns"], token_advantages, strict=True):
        torch.testing.assert_close(got, want.float())


def test_matches_grpo_when_every_token_carries_the_sample_scalar() -> None:
    """Broadcasting each sample's normalized reward to its tokens reproduces grpo, advantages and loss alike.
    The reward-level KL (kl_coef) is ignored by both."""
    make_parallel_state()
    grpo_args = make_args(advantage_estimator="grpo", kl_coef=0.1)
    precomputed_args = make_args(advantage_estimator="precomputed", kl_coef=0.1)
    inputs = _inputs(grpo_args)
    broadcast = [torch.full((n,), reward) for n, reward in zip(RESPONSE_LENS, inputs["rewards"], strict=True)]

    grpo = _advantages(grpo_args, inputs)
    precomputed = _advantages(precomputed_args, inputs, broadcast)

    for a, b in zip(grpo["advantages"], precomputed["advantages"], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    grpo_loss, grpo_grad = _loss(grpo_args, inputs, grpo["advantages"])
    precomputed_loss, precomputed_grad = _loss(precomputed_args, inputs, precomputed["advantages"])
    torch.testing.assert_close(grpo_loss, precomputed_loss, rtol=0, atol=0)
    torch.testing.assert_close(grpo_grad, precomputed_grad, rtol=0, atol=0)


def test_a_zero_token_advantage_leaves_that_token_out_of_the_policy_gradient() -> None:
    """Per-token values reach the loss per token: tokens whose advantage is 0 get no gradient (no entropy or KL
    term), the others do. The clip range is opened up because make_inputs draws the old log-probs at random, so
    PPO clipping would otherwise zero the gradient of most tokens regardless of their advantage."""
    make_parallel_state()
    args = make_args(
        advantage_estimator="precomputed",
        entropy_coef=0.0,
        observe_training_entropy=False,
        use_kl_loss=False,
        eps_clip=1e9,
        eps_clip_high=1e9,
    )
    inputs = _inputs(args)
    zeroed = {0: [1, 3], 1: [0, 1, 2, 6], 2: [2]}
    token_advantages = []
    for i, n in enumerate(RESPONSE_LENS):
        values = torch.linspace(-1.0, 1.5, n) + 0.1
        values[zeroed[i]] = 0.0
        token_advantages.append(values)

    rollout_data = _advantages(args, inputs, token_advantages)
    _, grad = _loss(args, inputs, rollout_data["advantages"])

    # thd: the logit row that predicts response token j of sample i sits at its packed prompt end + j - 1
    start = 0
    for i, (prompt_len, response_len) in enumerate(zip(PROMPT_LENS, RESPONSE_LENS, strict=True)):
        rows = grad[0, start + prompt_len - 1 : start + prompt_len - 1 + response_len]
        for j in range(response_len):
            if j in zeroed[i]:
                assert torch.count_nonzero(rows[j]) == 0, f"sample {i} token {j}"
            else:
                assert torch.count_nonzero(rows[j]) > 0, f"sample {i} token {j}"
        start += prompt_len + response_len


def test_opd_adjusts_the_advantages_and_leaves_the_returns() -> None:
    make_parallel_state()
    args = make_args(advantage_estimator="precomputed", use_opd=True, opd_type="sglang", opd_kl_coef=0.5)
    inputs = _inputs(args)
    token_advantages = [torch.ones(n) for n in RESPONSE_LENS]
    reverse_kl = [torch.full((n,), 2.0) for n in RESPONSE_LENS]
    rollout_data = make_rollout_data(inputs)
    rollout_data["token_advantages"] = token_advantages
    rollout_data["opd_reverse_kl"] = reverse_kl

    compute_advantages_and_returns(args, rollout_data)

    for advantage, returned in zip(rollout_data["advantages"], rollout_data["returns"], strict=True):
        torch.testing.assert_close(advantage, torch.zeros_like(advantage))
        torch.testing.assert_close(returned, torch.ones_like(returned))


def test_missing_token_advantages_fail() -> None:
    make_parallel_state()
    args = make_args(advantage_estimator="precomputed")
    with pytest.raises(ValueError, match="needs token_advantages"):
        _advantages(args, _inputs(args))


def test_token_advantages_of_the_wrong_length_fail() -> None:
    make_parallel_state()
    args = make_args(advantage_estimator="precomputed")
    token_advantages = [torch.zeros(n) for n in RESPONSE_LENS]
    token_advantages[1] = torch.zeros(RESPONSE_LENS[1] + 1)
    with pytest.raises(ValueError, match=r"token_advantages of sample 1 has shape \(8,\), its response \(7,\)"):
        _advantages(args, _inputs(args), token_advantages)
