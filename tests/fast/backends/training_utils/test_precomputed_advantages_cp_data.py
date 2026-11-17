"""Precomputed per-token advantages follow the rollout CP slice on load and reach the advantages unchanged."""

import pytest
import torch

from miles.backends.training_utils import parallel
from miles.backends.training_utils.data import multimodal
from miles.backends.training_utils.loss.objective import compute_advantages_and_returns

from .loss.loss_test_utils import make_args
from .test_opd_cp_data import _OPD_VALUES, _load_rollout_data, _parallel_state

_TOKEN_ADVANTAGES = _OPD_VALUES


@pytest.mark.parametrize(
    ("qkv_format", "cp_size", "cp_rank", "expected_indices"),
    [
        ("thd", 1, 0, [0, 1, 2, 3, 4, 5, 6]),
        ("thd", 2, 0, [6]),
        ("thd", 2, 1, [0, 1, 2, 3, 4, 5]),
        ("bshd", 2, 0, [0]),
        ("bshd", 2, 1, [1, 2, 3, 4, 5, 6]),
    ],
)
def test_token_advantages_follow_the_rollout_cp_slice_into_the_advantages(
    monkeypatch: pytest.MonkeyPatch,
    qkv_format: str,
    cp_size: int,
    cp_rank: int,
    expected_indices: list[int],
) -> None:
    rollout_data = _load_rollout_data(
        monkeypatch, qkv_format=qkv_format, cp_size=cp_size, cp_rank=cp_rank, opd_key="token_advantages"
    )
    expected = _TOKEN_ADVANTAGES[torch.tensor(expected_indices)]
    torch.testing.assert_close(rollout_data["token_advantages"][0], expected)
    assert rollout_data["token_advantages"][0].dtype == torch.float32

    # The CP rank's own slice is what the estimator sees: --use-rollout-logprobs gives KL the same local shape.
    monkeypatch.setattr(parallel, "_parallel_state", _parallel_state(cp_size=cp_size, cp_rank=cp_rank))
    rollout_data["rewards"] = [0.0]
    args = make_args(advantage_estimator="precomputed", use_rollout_logprobs=True, kl_coef=0.0, qkv_format=qkv_format)
    compute_advantages_and_returns(args, rollout_data)

    torch.testing.assert_close(rollout_data["advantages"][0], expected)
    torch.testing.assert_close(rollout_data["returns"][0], expected)


def test_multimodal_cp_reslices_token_advantages(monkeypatch: pytest.MonkeyPatch) -> None:
    rollout_data = {
        "tokens": [torch.tensor([multimodal.KIMI_VL_MEDIA_TOKEN_ID, 1, 2])],
        "loss_masks": [torch.ones(2, dtype=torch.int)],
        "total_lengths": [3],
        "response_lengths": [2],
        "multimodal_train_inputs": [{"grid_thws": torch.tensor([[1, 4, 4]])}],
        "token_advantages": [torch.tensor([0.1, 0.2])],
    }
    gathered = []

    monkeypatch.setattr(multimodal, "get_parallel_state", lambda: _parallel_state(cp_size=2))

    def _all_gather(value: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        gathered.append(value)
        return value

    monkeypatch.setattr(multimodal, "all_gather_with_cp", _all_gather)
    monkeypatch.setattr(multimodal, "slice_log_prob_with_cp", lambda value, *args, **kwargs: value)

    multimodal.expand_multimodal_rollout_data_in_place(rollout_data)

    assert len(gathered) == 1
    assert gathered[0] is rollout_data["token_advantages"][0]
