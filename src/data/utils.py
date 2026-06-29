import torch
from jaxtyping import Float, Int
from typing import NamedTuple


class Timeseries(NamedTuple):
    """A time-series window and its point-level labels.

    ``label`` is an empty tensor when the source series has no labels.
    """

    series: Float[torch.Tensor, "*batch series_len features"]  # noqa: F722
    label: Int[torch.Tensor, "*batch series_len"]  # noqa: F722

    def to(self, device: torch.device) -> "Timeseries":
        return Timeseries(
            series=self.series.to(device),
            label=self.label.to(device),
        )
