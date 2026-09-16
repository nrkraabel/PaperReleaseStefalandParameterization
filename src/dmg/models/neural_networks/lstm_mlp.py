from typing import Optional

import torch

from dmg.models.neural_networks.ann import AnnModel
from dmg.models.neural_networks.cudnn_lstm import CudnnLstmModel
from dmg.models.neural_networks.lstm import LstmModel
from dmg.models.neural_networks.triton_lstm import TritonLstmModel
from dmg.models.neural_networks.triton_mlp import TritonMLP


class LstmMlpModel(torch.nn.Module):
    """LSTM-MLP model for multi-scale learning.

    Supports GPU and CPU forwarding.

    Parameters
    ----------
    nx1
        Number of LSTM input features.
    ny1
        Number of LSTM output features.
    hiddeninv1
        LSTM hidden size.
    nx2
        Number of MLP input features.
    ny2
        Number of MLP output features.
    hiddeninv2
        MLP hidden size.
    dr1
        Dropout rate for LSTM. Default is 0.5.
    dr2
        Dropout rate for MLP. Default is 0.5.
    cache_states
        Whether to cache hidden and cell states for LSTM.
    device
        Device to run the model on. Default is 'cpu'.
    """

    def __init__(
        self,
        *,
        nx1: int,
        ny1: int,
        hiddeninv1: int,
        nx2: int,
        ny2: int,
        hiddeninv2: int,
        dr1: Optional[float] = 0.5,
        dr2: Optional[float] = 0.5,
        cache_states: Optional[bool] = False,
        device: Optional[str] = 'cpu',
        compilable: bool = False,
        lstm_type: Optional[str] = None,
        mlp_type: Optional[str] = None,
        n_chunks: int = 12,
    ) -> None:
        super().__init__()
        self.name = 'LstmMlpModel'
        self.nx1 = nx1
        self.ny1 = ny1
        self.hiddeninv1 = hiddeninv1
        self.nx2 = nx2
        self.ny2 = ny2
        self.hiddeninv2 = hiddeninv2
        self.dr1 = dr1
        self.dr2 = dr2
        self.cache_states = cache_states
        self.device = device

        self.n_chunks = n_chunks
        self.hn, self._hn_cache = None, None  # hidden state
        self.cn, self._cn_cache = None, None  # cell state

        if lstm_type == 'triton':
            self.lstminv = TritonLstmModel(
                nx=nx1,
                ny=ny1,
                hidden_size=hiddeninv1,
                dr=dr1,
            )
        elif torch.device(device).type == 'cpu' or compilable:
            # PyTorch LSTM (CPU-compatible, torch.compile-compatible).
            self.lstminv = LstmModel(
                nx=nx1,
                ny=ny1,
                hidden_size=hiddeninv1,
                dr=dr1,
                cache_states=cache_states,
            )
        else:
            # GPU-only CuDNN LSTM.
            self.lstminv = CudnnLstmModel(
                nx=nx1,
                ny=ny1,
                hidden_size=hiddeninv1,
                dr=dr1,
            )

        self.activation = torch.nn.Sigmoid()

        # Use Triton MLP if specified, otherwise use standard ANN
        if mlp_type == 'triton':
            self.ann = TritonMLP(
                nx=nx2,
                ny=ny2,
                hidden_size=hiddeninv2,
                dr=dr2,
            )
        else:
            self.ann = AnnModel(
                nx=nx2,
                ny=ny2,
                hidden_size=hiddeninv2,
                dr=dr2,
            )

    def get_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get hidden and cell states."""
        return self._hn_cache, self._cn_cache

    def load_states(
        self,
        states: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Load hidden and cell states."""
        for state in states:
            if state and not isinstance(state, torch.Tensor):
                raise ValueError("Each element in `states` must be a tensor.")
        if not (isinstance(states, tuple) and len(states) == 2):
            raise ValueError("`states` must be a tuple of 2 tensors.")

        device = next(self.parameters()).device
        self.hn = states[0].detach().to(device)
        self.cn = states[1].detach().to(device)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        NOTE (caching): Hidden states are always cached so that they can be
        accessed by `get_states`, but they are only available to the LSTM if
        `cache_states` is set to True.

        Parameters
        ----------
        x1
            The LSTM input tensor.
        x2
            The MLP input tensor.

        Returns
        -------
        tuple
            The LSTM and MLP output tensors.
        """
        # Split large batches into chunks to be more frugal with mem.
        n_cat = x1.size(1)
        if n_cat > self.n_chunks:
            chunk_size = n_cat // self.n_chunks
            iS = list(range(0, n_cat, chunk_size))
            iE = iS[1:] + [n_cat]
            lstm_chunks = [self.lstminv(x1[:, s:e, :]) for s, e in zip(iS, iE)]
            lstm_out = torch.cat(lstm_chunks, dim=1)
        else:
            lstm_out = self.lstminv(x1)

        act_out = self.activation(lstm_out)
        ann_out = self.ann(x2)

        if self.cache_states:
            self._hn_cache, self._cn_cache = self.lstminv.get_states()
            self.hn = self._hn_cache.to(x1.device)
            self.cn = self._cn_cache.to(x1.device)

        return (act_out, ann_out)
