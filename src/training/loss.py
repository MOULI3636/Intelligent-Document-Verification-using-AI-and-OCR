"""
PyTorch CTC (Connectionist Temporal Classification) Loss Wrapper.

Provides robust loss computation for sequence alignment, handling log-probabilities,
zero-infinity masking, and sequence length validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLossWrapper(nn.Module):
    """
    Wrapper for PyTorch nn.CTCLoss with safety guards for unaligned sequences.
    """

    def __init__(self, blank_idx: int = 0, zero_infinity: bool = True) -> None:
        """
        Args:
            blank_idx (int): Character vocabulary index reserved for CTC blank token.
            zero_infinity (bool): Whether to zero infinite losses when targets are longer than inputs.
        """
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=zero_infinity)

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates CTC Loss.

        Args:
            log_probs (torch.Tensor): Unnormalized or normalized network logits [W_seq, Batch, Num_Classes].
            targets (torch.Tensor): Target character index tensor [Batch, Max_Target_Len].
            target_lengths (torch.Tensor): 1D tensor containing unpadded target sequence lengths.

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        seq_len, batch_size, _ = log_probs.shape
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=log_probs.device)

        # Apply log-softmax over character logits dimension
        log_probs_norm = F.log_softmax(log_probs, dim=2)

        loss = self.ctc_loss(log_probs_norm, targets, input_lengths, target_lengths)
        return loss
