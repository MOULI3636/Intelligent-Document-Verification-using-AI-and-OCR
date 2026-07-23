"""
PyTorch Custom Collate Function and DataLoader Creation Factory.
"""

from typing import List, Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader

from src.data.dataset import DocumentOCRDataset, VocabularyEncoder


class OCRCollate:
    """
    Custom Collate Callable padding variable-length targets into a uniform tensor batch for CTC Loss.
    """

    def __init__(self, pad_idx: int = 0) -> None:
        self.pad_idx = pad_idx

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor, int, str]]) -> Dict[str, Any]:
        """
        Pads target sequences and batches images.

        Returns:
            Dict[str, Any]: Batched dictionary with keys:
                - 'images': [B, C, H, W] tensor
                - 'targets': [B, max_target_len] padded tensor
                - 'target_lengths': [B] tensor containing unpadded sequence lengths
                - 'raw_texts': List[str] of original label strings
        """
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        target_lengths = [item[2] for item in batch]
        raw_texts = [item[3] for item in batch]

        images_tensor = torch.stack(images, dim=0)

        # Pad 1D label tensors to maximum target length in batch
        max_len = max(target_lengths)
        padded_targets = torch.full((len(targets), max_len), self.pad_idx, dtype=torch.long)
        
        for idx, target in enumerate(targets):
            padded_targets[idx, :len(target)] = target

        return {
            "images": images_tensor,
            "targets": padded_targets,
            "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
            "raw_texts": raw_texts
        }


def create_dataloader(
    dataset: DocumentOCRDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True
) -> DataLoader:
    """
    Creates a PyTorch DataLoader instance configured with custom OCR sequence collator.
    """
    collate_fn = OCRCollate(pad_idx=dataset.vocab.blank_idx)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
