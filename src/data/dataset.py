"""
PyTorch Document OCR Dataset and Character Vocabulary Tokenizer.

Implements character-level index encoding/decoding for Connectionist Temporal Classification (CTC)
loss and text recognition fine-tuning.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from src.preprocessing.image_processor import DocumentImageProcessor
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from ..preprocessing.image_processor import DocumentImageProcessor
    from ..utils.logger import get_logger

logger = get_logger("Dataset")


class VocabularyEncoder:
    """
    Bidirectional mapping between character string tokens and integer sequence indices.
    Index 0 is reserved for CTC Blank Token (<BLANK>).
    """

    BLANK_TOKEN = "<BLANK>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, vocabulary_str: str) -> None:
        """
        Initializes Vocabulary Encoder.

        Args:
            vocabulary_str (str): String containing all supported character tokens.
        """
        self.chars = [self.BLANK_TOKEN, self.UNK_TOKEN] + list(vocabulary_str)
        self.char2idx: Dict[str, int] = {char: idx for idx, char in enumerate(self.chars)}
        self.idx2char: Dict[int, str] = {idx: char for idx, char in enumerate(self.chars)}
        self.blank_idx = 0
        self.unk_idx = 1

    def encode(self, text: str) -> List[int]:
        """Converts string into integer indices."""
        return [self.char2idx.get(char, self.unk_idx) for char in text]

    def decode(self, indices: List[int], remove_blank: bool = True) -> str:
        """
        Converts integer indices back to string, removing repeated tokens and CTC blank tokens.

        Args:
            indices (List[int]): List of predicted character class indices.
            remove_blank (bool): Whether to perform CTC collapse decoding.

        Returns:
            str: Decoded target text.
        """
        decoded_chars = []
        previous_idx = None

        for idx in indices:
            if remove_blank:
                if idx == self.blank_idx:
                    previous_idx = idx
                    continue
                if idx == previous_idx:
                    continue
            char = self.idx2char.get(idx, "")
            if char not in (self.BLANK_TOKEN, self.UNK_TOKEN):
                decoded_chars.append(char)
            previous_idx = idx

        return "".join(decoded_chars)

    def __len__(self) -> int:
        return len(self.chars)


class DocumentOCRDataset(Dataset):
    """
    PyTorch Dataset for Document OCR text line recognition.
    Supports synthetic line generation if external raw image directories are absent.
    """

    def __init__(
        self,
        samples: List[Tuple[str, str]],
        vocab_encoder: VocabularyEncoder,
        processor: Optional[DocumentImageProcessor] = None,
        transform: Optional[Any] = None
    ) -> None:
        """
        Args:
            samples (List[Tuple[str, str]]): List of (image_path, ground_truth_text) pairs.
            vocab_encoder (VocabularyEncoder): Vocabulary tokenizer.
            processor (Optional[DocumentImageProcessor]): Image preprocessor instance.
            transform (Optional[Any]): Albumentations augmentation pipeline.
        """
        self.samples = samples
        self.vocab = vocab_encoder
        self.processor = processor or DocumentImageProcessor()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def _generate_synthetic_document_patch(self, text: str) -> np.ndarray:
        """
        Generates a synthetic document line patch on the fly if file is missing (for research testing).
        """
        img = np.full((40, 200, 3), 255, dtype=np.uint8)
        cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        img_path, label_text = self.samples[idx]

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is None:
                img = self._generate_synthetic_document_patch(label_text)
        else:
            img = self._generate_synthetic_document_patch(label_text)

        if self.transform is not None:
            img = self.transform(img)

        processed_img = self.processor.process(img, do_deskew=False)

        # Normalize image tensor to range [-1.0, 1.0]
        if len(processed_img.shape) == 2:
            img_tensor = torch.from_numpy(processed_img).unsqueeze(0).float() / 127.5 - 1.0
        else:
            img_tensor = torch.from_numpy(processed_img).permute(2, 0, 1).float() / 127.5 - 1.0

        encoded_label = self.vocab.encode(label_text)
        label_tensor = torch.tensor(encoded_label, dtype=torch.long)

        return img_tensor, label_tensor, len(encoded_label), label_text
