"""
data.encoder
============
CTC Label Encoder converting text labels to PyTorch LongTensors for PyTorch CTCLoss.

Author: Ayesha Amer (Day 4 Scope)
Project: NAQSHKASH FYP

Design Contract:
----------------
- `encode_text`: Converts a single label string to a 1D `torch.LongTensor`.
- `encode_batch`: Encodes a list of label strings into flat concatenated 1D `targets`
  and 1D `target_lengths` strictly following PyTorch `nn.CTCLoss` specifications.
- Error handling: Raises informative KeyError on any unknown token.
"""

from typing import List, Dict, Union, Iterable, Optional
import torch

from .vocabulary import TalimVocabulary, normalize_label_text


class CTCLabelEncoder:
    """
    Encoder converting text labels to PyTorch tensors for CTC Loss computation.
    
    Parameters
    ----------
    vocabulary : TalimVocabulary
        Initialized TalimVocabulary mapping instance.
    """

    def __init__(self, vocabulary: TalimVocabulary):
        if not isinstance(vocabulary, TalimVocabulary):
            raise TypeError(f"Expected TalimVocabulary, got {type(vocabulary)}")
        self.vocabulary: TalimVocabulary = vocabulary

    @property
    def blank_index(self) -> int:
        """Return the reserved CTC blank token index (0)."""
        return self.vocabulary.blank_index

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size including the CTC blank token."""
        return len(self.vocabulary)

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode a single label text string into a 1D PyTorch LongTensor of token IDs.
        
        Parameters
        ----------
        text : str
            Raw or normalized label string.
            
        Returns
        -------
        torch.Tensor
            1D tensor of shape (L,) and dtype torch.long.
        """
        token_ids = self.vocabulary.encode(text)
        return torch.tensor(token_ids, dtype=torch.long)

    def encode_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of label strings into concatenated 1D targets and target lengths.
        
        This matches the standard PyTorch `nn.CTCLoss` 1D targets format:
        - `targets`: 1D LongTensor of shape (sum(target_lengths),).
        - `target_lengths`: 1D LongTensor of shape (N,) containing length of each label.
        
        Parameters
        ----------
        texts : List[str]
            List of label text strings.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - `"targets"`: 1D torch.LongTensor with concatenated token IDs.
            - `"target_lengths"`: 1D torch.LongTensor with individual label lengths.
        """
        if not isinstance(texts, (list, tuple)):
            raise TypeError(f"Expected list of strings, got {type(texts)}")

        all_token_ids: List[int] = []
        lengths: List[int] = []

        for text in texts:
            token_ids = self.vocabulary.encode(text)
            all_token_ids.extend(token_ids)
            lengths.append(len(token_ids))

        targets = torch.tensor(all_token_ids, dtype=torch.long)
        target_lengths = torch.tensor(lengths, dtype=torch.long)

        return {
            "targets": targets,
            "target_lengths": target_lengths,
        }

    def decode_tokens(self, token_ids: Union[List[int], torch.Tensor], remove_blank: bool = True) -> str:
        """
        Decode a sequence of token IDs back into string format.
        
        Parameters
        ----------
        token_ids : Union[List[int], torch.Tensor]
            1D sequence or tensor of token IDs.
        remove_blank : bool, default=True
            Whether to skip CTC blank index (0).
            
        Returns
        -------
        str
            Decoded text string.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return self.vocabulary.decode(token_ids, remove_blank=remove_blank)
