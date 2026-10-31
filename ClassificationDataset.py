"""
PyTorch Dataset class for Indonesian Text Sequence Classification.
Handles tokenization and tensor mapping for IndoBERT and transformer-based classifiers.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Union, Any, Optional


class ClassificationDataset(Dataset):
    def __init__(
        self,
        text: List[str],
        target: List[Any],
        model_name: str,
        max_len: int,
        label_map: Dict[str, int],
        preprocessor: Optional[Any] = None,
    ):
        """
        Args:
            text (List[str]): List of texts (documents/sentences/tweets).
            target (List[Any]): List of class labels.
            model_name (str): The pretrained tokenizer name (e.g. 'indobenchmark/indobert-base-p1').
            max_len (int): Maximum sequence token length.
            label_map (Dict[str, int]): Dictionary mapping class labels to integers.
            preprocessor (Optional[Any]): Text preprocessor (e.g. IndonesianPreprocessor).
        """
        super().__init__()
        self.text = [str(t) for t in text]
        self.target = target
        self.tokenizer_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.label_map = label_map
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        raw_text = self.text[item]
        if self.preprocessor is not None:
            text_str = self.preprocessor.preprocess(raw_text)
        else:
            text_str = " ".join(raw_text.split())

        inputs = self.tokenizer(
            text_str,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        raw_label = self.target[item]
        # Handle string or int labels
        if raw_label in self.label_map:
            label_id = self.label_map[raw_label]
        else:
            label_id = int(raw_label)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "token_type_ids": inputs.get(
                "token_type_ids", torch.zeros_like(inputs["input_ids"])
            ).squeeze(0),
            "label": torch.tensor(label_id, dtype=torch.long),
        }
