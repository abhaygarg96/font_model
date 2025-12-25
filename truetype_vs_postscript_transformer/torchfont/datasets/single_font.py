"""Single font dataset module."""

import random
from collections.abc import Callable
from typing import Any, Literal

import torch
from fontTools.ttLib import TTFont
from torch.utils.data import Dataset

from truetype_vs_postscript_transformer.torchfont.io.font import (
    SegmentOutline,
    extract_point_outline,
    extract_segment_outline,
)


class SingleFontDataset(Dataset):
    """Dataset for a single font."""

    def __init__(
        self,
        font: TTFont,
        *,
        outline_mode: Literal["segment", "point"] = "segment",
        codepoints: list[int] | None = None,
        split: Literal["train", "valid", "test"] | None = None,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int | None = None,
        transform: Callable[[SegmentOutline, TTFont], Any] | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            font: Font file as a TTFont object.
            outline_mode: Outline mode to use ("segment" or "point").
            codepoints: Optional list of codepoints to filter.
            split: Subset of the dataset to load ("train", "valid", "test").
            split_ratios: Ratios for splitting the dataset (train, valid, test).
            seed: Random seed for reproducible splits.
            transform: Optional transformation function applied to each glyph.

        """
        self.font = font
        self.outline_mode = outline_mode
        self.transform = transform

        cmap = self.font.getBestCmap()
        all_codepoints = list(cmap.keys())
        self.glyph_name_codepoint_map = {}
        for codepoint in all_codepoints:
            glyph_name = cmap.get(codepoint)
            self.glyph_name_codepoint_map[glyph_name] = codepoint

        useful_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        useful_codepoints = []
        for char in useful_chars:
            codepoint = ord(char)
            useful_codepoints.append(codepoint)
        
        self.num_words = 500
        self.words = []
        for idx in range(self.num_words):
            random_word = random.choices(useful_codepoints, k=random.randint(2, 7))
            self.words.append(random_word)
        # self.words = ["hello", "how", "are", "you"]


        if codepoints is not None:
            self.codepoints = list(set(all_codepoints) & set(codepoints))
        else:
            self.codepoints = all_codepoints

        split_ratios_sum = sum(split_ratios)
        split_ratios = (
            split_ratios[0] / split_ratios_sum,
            split_ratios[1] / split_ratios_sum,
            split_ratios[2] / split_ratios_sum,
        )

        if seed is not None:
            random.seed(seed)

        shuffled_words = random.sample(
            self.words,
            len(self.words),
        )
        train_end = int(split_ratios[0] * len(shuffled_words))
        valid_end = train_end + int(split_ratios[1] * len(shuffled_words))

        self.splits = {
            "train": shuffled_words[:train_end],
            "valid": shuffled_words[train_end:valid_end],
            "test": shuffled_words[valid_end:],
        }

        if split is not None:
            self.codepoints = self.splits[split]

    def __len__(self) -> int:
        """Get the number of codepoints."""
        return len(self.words)

    def __getitem__(self, idx: int) -> tuple[int, Any]:
        """Get the glyph path and its corresponding codepoint."""
        word = self.words[idx]
        glyph_commands_list = []
        glyph_points_list = []
        for codepoint in word:
            # codepoint = self.glyph_name_codepoint_map.get(char)
            if self.outline_mode == "segment":
                glyph = extract_segment_outline(self.font, codepoint)
            else:
                glyph = extract_point_outline(self.font, codepoint)

            if self.transform is not None and glyph is not None:
                glyph = self.transform(glyph, self.font)
            
            # glyph_list.append(glyph)
            glyph_commands_list.append(glyph[0])
            glyph_points_list.append(glyph[1])

        
        # codepoint = self.codepoints[idx]

        # if self.outline_mode == "segment":
        #     glyph = extract_segment_outline(self.font, codepoint)
        # else:
        #     glyph = extract_point_outline(self.font, codepoint)

        # if self.transform is not None and glyph is not None:
        #     glyph = self.transform(glyph, self.font)
        glyph_commands = torch.cat(glyph_commands_list, dim=0)
        glyph_points = torch.cat(glyph_points_list, dim=0)
        return word, (glyph_commands, glyph_points)
