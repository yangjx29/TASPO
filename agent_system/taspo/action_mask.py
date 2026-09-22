"""Locate environment-action spans inside generated agent turns."""

from __future__ import annotations

import re

import torch

ACTION_PATTERN = re.compile(
    r"<(action|search|answer)>.*?</\1>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _find_subsequence(sequence: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(sequence):
        return None
    for start in range(len(sequence) - len(pattern) + 1):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    return None


def build_action_token_mask(batch, tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a token mask for executable spans and a per-row parse flag.

    Token pieces are decoded individually so their character intervals can be
    intersected with the first executable XML-like span. A malformed turn gets
    an empty action mask. Credit construction then abstains for its whole
    trajectory, preserving outcome-only GRPO without letting PI touch reasoning
    tokens.
    """
    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"].bool()
    action_mask = torch.zeros_like(response_mask)
    parsed = torch.zeros(responses.size(0), dtype=torch.bool, device=responses.device)
    response_texts = batch.non_tensor_batch.get("model_response_text")

    for row in range(responses.size(0)):
        valid_positions = response_mask[row].nonzero(as_tuple=True)[0]
        if valid_positions.numel() == 0:
            continue
        pieces = [tokenizer.decode([int(responses[row, position])], skip_special_tokens=False) for position in valid_positions]
        joined = "".join(pieces)
        match = ACTION_PATTERN.search(joined)
        if match is None:
            model_text = str(response_texts[row]) if response_texts is not None else ""
            text_match = ACTION_PATTERN.search(model_text)
            if text_match is not None:
                valid_ids = [int(responses[row, position]) for position in valid_positions]
                span_text = text_match.group(0)
                for candidate in (span_text, " " + span_text):
                    span_ids = tokenizer.encode(candidate, add_special_tokens=False)
                    start = _find_subsequence(valid_ids, span_ids)
                    if start is not None:
                        selected = valid_positions[start : start + len(span_ids)]
                        action_mask[row, selected] = True
                        parsed[row] = True
                        break
            continue

        parsed[row] = True
        span_start, span_end = match.span()
        cursor = 0
        for position, piece in zip(valid_positions, pieces):  # noqa: B905 - both derive from valid_positions
            piece_start = cursor
            piece_end = cursor + len(piece)
            if piece_end > span_start and piece_start < span_end:
                action_mask[row, position] = True
            cursor = piece_end

        if not torch.any(action_mask[row]):
            parsed[row] = False

    return action_mask, parsed
