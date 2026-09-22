"""Construct privileged-teacher inputs without changing realized actions."""

from __future__ import annotations

import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


def build_taspo_teacher_batch(
    batch: DataProto,
    tokenizer,
    pi_token_budget: int = 256,
) -> tuple[DataProto, torch.Tensor]:
    """Prepend one fixed trajectory PI while preserving the full student prompt.

    The teacher receives the exact sampled action tokens. A dedicated PI budget
    extends the teacher prompt, so adding PI never evicts history from the clean
    student prompt. The returned mask records rows where non-empty PI was
    actually inserted.
    """
    if pi_token_budget <= 0:
        raise ValueError("pi_token_budget must be positive")

    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"]
    responses = batch.batch["responses"]
    batch_size = input_ids.size(0)
    response_length = responses.size(1)
    prompt_width = input_ids.size(1) - response_length
    teacher_prompt_width = prompt_width + pi_token_budget
    pi_values = batch.non_tensor_batch.get("taspo_teacher_pi")

    teacher_ids = []
    teacher_masks = []
    teacher_positions = []
    effective_pi = []

    for index in range(batch_size):
        prompt_ids = input_ids[index, :prompt_width]
        prompt_mask = attention_mask[index, :prompt_width].bool()
        clean_prompt_ids = prompt_ids[prompt_mask]

        pi_text = str(pi_values[index]).strip() if pi_values is not None else ""
        prefix_ids: list[int] = []
        guidance_ids: list[int] = []
        if pi_text:
            header_ids = tokenizer.encode(
                "[Privileged Guidance]\n",
                add_special_tokens=False,
            )
            guidance_ids = tokenizer.encode(pi_text, add_special_tokens=False)
            footer_ids = tokenizer.encode(
                "\n[End Privileged Guidance]\n\n",
                add_special_tokens=False,
            )
            wrapper_size = len(header_ids) + len(footer_ids)
            if wrapper_size < pi_token_budget:
                guidance_budget = pi_token_budget - wrapper_size
                prefix_ids = header_ids + guidance_ids[:guidance_budget] + footer_ids
            else:
                # An unusually small budget still retains actual guidance
                # rather than conditioning the teacher on a header alone.
                prefix_ids = guidance_ids[:pi_token_budget]

        effective_pi.append(bool(prefix_ids and guidance_ids))
        content = torch.cat(
            [
                torch.tensor(prefix_ids, dtype=torch.long, device=input_ids.device),
                clean_prompt_ids,
            ]
        )
        if content.numel() > teacher_prompt_width:
            raise RuntimeError("TASPO teacher prompt unexpectedly exceeded its dedicated PI budget")

        pad_length = teacher_prompt_width - content.numel()
        padded_prompt = torch.cat(
            [
                torch.full(
                    (pad_length,),
                    tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=input_ids.device,
                ),
                content,
            ]
        )
        prompt_attention = torch.cat(
            [
                torch.zeros(pad_length, dtype=attention_mask.dtype, device=input_ids.device),
                torch.ones(content.numel(), dtype=attention_mask.dtype, device=input_ids.device),
            ]
        )
        response_mask = attention_mask[index, -response_length:]
        full_ids = torch.cat([padded_prompt, responses[index]])
        full_mask = torch.cat([prompt_attention, response_mask])

        teacher_ids.append(full_ids)
        teacher_masks.append(full_mask)
        teacher_positions.append(compute_position_id_with_mask(full_mask.unsqueeze(0))[0])

    teacher_batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.stack(teacher_ids),
            "attention_mask": torch.stack(teacher_masks),
            "position_ids": torch.stack(teacher_positions),
            "responses": responses,
        },
        meta_info=dict(batch.meta_info),
    )
    return teacher_batch, torch.tensor(effective_pi, dtype=torch.bool, device=input_ids.device)
