import math


def get_adaptive_hyperparameters(
    bsz,
    verification_capacity,
    max_draft_token_length,
    max_draft_k,
    max_verification_num,
    min_draft_token_length,
    draft_token_length_c,
):
    if bsz <= 0:
        raise ValueError("bsz must be positive when computing adaptive hyperparameters.")

    verification_num = min(math.floor(verification_capacity / bsz), max_verification_num)

    if verification_num <= 1:
        raise ValueError(
            "verification_capacity is too small for the active generation batch: "
            f"floor({verification_capacity}/{bsz})={verification_num}. "
            "Need at least 2 verification slots per active sequence; increase "
            "--verification_capacity or reduce --batch_size/--repeated_generate_nums."
        )

    draft_token_length = min(
        math.floor(math.log2(verification_num / draft_token_length_c)),
        max_draft_token_length,
    )
    draft_token_length = max(draft_token_length, min_draft_token_length)

    draft_k = min(verification_num - 1, max_draft_k)
    draft_total_token = verification_num - 1

    return draft_token_length, draft_k, draft_total_token
