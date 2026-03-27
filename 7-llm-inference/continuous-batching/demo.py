from collections import deque
from dataclasses import dataclass
from enum import Enum

import torch

from model import SingleLayerTransformer


HIDDEN_SIZE = 32
N_HEADS = 4
D_HEAD = 8
MAX_BATCH = 3
MAX_SEQ_LEN = 12
VOCAB_SIZE = 128
MLP_SIZE = 64


class RequestState(Enum):
    QUEUED = 1
    PREFILL = 2
    DECODE = 3
    COMPLETED = 4


# NOTE: in real LLM, number of decode tokens for each would not be known beforehand, but for demo decode sequence lengths are predetermined


@dataclass
class Request:
    id: int
    arrival_step: int
    prompt_length: int
    decode_tokens: int
    slot: int | None = None
    cache_length: int = 0
    generated_tokens: int = 0
    last_token_id: int | None = None
    state: RequestState = RequestState.QUEUED


def init_requests() -> list[Request]:
    return [
        Request(0, arrival_step=0, prompt_length=3, decode_tokens=4),
        Request(1, arrival_step=0, prompt_length=5, decode_tokens=2),
        Request(2, arrival_step=1, prompt_length=4, decode_tokens=3),
        Request(3, arrival_step=2, prompt_length=2, decode_tokens=5),
        Request(4, arrival_step=4, prompt_length=6, decode_tokens=2),
    ]


def allocate_slot(free_slots: list[int]) -> int:
    if not free_slots:
        raise RuntimeError("No free slots available")
    return free_slots.pop(0)


def free_slot(
    request: Request,  # completed request
    free_slots: list[int],
):
    assert request.slot is not None

    free_slots.append(request.slot)
    free_slots.sort()
    request.slot = None


def run_demo():
    torch.manual_seed(0)

    # init model
    device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")

    model = SingleLayerTransformer(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        mlp_size=MLP_SIZE,
    ).to(device)

    # pre-allocate KV cache
    k_cache = torch.zeros(MAX_BATCH, N_HEADS, MAX_SEQ_LEN, D_HEAD, device=device)
    v_cache = torch.zeros(MAX_BATCH, N_HEADS, MAX_SEQ_LEN, D_HEAD, device=device)

    arrivals = deque(sorted(init_requests(), key=lambda r: r.arrival_step))
    request_queue: deque[Request] = deque()
    decode_requests: list[Request] = []
    free_slots = list(range(MAX_BATCH))

    step = 0
    while arrivals or request_queue or decode_requests:
        print("\n" + "=" * 40 + f"  step {step}  " + "=" * 40)
        while arrivals and arrivals[0].arrival_step <= step:
            request = arrivals.popleft()
            request_queue.append(request)
            print(
                f"step {step}: arrived {request.id} "
                f"(prompt length={request.prompt_length}, predetermined decode length={request.decode_tokens})"
            )

        # prepare forward pass input for in-flight decode requests
        decode_token_ids: list[int] = []
        decode_lengths: list[int] = []
        decode_batch_idxs: list[int] = []
        decode_ids: list[int] = []

        for request in decode_requests:
            assert request.slot is not None
            if request.last_token_id is None:
                raise RuntimeError(
                    f"Decode request {request.id} is missing a last generated token"
                )
            # use most recent output token as input
            decode_token_ids.append(request.last_token_id)
            decode_lengths.append(request.cache_length)
            decode_batch_idxs.append(request.slot)
            decode_ids.append(request.id)

        # allocate slots for new prefill requests
        prefill_requests: list[Request] = []
        while request_queue and free_slots:
            request = request_queue.popleft()
            request.slot = allocate_slot(free_slots)
            request.state = RequestState.PREFILL
            prefill_requests.append(request)

        # prepare forward pass input for new prefill requests
        prefill_token_ids: list[torch.Tensor] = []
        prefill_lengths: list[int] = []
        prefill_batch_idxs: list[int] = []
        prefill_ids: list[int] = []
        for request in prefill_requests:
            assert request.slot is not None

            # just use some arbitrary tokens for the prompt
            token_ids = (
                torch.arange(request.prompt_length, device=device) + step
            ) % VOCAB_SIZE
            prefill_token_ids.append(token_ids)
            prefill_lengths.append(request.prompt_length)
            prefill_batch_idxs.append(request.slot)
            prefill_ids.append(request.id)

        n_decode = len(decode_token_ids)
        n_prefill = len(prefill_token_ids)
        if n_decode == 0 and n_prefill == 0:
            print(f"step {step}: idle")
            step += 1
            continue

        # concatenate decode and prefill token ids into one big input tensor
        token_id_components: list[torch.Tensor] = []
        if decode_token_ids:
            token_id_components.append(torch.tensor(decode_token_ids, device=device))
        if prefill_token_ids:
            token_id_components.append(torch.cat(prefill_token_ids, dim=0))
        token_ids = torch.cat(token_id_components, dim=0)

        # run forward pass through model
        # output contains logit vectors for each input token
        output = model(
            token_ids=token_ids,
            n_decode=n_decode,
            decode_sequence_lengths=decode_lengths,
            decode_batch_idxs=decode_batch_idxs,
            n_prefill=n_prefill,
            prefill_lengths=prefill_lengths,
            prefill_batch_idxs=prefill_batch_idxs,
            k_cache=k_cache,
            v_cache=v_cache,
        )  # (N_DECODE + N_PREFILL_TOKENS) x VOCAB_SIZE

        print(
            f"step {step}: batched {n_decode} decode + {n_prefill} prefill -> "
            f"output {tuple(output.shape)}"
        )
        if decode_ids:
            print(f"  decode: {decode_ids}")
        if prefill_ids:
            print(f"  prefill: {prefill_ids}")

        # get token id with max logit for each input token
        next_token_ids = output.argmax(dim=-1)  # (N_DECODE + N_PREFILL_TOKENS) x 1

        # store most recent output token id for decode requests
        for row_idx, request in enumerate(decode_requests):
            request.last_token_id = int(next_token_ids[row_idx].item())

        # for prefill requests, every input token will have a corresponding output token, but we only want the output token for the last input token
        prefill_offset = n_decode
        running_prefill_tokens = 0
        for request in prefill_requests:
            last_prefill_row = (
                prefill_offset + running_prefill_tokens + request.prompt_length - 1
            )
            request.last_token_id = int(next_token_ids[last_prefill_row].item())
            running_prefill_tokens += request.prompt_length

        # check which decode requests done and which still in progress
        still_active: list[Request] = []
        completed: list[int] = []
        for request in decode_requests:
            request.cache_length += 1
            request.generated_tokens += 1
            if request.generated_tokens >= request.decode_tokens:
                completed.append(request.id)
                request.state = RequestState.COMPLETED
                free_slot(request, free_slots)
            else:
                still_active.append(request)
        decode_requests = still_active

        # move completed prefill stages to decode
        for request in prefill_requests:
            request.cache_length = request.prompt_length
            request.state = RequestState.DECODE
            decode_requests.append(request)

        if completed:
            print(f"  completed: {completed}")

        decode_info = [
            f"request={request.id} (slot={request.slot}, cache_len={request.cache_length}, "
            f"generated={request.generated_tokens}/{request.decode_tokens})"
            for request in decode_requests
        ]
        print(f"  active decode sequences: {decode_info}")

        step += 1

    print(f"finished after {step} steps")


if __name__ == "__main__":
    run_demo()
