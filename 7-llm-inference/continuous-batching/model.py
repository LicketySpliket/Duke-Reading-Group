import math
import torch
import torch.nn as nn


def batched_attn_mask(seq_lengths: list[int], dev) -> torch.Tensor:
    """
    bool mask (N_PREFILL_TOKENS x N_PREFILL_TOKENS)
    mask[i, j] = True means allow token i to attend to token j.
    """
    n_prefill_tokens = sum(seq_lengths)
    seq_id = torch.empty(n_prefill_tokens, dtype=torch.long, device=dev)
    pos_in_seq = torch.empty(n_prefill_tokens, dtype=torch.long, device=dev)

    start = 0
    for sequence, seq_len in enumerate(seq_lengths):
        end = start + seq_len
        seq_id[start:end] = sequence
        pos_in_seq[start:end] = torch.arange(seq_len, device=dev)
        start = end

    same_sequence = seq_id[:, None] == seq_id[None, :]
    causal = pos_in_seq[:, None] >= pos_in_seq[None, :]
    return same_sequence & causal


class BatchedMHA(nn.Module):
    """
    Forward pass takes multiple sequences and handles all attention heads
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        d_head: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = d_head
        assert hidden_size == n_heads * d_head

        """
        Weights concatenated over attn heads.
        Number of columns: hidden_size = n_heads * d_head.
        e.g. for hidden_size = 768 and d_head = 64, wgts have height 768.
        Columns 0-63 form 768x64 matrix, wgts for the first attn head.
        Columns 64-127 form 768x64 matrix, wgts for the second attn head.
        etc.
        
        768  +----+----+----+----+----+----+----+----+----+----+----+----+
        rows | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 | H9 |H10 |H11 |H12 |
             |    |    |    |    |    |    |    |    |    |    |    |    |
             | :  | :  | :  | :  | :  | :  | :  | :  | :  | :  | :  | :  |
             | :  | :  | :  | :  | :  | :  | :  | :  | :  | :  | :  | :  |
             |    |    |    |    |    |    |    |    |    |    |    |    |
             +----+----+----+----+----+----+----+----+----+----+----+----+
               64   64   64   64   64   64   64   64   64   64   64   64
        """

        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        n_decode: int,
        decode_sequence_lengths: list[int],
        decode_batch_idxs: list[int],
        n_prefill: int,
        prefill_lengths: list[int],
        prefill_batch_idxs: list[int],
        k_cache: torch.Tensor,  # preallocated KV cache: MAX_BATCH x N_HEADS x MAX_SEQ_LEN x D_HEAD (does waste space)
        v_cache: torch.Tensor,  # preallocated KV cache: MAX_BATCH x N_HEADS x MAX_SEQ_LEN x D_HEAD (does waste space)
    ) -> torch.Tensor:
        """
        # x: N_TOKENS x HIDDEN_SIZE
        # First n_decode rows are tokens part of decode pass (decode pass only has one input token).
        # After that, prefill_lengths indicates how the remaining rows are divided among prefill passes.
        # k_cache and v_cache are both individual tensors divided into chunks of sizes decode_sequence_lengths,
        # with each chunk corresponding to one of n_decode sequences.

        decode_sequence_lengths: list of size n_decode, contains existing lengths of the decode sequences
        decode_batch_idxs: list of size n_decode, contains idxs of decode sequences to read/write their KV caches
        prefill_lengths: list of n_prefill, contains lengths of the prefill sequences (i.e. prompt lengths)
        prefill_batch_idxs: list of n_prefill, contains idxs of prefill sequences to write their KV caches
        """
        assert n_decode == len(decode_sequence_lengths)
        assert n_decode == len(decode_batch_idxs)
        assert n_prefill == len(prefill_lengths)
        assert n_prefill == len(prefill_batch_idxs)

        device = x.device
        n_tokens = x.shape[0]
        Q = self.w_q(x)  # N_TOKENS x HIDDEN_SIZE
        K = self.w_k(x)  # N_TOKENS x HIDDEN_SIZE
        V = self.w_v(x)  # N_TOKENS x HIDDEN_SIZE

        # last dimension (with size HIDDEN_SIZE) has the Q, K, and V values concatenated across heads (HIDDEN_SIZE = N_HEADS * D_HEAD)
        # reshape so that attn head is the first dimension and the last dimension has size D_HEAD
        Q = Q.view(n_tokens, self.n_heads, self.d_head)  # N_TOKENS x N_HEADS x D_HEAD
        K = K.view(n_tokens, self.n_heads, self.d_head)  # N_TOKENS x N_HEADS x D_HEAD
        V = V.view(n_tokens, self.n_heads, self.d_head)  # N_TOKENS x N_HEADS x D_HEAD

        # handle decode sequences and prefill sequences separately (assuming no chunked prefill in this demo)

        # ========================= start decode sequences =========================
        # decode: extract Q/K/V for decode sequences
        Q_decode = Q[:n_decode, :, :].unsqueeze(-2)  # N_DECODE x N_HEADS x 1 x D_HEAD
        K_decode = K[:n_decode, :, :].unsqueeze(-2)  # N_DECODE x N_HEADS x 1 x D_HEAD
        V_decode = V[:n_decode, :, :].unsqueeze(-2)  # N_DECODE x N_HEADS x 1 x D_HEAD

        # write K/V to cache before computing attn scores so that current token can attend to itself
        decode_batches = torch.tensor(
            decode_batch_idxs, device=k_cache.device, dtype=torch.long
        )
        decode_cols = torch.tensor(
            decode_sequence_lengths, device=k_cache.device, dtype=torch.long
        )
        decode_valid_lengths = decode_cols + 1
        k_cache[decode_batches, :, decode_cols, :] = K_decode.squeeze(2)
        v_cache[decode_batches, :, decode_cols, :] = V_decode.squeeze(2)

        # decode: get cached K and V
        # decode batches may not have indices 0...n_decode-1 in KV cache, need to index with decode_batch_idxs
        k_cache_decode = k_cache[
            decode_batch_idxs, :, :, :
        ]  # N_DECODE x N_HEADS x MAX_SEQ_LEN x D_HEAD
        v_cache_decode = v_cache[
            decode_batch_idxs, :, :, :
        ]  # N_DECODE x N_HEADS x MAX_SEQ_LEN x D_HEAD

        attn_scores_decode = torch.matmul(
            Q_decode, k_cache_decode.transpose(-2, -1)
        )  # N_DECODE x N_HEADS x 1 x MAX_SEQ_LEN

        # apply mask based on sequence length for each decode batch so no need to zero KV cache
        attn_mask_decode = torch.arange(
            k_cache_decode.shape[2], device=attn_scores_decode.device
        ).unsqueeze(0) < decode_valid_lengths.unsqueeze(1)  # [n_decode, max_seq_len]
        attn_mask_expanded = attn_mask_decode[:, None, None, :]
        attn_scores_decode = attn_scores_decode.masked_fill(
            ~attn_mask_expanded, float("-inf")
        )
        attn_scores_decode = attn_scores_decode / math.sqrt(self.d_head)
        attn_weights_decode = self.softmax(attn_scores_decode)

        weighted_vals_decode = torch.matmul(
            attn_weights_decode, v_cache_decode
        )  # N_DECODE x N_HEADS x 1 x D_HEAD
        out_decode = weighted_vals_decode.squeeze(2)  # N_DECODE x N_HEADS x D_HEAD
        # ========================= end decode sequences =========================

        # ========================= start prefill sequences =========================
        # N_PREFILL_TOKENS is total prefill tokens in all prefill sequences
        # On the other hand, N_PREFILL is number of sequences, each of which may have multiple tokens, adding up to N_PREFILL_TOKENS
        Q_prefill = Q[n_decode:, :, :]  # N_PREFILL_TOKENS x N_HEADS x D_HEAD
        K_prefill = K[n_decode:, :, :]  # N_PREFILL_TOKENS x N_HEADS x D_HEAD
        V_prefill = V[n_decode:, :, :]  # N_PREFILL_TOKENS x N_HEADS x D_HEAD

        # put head on the first dim
        Q_prefill = Q_prefill.permute(
            1, 0, 2
        ).contiguous()  # N_HEADS x N_PREFILL_TOKENS x D_HEAD
        K_prefill = K_prefill.permute(
            1, 0, 2
        ).contiguous()  # N_HEADS x N_PREFILL_TOKENS x D_HEAD
        V_prefill = V_prefill.permute(
            1, 0, 2
        ).contiguous()  # N_HEADS x N_PREFILL_TOKENS x D_HEAD

        # compute attn scores between all tokens
        attn_scores_prefill = torch.matmul(
            Q_prefill, K_prefill.transpose(-2, -1)
        )  # N_HEADS x N_PREFILL_TOKENS x N_PREFILL_TOKENS
        attn_scores_prefill = attn_scores_prefill / math.sqrt(self.d_head)

        # apply attn mask, need to enforce causal attn and no attn between different sequences
        # could create new dim to separate sequences, but this would require extra padding
        attn_mask_prefill = batched_attn_mask(prefill_lengths, dev=device)
        attn_scores_prefill = attn_scores_prefill.masked_fill(
            ~attn_mask_prefill, float("-inf")
        )

        attn_weights_prefill = self.softmax(
            attn_scores_prefill
        )  # N_HEADS x N_PREFILL_TOKENS x N_PREFILL_TOKENS
        weighted_vals_prefill = torch.matmul(
            attn_weights_prefill, V_prefill
        )  # N_HEADS x N_PREFILL_TOKENS x D_HEAD

        # move tokens to the first dimension
        out_prefill = weighted_vals_prefill.permute(
            1, 0, 2
        ).contiguous()  # N_PREFILL_TOKENS x N_HEADS x D_HEAD

        # write KV cache for prefill sequences
        if n_prefill > 0:
            prefill_batches = torch.tensor(
                prefill_batch_idxs, device=k_cache.device, dtype=torch.long
            )
            prefill_lengths_tensor = torch.tensor(
                prefill_lengths, device=k_cache.device, dtype=torch.long
            )

            # this creates something like this: [idx1, idx1, ... , idx1, idx2, ... , idx2, idx3, ... , idx3, ...]
            # where count of each index equals the corresponding prefill sequence length
            prefill_rows = torch.repeat_interleave(
                prefill_batches, prefill_lengths_tensor
            )
            
            # each row corresponds to a prefill sequence, and has indexes 0...(prefill sequence length - 1)
            prefill_cols = torch.cat(
                [
                    torch.arange(length, device=k_cache.device)
                    for length in prefill_lengths
                ]
            )

            k_cache[prefill_rows, :, prefill_cols, :] = K[n_decode:, :, :]
            v_cache[prefill_rows, :, prefill_cols, :] = V[n_decode:, :, :]

        # ========================= end prefill sequences =========================

        # combine decode and prefill
        out = torch.concat(
            [out_decode, out_prefill]
        )  # (N_DECODE + N_PREFILL_TOKENS) x N_HEADS x D_HEAD

        # concatenate along heads
        out = out.view(
            n_tokens, self.hidden_size
        )  # (N_DECODE + N_PREFILL_TOKENS) x HIDDEN_SIZE

        # final output projection
        out = self.w_o(out)  # (N_DECODE + N_PREFILL_TOKENS) x HIDDEN_SIZE
        return out


class SingleLayerTransformer(nn.Module):
    """
    Token embedding + one transformer layer for the continuous batching demo.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        n_heads: int,
        d_head: int,
        mlp_size: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = BatchedMHA(
            hidden_size=hidden_size,
            n_heads=n_heads,
            d_head=d_head,
        )
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size=hidden_size, mlp_size=mlp_size)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.unembed = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,
        n_decode: int,
        decode_sequence_lengths: list[int],
        decode_batch_idxs: list[int],
        n_prefill: int,
        prefill_lengths: list[int],
        prefill_batch_idxs: list[int],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedding(token_ids)  # (N_DECODE + N_PREFILL_TOKENS) x HIDDEN_SIZE
        attn_input = self.attn_norm(x)
        x = x + self.attn(
            x=attn_input,
            n_decode=n_decode,
            decode_sequence_lengths=decode_sequence_lengths,
            decode_batch_idxs=decode_batch_idxs,
            n_prefill=n_prefill,
            prefill_lengths=prefill_lengths,
            prefill_batch_idxs=prefill_batch_idxs,
            k_cache=k_cache,
            v_cache=v_cache,
        )
        x = x + self.mlp(self.mlp_norm(x))
        x = self.final_norm(x)
        return self.unembed(x)  # (N_DECODE + N_PREFILL_TOKENS) x VOCAB_SIZE


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x)


class MLP(nn.Module):
    def __init__(
        self,
        mlp_size: int,
        hidden_size: int,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            nn.GELU(),
            nn.Linear(mlp_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
