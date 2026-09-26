# Exact SGLang bodies. Import isolation only.
import triton
import triton.language as tl

@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    a_desc,
    b_ptr,
    b_desc,
    bias_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    add_mask_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bias_e,
    stride_bias_n,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    even_Ks: tl.constexpr,
    c_sorted: tl.constexpr,
    filter_expert: tl.constexpr,
    swap_ab: tl.constexpr,
    FUSE_ADD_TO_OUTPUT: tl.constexpr,
    MASK_OUTPUT: tl.constexpr,
    FUSE_SUM_ALL_REDUCE: tl.constexpr,
    LORA_PRESERVE_BASE: tl.constexpr,
    ROUTER_TOPK: tl.constexpr,
    FUSE_SWIGLU: tl.constexpr = False,
    USE_GDC: tl.constexpr = False,
    GDC_EARLY: tl.constexpr = False,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        if GDC_EARLY:
            tl.extra.cuda.gdc_launch_dependents()

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts_i32 = tl.load(expert_ids_ptr + pid_m)
    off_experts = off_experts_i32.to(tl.int64)

    if filter_expert and off_experts == -1:
        if FUSE_SWIGLU:
            # C is the half-width post-activation buffer here. Rows owned by a
            # filtered expert are never read (the down-GEMM CTA for this block
            # early-exits before loading A), and an N-wide zero store would run
            # past the row end into a neighboring token's data.
            return
        if not FUSE_ADD_TO_OUTPUT and not (FUSE_SUM_ALL_REDUCE and LORA_PRESERVE_BASE):
            # Write zeros only when this kernel owns the full output; the experimental LoRA
            # add path (LORA_PRESERVE_BASE) keeps the base output from the prior MoE kernel.
            write_zeros_to_output(
                c_ptr,
                stride_cm,
                stride_cn,
                pid_n,
                N,
                offs_token,
                token_mask,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                compute_type,
            )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if a_desc is not None:
        assert use_fp8_w8a8 and group_n > 0 and group_k > 0
        start_offs_m = pid_m * BLOCK_SIZE_M
    else:
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )

    if b_desc is not None:
        start_offs_n = pid_n * BLOCK_SIZE_N
    else:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )

    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + off_experts * stride_bias_e + offs_bn[None, :] * stride_bias_n
        )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            if a_desc is not None:
                a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
            else:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            if BLOCK_SIZE_N > group_n:
                offs_bsn = offs_bn // group_n
            else:
                offs_bsn = pid_n * BLOCK_SIZE_N // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if swap_ab:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if a_desc is not None:
            a = a_desc.load([start_offs_m, k_start])
        elif even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )

        if b_desc is not None:
            b = (
                b_desc.load([off_experts_i32, start_offs_n, k_start])
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .T
            )
        elif even_Ks:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                if swap_ab:
                    a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                    a_scale, b_scale = b_scale, a_scale
                if BLOCK_SIZE_N > group_n:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                else:
                    accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
            else:
                if use_fp8_w8a8:
                    if swap_ab:
                        a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        if a_desc is None:
            a_ptrs += BLOCK_SIZE_K * stride_ak
        if b_desc is None:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if swap_ab:
        accumulator = tl.trans(accumulator, (1, 0))

    if use_int8_w8a16:
        accumulator *= b_scale
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k == 0 or group_n == 0:
            accumulator *= a_scale * b_scale

    if bias_ptr is not None:
        accumulator += bias

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if FUSE_SWIGLU:
        # W13 rows were interleaved at load time, so gate/up of the same
        # intermediate channel sit in adjacent (even, odd) columns of this
        # tile; silu(gate) * up is applied in-register and only the
        # half-width activation is stored, eliminating intermediate_cache1
        # and the standalone activation launch.
        #
        # The asm below is the bit-parity contract with the `silu_and_mul` this
        # replaces, not an optimization: that kernel uses the fast-math
        # intrinsics `__fdividef(x, 1 + __expf(-x))`, while Triton's operators
        # lower to accurate expf and IEEE `div.rn`. The 1-2 ULP gap is enough to
        # flip the stored bf16 (8 mantissa bits) on many inputs. The final
        # multiply needs no asm -- plain fp32 multiply matches `mul.ftz.f32`
        # except on denormals. Silu stays fp32 until the store, as in the
        # reference; rounding it to bf16 first double-rounds and diverges.
        acc_pairs = tl.reshape(accumulator, (BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
        gate_b, up_b = tl.split(acc_pairs)
        gate_f = gate_b.to(tl.float32)
        # __expf(-x) == ex2.approx(x * -log2(e)); folding the negation into
        # the constant (0fBFB8AA3B == -log2(e)) flips the sign exactly.
        exp_neg = tl.inline_asm_elementwise(
            "{ mul.ftz.f32 $0, $1, 0fBFB8AA3B; ex2.approx.ftz.f32 $0, $0; }",
            "=f,f",
            [gate_f],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        silu_f = tl.inline_asm_elementwise(
            "div.approx.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            [gate_f, 1.0 + exp_neg],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        out_act = (silu_f * up_b.to(tl.float32)).to(compute_type)
        offs_half = pid_n * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)
        if c_sorted:
            c_ptrs = (
                c_ptr
                + stride_cm * offs_token_id[:, None]
                + stride_cn * offs_half[None, :]
            )
        else:
            c_ptrs = (
                c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_half[None, :]
            )
        c_mask = token_mask[:, None] & (offs_half[None, :] < N // 2)
        tl.store(c_ptrs, out_act, mask=c_mask)
    elif FUSE_ADD_TO_OUTPUT:
        # Accumulate into existing output with per-token mask.
        offs_token_out = offs_token // ROUTER_TOPK
        add_mask = tl.load(add_mask_ptr + offs_token_out, mask=token_mask, other=False)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & add_mask[:, None] & (offs_cn[None, :] < N)
        existing = tl.load(c_ptrs, mask=c_mask, other=0.0)
        tl.store(c_ptrs, existing + accumulator, mask=c_mask)
    # ===== TO BE REFACTORED ====
    elif MASK_OUTPUT:
        # Store a fresh output while zeroing rows whose request has no active LoRA.
        offs_token_out = offs_token // ROUTER_TOPK
        output_mask = tl.load(
            add_mask_ptr + offs_token_out, mask=token_mask, other=False
        )
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        accumulator = tl.where(output_mask[:, None], accumulator, 0.0)
        tl.store(c_ptrs, accumulator, mask=c_mask)
    # ===== END TO BE REFACTORED ====
    elif FUSE_SUM_ALL_REDUCE:
        offs_token_out = offs_token // ROUTER_TOPK
        c_ptrs = (
            c_ptr + stride_cm * offs_token_out[:, None] + stride_cn * offs_cn[None, :]
        )
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)
    else:
        if c_sorted:
            c_ptrs = (
                c_ptr
                + stride_cm * offs_token_id[:, None]
                + stride_cn * offs_cn[None, :]
            )
        else:
            c_ptrs = (
                c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
            )
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    if USE_GDC and not GDC_EARLY:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _moe_align_small_numel_kernel(
    topk_ids_ptr,  # [numel] int, flattened (token, slot) expert ids, -1 = filtered
    sorted_token_ids_ptr,  # [max_num_tokens_padded] int32
    expert_ids_ptr,  # [max_num_m_blocks] int32
    num_tokens_post_pad_ptr,  # [1] int32
    num_experts,  # E + 1 (the "+1 offset" convention's bucket count)
    block_size,
    numel,
    NP: tl.constexpr,  # power-of-2 >= numel
    NB: tl.constexpr,  # power-of-2 >= max blocks used
    USE_GDC: tl.constexpr = False,
):
    """Single-CTA moe_align for tiny batches with MANY experts.

    Everything works on the PAIR axis ([NP, NP] pairwise comparisons plus a
    rank-0 representative per bucket) -- an expert-axis formulation (histogram
    / cumsum over ~1k buckets) is ~3x more single-SM work and measured slower
    than the two-kernel path it replaces.

    Reference semantics reproduced:
    - "+1 offset" convention: expert -1 (EP-filtered) maps to bucket 0 and its
      blocks get expert_ids = -1 (skipped by fused_moe's filter_expert);
    - every bucket is padded to a block_size multiple, offsets in bucket order;
    - pad slots inside [0, num_tokens_post_pad) hold `numel`.

    Intended deviations, both invisible to fused_moe:
    - intra-bucket order is stable in pair index (the reference's atomicAdd
      order is scheduling-dependent; every pair writes its own output row);
    - sorted_token_ids beyond num_tokens_post_pad is left unwritten (the
      reference pre-fills the whole buffer; consumers only read below the
      published total).
    """
    if USE_GDC:
        # Consumer side of the router top-k that produced topk_ids.
        tl.extra.cuda.gdc_wait()

    offs_p = tl.arange(0, NP)
    mask_p = offs_p < numel
    ids = tl.load(topk_ids_ptr + offs_p, mask=mask_p, other=-2)
    # Padded lanes get an out-of-range bucket and are masked out everywhere.
    bucket = tl.where(mask_p, (ids + 1).to(tl.int32), num_experts)

    # Pairwise stats: stable rank within the bucket and bucket population.
    same = (bucket[None, :] == bucket[:, None]) & mask_p[None, :] & mask_p[:, None]
    earlier = offs_p[None, :] < offs_p[:, None]
    rank = tl.sum((same & earlier).to(tl.int32), axis=1)  # [NP]
    cnt = tl.sum(same.to(tl.int32), axis=1)  # [NP], own-bucket population
    padded_cnt = ((cnt + block_size - 1) // block_size) * block_size
    is_rep = (rank == 0) & mask_p  # one representative pair per bucket

    # Bucket-ordered exclusive offsets: sum the padded counts of every
    # representative with a strictly smaller bucket id.
    smaller_rep = (bucket[None, :] < bucket[:, None]) & is_rep[None, :]
    excl = tl.sum(smaller_rep.to(tl.int32) * padded_cnt[None, :], axis=1)  # [NP]

    total = tl.sum(tl.where(is_rep, padded_cnt, 0), axis=0)
    tl.store(num_tokens_post_pad_ptr, total.to(tl.int32))

    # expert_ids per used block: representative r owns blocks
    # [excl[r], excl[r] + padded_cnt[r]); the written id is bucket - 1
    # (bucket 0 = filtered -> -1).
    offs_b = tl.arange(0, NB)
    block_start = offs_b * block_size
    in_range = (
        (block_start[:, None] >= excl[None, :])
        & (block_start[:, None] < (excl + padded_cnt)[None, :])
        & is_rep[None, :]
    )
    eid = tl.sum(in_range.to(tl.int32) * (bucket[None, :] - 1), axis=1)
    tl.store(expert_ids_ptr + offs_b, eid.to(tl.int32), mask=block_start < total)

    # Fill the used region's pad slots with `numel`, then scatter the real
    # pair indices over them. The barrier is required: fill and scatter run on
    # different warps of this CTA, and a scatter store must not be overtaken
    # by a later-warp fill store to the same address.
    n_fill = (total + NP - 1) // NP
    for it in range(n_fill):
        f_offs = it * NP + offs_p
        tl.store(
            sorted_token_ids_ptr + f_offs,
            tl.full([NP], 0, tl.int32) + numel,
            mask=f_offs < total,
        )
    tl.debug_barrier()
    pos = excl + rank
    tl.store(sorted_token_ids_ptr + pos, offs_p.to(tl.int32), mask=mask_p)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()

