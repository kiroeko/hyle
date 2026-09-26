# Original SGLang, Apache-2.0. Exact bodies; import isolation only.
import triton
import triton.language as tl

@triton.jit
def _per_token_group_quant_8bit(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Avoid to divide zero
    eps,
    # Information for float8
    bit8_min,
    bit8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.

    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / bit8_max
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_8bit_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    bit8_min,
    bit8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id.to(tl.int64) * group_size
    y_q_ptr += g_id.to(tl.int64) * group_size

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    y_s_ptr += scale_col * y_s_col_stride + scale_row

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / bit8_max
    if SCALE_UE8M0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.abs(y_s))))
    y_q = tl.clamp(y / y_s, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _w8a8_block_fp8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and store the result in output
    tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if needs_masking:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
