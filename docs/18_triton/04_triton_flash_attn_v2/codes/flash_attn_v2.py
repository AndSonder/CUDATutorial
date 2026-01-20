"""
Flash Attention V2 - Triton 实现

参考论文: FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
https://arxiv.org/abs/2307.08691

本实现展示了如何用 Triton 实现 Flash Attention V2，相比 V1 有更好的并行性。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_v2_kernel(
    q_ptr, k_ptr, v_ptr,          # 输入指针
    o_ptr,                        # 输出指针
    stride_qz, stride_qh, stride_qm, stride_qk,  # Q 的 strides
    stride_kz, stride_kh, stride_kn, stride_kk,  # K 的 strides
    stride_vz, stride_vh, stride_vn, stride_vk,  # V 的 strides
    stride_oz, stride_oh, stride_om, stride_ok,  # O 的 strides
    batch, heads,                 # batch size 和 head 数
    seqlen_q, seqlen_k,           # Q 和 K 的序列长度
    head_dim,                     # head 维度
    BLOCK_M: tl.constexpr,        # Q block 大小
    BLOCK_N: tl.constexpr,        # K/V block 大小
):
    """
    Flash Attention V2 Kernel

    每个 Program 处理一个 Q block（大小 BLOCK_M x head_dim），
    遍历所有 K/V block（每个大小 BLOCK_N x head_dim），
    使用在线 Softmax 增量更新最终结果。
    """
    # === Part 1: Program ID 和指针计算 ===

    # 计算 program_id 对应的 (batch, head, q_block)
    n_q_blocks = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    pid = tl.program_id(axis=0)

    q_block_id = pid % n_q_blocks
    pid_ = pid // n_q_blocks
    head_id = pid_ % heads
    batch_id = pid_ // heads

    # Q 的起始地址（跳过当前 batch 和 head）
    q_ptr += batch_id * stride_qz + head_id * stride_qh

    # Q block 的行偏移（query 位置）
    # m_mask: 处理边界情况，当 seqlen_q 不是 BLOCK_M 的倍数时
    m_mask = q_block_id * BLOCK_M + tl.arange(0, BLOCK_M) < seqlen_q
    q_ptr += (q_block_id * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_qm

    # 加载 Q block
    # shape: [BLOCK_M, head_dim]
    q = tl.load(
        q_ptr + tl.arange(0, head_dim)[None, :] * stride_qk,
        mask=m_mask[:, None],
        other=0.0
    )

    # === Part 2: 初始化累加器 ===

    # 在线 Softmax 的状态
    # m_i: 当前见到的最大值
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # l_i: 归一化系数（初始化为 1，这样第一次更新时 alpha 计算正确）
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    # acc: 输出累加器
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # 缩放因子
    sm_scale = 1.0 / (head_dim ** 0.5)

    # === Part 3: 外层循环 - 遍历 K/V block ===

    # K/V 的起始地址（跳过当前 batch 和 head）
    k_ptr += batch_id * stride_kz + head_id * stride_kh
    v_ptr += batch_id * stride_vz + head_id * stride_vh

    # 遍历所有 K/V block
    lo = 0
    hi = (seqlen_k + BLOCK_N - 1) // BLOCK_N

    for k_block_id in range(lo, hi):
        # K block 的列偏移（key 位置）
        n_mask = k_block_id * BLOCK_N + tl.arange(0, BLOCK_N) < seqlen_k

        # 加载 K block
        # shape: [head_dim, BLOCK_N]
        k_ptr_offset = (k_block_id * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * stride_kn
        k = tl.load(
            k_ptr + k_ptr_offset + tl.arange(0, head_dim)[:, None] * stride_kk,
            mask=n_mask[None, :],
            other=0.0
        )

        # 加载 V block
        # shape: [head_dim, BLOCK_N]
        v_ptr_offset = (k_block_id * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * stride_vn
        v = tl.load(
            v_ptr + v_ptr_offset + tl.arange(0, head_dim)[:, None] * stride_vk,
            mask=n_mask[None, :],
            other=0.0
        )

        # === Part 4: 在线 Softmax 更新 ===

        # 计算 Q @ K^T，然后应用缩放
        # shape: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k) * sm_scale

        # 计算当前 block 内的最大值
        m_block = tl.max(qk, axis=1)  # [BLOCK_M]

        # 更新全局最大值
        m_i_new = tl.maximum(m_i, m_block)

        # 计算 alpha = exp(m_old - m_new)
        alpha = tl.exp(m_i - m_i_new)

        # 计算 p = exp(S - m_new)
        p = tl.exp(qk - m_i_new[:, None])

        # 更新归一化系数
        l_i_new = alpha * l_i + tl.sum(p, axis=1)

        # 更新输出累加器
        # 注意：p 需要转换为 float16 才能和 v 做 dot
        acc = alpha * acc + tl.dot(p.to(q.dtype), v)

        # 更新状态
        m_i, l_i = m_i_new, l_i_new

    # === Part 5: 写回结果 ===

    # 最终归一化
    acc = acc / l_i[:, None]

    # 写回输出
    o_ptr += batch_id * stride_oz + head_id * stride_oh
    o_ptr += (q_block_id * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_om
    tl.store(
        o_ptr + tl.arange(0, head_dim)[None, :] * stride_ok,
        acc,
        mask=m_mask[:, None]
    )


def flash_attn_v2(q, k, v):
    """
    Flash Attention V2 的 Python 包装函数

    Args:
        q: [batch, heads, seqlen_q, head_dim]
        k: [batch, heads, seqlen_k, head_dim]
        v: [batch, heads, seqlen_k, head_dim]

    Returns:
        o: [batch, heads, seqlen_q, head_dim]
    """
    batch, heads, seqlen_q, head_dim = q.shape
    seqlen_k = k.shape[2]

    # 计算网格大小
    n_q_blocks = (seqlen_q + 127) // 128
    grid = (batch * heads * n_q_blocks,)

    # 输出张量
    o = torch.empty_like(q)

    # 启动 kernel
    flash_attn_v2_kernel[grid](
        q, k, v, o,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        batch, heads, seqlen_q, seqlen_k, head_dim,
        BLOCK_M=128, BLOCK_N=128,
    )

    return o


def test_correctness():
    """验证实现的正确性"""
    import torch.nn.functional as F

    print("=" * 60)
    print("Flash Attention V2 正确性验证")
    print("=" * 60)

    # 测试不同配置
    test_cases = [
        (2, 4, 256, 256, 64),
        (1, 8, 512, 512, 64),
        (2, 4, 1024, 1024, 64),
    ]

    for batch, heads, seqlen_q, seqlen_k, head_dim in test_cases:
        print(f"\n测试配置: batch={batch}, heads={heads}, "
              f"seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, head_dim={head_dim}")

        # 生成随机输入
        q = torch.randn(batch, heads, seqlen_q, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch, heads, seqlen_k, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch, heads, seqlen_k, head_dim, device='cuda', dtype=torch.float16)

        # Triton 实现
        o_triton = flash_attn_v2(q, k, v)

        # PyTorch 实现
        o_torch = F.scaled_dot_product_attention(q, k, v)

        # 验证
        max_error = torch.max(torch.abs(o_triton - o_torch)).item()
        mean_error = torch.mean(torch.abs(o_triton - o_torch)).item()

        print(f"  Max error: {max_error:.6f}")
        print(f"  Mean error: {mean_error:.6f}")

        if torch.allclose(o_triton, o_torch, atol=1e-2):
            print("  ✓ 通过")
        else:
            print("  ✗ 失败")

    print("\n" + "=" * 60)


def benchmark():
    """性能基准测试"""
    import triton.testing

    print("\n" + "=" * 60)
    print("Flash Attention V2 性能基准测试")
    print("=" * 60)

    # 测试不同序列长度
    test_cases = [
        (1, 8, 512, 64),
        (1, 8, 1024, 64),
        (1, 8, 2048, 64),
    ]

    print(f"\n{'配置':<30} {'时间(ms)':<12} {'吞吐量(GB/s)'}")
    print("-" * 60)

    for batch, heads, seqlen, head_dim in test_cases:
        config = f"{batch}x{heads}x{seqlen}x{head_dim}"

        q = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch, heads, seqlen, head_dim, device='cuda', dtype=torch.float16)

        # 预热
        for _ in range(10):
            _ = flash_attn_v2(q, k, v)
        torch.cuda.synchronize()

        # 计时
        time_ms = triton.testing.do_bench(lambda: flash_attn_v2(q, k, v), rep=100)

        # 计算吞吐量
        total_bytes = (q.numel() + k.numel() + v.numel() + q.numel()) * 2  # fp16
        throughput = (total_bytes / time_ms * 1e-3) / 1e9  # GB/s

        print(f"{config:<30} {time_ms:<12.3f} {throughput:.1f}")

    print("=" * 60)


if __name__ == "__main__":
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过测试")
        exit(0)

    # 运行正确性验证
    test_correctness()

    # 运行性能基准测试
    benchmark()
