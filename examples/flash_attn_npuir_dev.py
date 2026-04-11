# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T


seq_len = 512
dim = 128
batch = 2
heads_q = 8
heads_kv = 2  # GQA: each KV head is shared by 4 Q heads

torch.npu.set_device(0)

@tilelang.jit(out_idx=[-1], target="npuir")
def online_flash_attention(block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    shape_q = [batch, seq_len, heads_q, dim]
    shape_k = [batch, seq_len, heads_kv, dim]
    shape_v = [batch, seq_len, heads_kv, dim]
    shape_o = [batch, seq_len, heads_q, dim]
    block_m = block_M
    block_n = block_N
    num_seq_blocks = T.ceildiv(seq_len, block_m)
    kv_group_size = heads_q // heads_kv

    @T.prim_func
    def flash_attention(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        Output: T.Tensor(shape_o, dtype),
    ):
        with T.Kernel(batch * heads_q * num_seq_blocks, is_npu=True) as (cid, _):
            seq_block_idx = cid % num_seq_blocks
            q_head_idx = (cid // num_seq_blocks) % heads_q
            batch_idx = cid // (heads_q * num_seq_blocks)
            kv_head_idx = q_head_idx // kv_group_size
            offset = seq_block_idx * block_m
            Q_shared = T.alloc_shared([block_m, dim], dtype)
            T.copy(Q[batch_idx, offset : offset + block_m, q_head_idx, 0 : dim], Q_shared)

            K_shared = T.alloc_shared([block_n, dim], dtype)
            V_shared = T.alloc_shared([block_n, dim], dtype)
            scores = T.alloc_fragment([block_m, block_n], accum_dtype)
            scores_cast = T.alloc_fragment([block_m, block_n], dtype)
            correction = T.alloc_fragment([block_m,1], accum_dtype)
            local_max = T.alloc_fragment([block_m,1], accum_dtype)
            local_sum = T.alloc_fragment([block_m,1], accum_dtype)
            acc_m = T.alloc_fragment([block_m, 1], accum_dtype)
            acc_l = T.alloc_fragment([block_m, 1], accum_dtype)
            acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
            tmp = T.alloc_fragment([block_m, block_n], accum_dtype)
            tmp1 = T.alloc_fragment([block_m,1], accum_dtype)
            new_max = T.alloc_fragment([block_m,1], accum_dtype)
            scales = T.alloc_fragment([block_m, block_n], accum_dtype)

            value_zero = 0
            scale = (1.0 / dim)**0.5
            value_min = -T.infinity(accum_dtype)
            T.vbrc(value_zero, acc_o)
            T.vbrc(value_zero, acc_l)
            T.vbrc(value_min, acc_m)
            T.vbrc(scale, scales)

            for k in T.Pipelined(T.ceildiv(seq_len, block_n), num_stages=2):

                # cube
                T.copy(K[batch_idx, k * block_n : (k + 1) * block_n, kv_head_idx, 0 : dim], K_shared)
                T.gemm(Q_shared, K_shared, scores, initC=True, b_transpose=True)

                # vec
                T.vmul(scores, scales, scores)
                T.reduce_max(scores, local_max, dim=1)
                T.vmax(acc_m, local_max, new_max)
                T.vsub(acc_m, new_max ,tmp1)
                T.vexp(tmp1, correction)
                #scores for current loop
                T.vsub(scores, new_max, tmp)
                T.vexp(tmp, scores)
                T.reduce_sum(scores, local_sum, dim=1)
                T.vmul(acc_l, correction, acc_l)
                T.vadd(acc_l, local_sum, acc_l)
                T.vmul(acc_o, correction, acc_o)
                T.vcast(scores, scores_cast, round_mode="rint")
                #copy new_max to acc_m
                T.vbrc(value_zero, tmp1)
                T.vadd(tmp1, new_max, acc_m)

                # cube
                T.copy(V[batch_idx, k * block_n : (k + 1) * block_n, kv_head_idx, 0 : dim], V_shared)
                T.gemm(scores_cast, V_shared, acc_o, initC=False)

            T.vdiv(acc_o, acc_l, acc_o)
            O_cast = T.alloc_shared([block_m, dim], dtype)
            T.vcast(acc_o, O_cast, round_mode="rint")
            real_m = T.min(block_m, seq_len - seq_block_idx * block_m)
            T.copy(O_cast, Output[batch_idx, seq_block_idx * block_m : seq_block_idx * block_m + real_m, q_head_idx, 0 : dim])

    return flash_attention

def main():
    # In the futrue, Developer mode and Expert Mode will transition smoothly without
    # requiring explicit declarations.
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    kernel = online_flash_attention(64, 64, 32)

    q = torch.randn((batch, seq_len, heads_q, dim), dtype=torch.float16).npu()
    k = torch.randn((batch, seq_len, heads_kv, dim), dtype=torch.float16).npu()
    v = torch.randn((batch, seq_len, heads_kv, dim), dtype=torch.float16).npu()

    output = kernel(q, k, v)

    scale = (1.0 / dim)**0.5
    # BSHD -> BHSD for reference computation
    q_bh = q.transpose(1, 2)  # [batch, heads_q, seq_len, dim]
    k_bh = k.transpose(1, 2)  # [batch, heads_kv, seq_len, dim]
    v_bh = v.transpose(1, 2)  # [batch, heads_kv, seq_len, dim]
    
    # GQA: repeat KV heads to match Q heads
    kv_group_size = heads_q // heads_kv
    k_bh = k_bh.repeat_interleave(kv_group_size, dim=1)  # [batch, heads_q, seq_len, dim]
    v_bh = v_bh.repeat_interleave(kv_group_size, dim=1)  # [batch, heads_q, seq_len, dim]
    
    ref_output = torch.nn.functional.softmax(
        (q_bh @ k_bh.transpose(-2, -1)).to(torch.float32) * scale, dim=-1).to(torch.float16) @ v_bh
    ref_output = ref_output.transpose(1, 2)  # [batch, seq_len, heads_q, dim]
    print("output:")
    print(output)
    print("ref_output:")
    print(ref_output)
    torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
    print("All check passed.")

if __name__ == "__main__":
    main()