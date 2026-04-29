module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  // ================================================================
  // AIC (Cube) 侧: 负责矩阵乘 (S = Q×K^T, O = P×V)
  // 与 AIV (Vector) 通过 workspace 和同步标志通信
  // ================================================================
  func.func @FlashAttnExp_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2%arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.add %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: memref<?xf16, #hivm.address_space<gm>>, %arg8: memref<?xf16, #hivm.ad, %arg9: memref<?xf16, #hivm.address_space<gm>>, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i3yncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm. #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c31_i32 = arith.constant 31 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c128_i32 = arith.constant 128 : i32
    %c3072_i32 = arith.constant 3072 : i32
    %c32_i32 = arith.constant 32 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [128, 1] : memref<?xf16, ace<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [128, 1] : memref<?xf16space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    // workspace_o: [24][2][128][128]  (多缓冲深度2)
    %reinterpret_cast_1 = memref.reinterpret_cast %arg9 to offset: [0], sizes: [24, 2, 128, 128], strides: [32768, 16384, 12xf16, #hivm.address_space<gm>> to memref<24x2x128x128xf16, strided<[32768, 16384, 128, 1]>, #hivm.address_space<gm>>
    // workspace_s: [24][2][128][256]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [24, 2, 128, 256], strides: [65536, 32768, 25xf16, #hivm.address_space<gm>> to memref<24x2x128x256xf16, strided<[65536, 32768, 256, 1]>, #hivm.address_space<gm>>
    // workspace_p: [24][2][128][256]
    %reinterpret_cast_3 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [24, 2, 128, 256], strides: [65536, 32768, 25xf16, #hivm.address_space<gm>> to memref<24x2x128x256xf16, strided<[65536, 32768, 256, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [128, 1] : memref<?xf16space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32

    // ================================================================
    // c_init_flags(): 初始化标志 4,5 为“空”（允许 Vector 写入 P）
    // flag = 4 + i  →  i=0 → Flag4, i=1 → Flag5
    // ================================================================
    scf.for %arg16 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %6 = arith.addi %arg16, %c4_i32 : i32      // flag = 4,5
      %7 = arith.extsi %6 : i32 to i64
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %7 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      // Set Flag4 (C→V: 释放 P 槽0), Set Flag5 (C→V: 释放 P 槽1)
    }
    %2 = arith.subi %c87_i32, %1 : i32
    %3 = arith.divsi %2, %c24_i32 : i32
    %4 = arith.muli %3, %c32_i32 : i32
    %5 = arith.addi %4, %c1_i32 : i32

    // ================================================================
    // 主软件流水循环: stream_id = %arg16
    // ================================================================
    scf.for %arg16 = %c0_i32 to %5 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>> // L1 Q
      %alloc_5 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>> // L1 P
      %alloc_6 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>> // L1 KV
      %alloc_7 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>   // L0C
      %6 = arith.subi %c87_i32, %1 : i32
      %7 = arith.divsi %6, %c24_i32 : i32
      %8 = arith.muli %7, %c32_i32 : i32
      %9 = arith.cmpi slt, %arg16, %8 : i32

      // ========== C1 阶段: 当 stream_id < num_loops_total 时执行 ==========
      scf.if %9 {
        %11 = arith.remsi %arg16, %c32_i32 : i32        // 当前 K/V 块索引 (block_id)
        %12 = arith.cmpi eq, %11, %c0_i32 : i32
        // 仅在第一个 K/V 块加载 Q
        scf.if %12 {
          %27 = arith.divsi %arg16, %c32_i32 : i32
          %28 = arith.muli %27, %c3072_i32 : i32
          %29 = arith.muli %1, %c128_i32 : i32
          %30 = arith.addi %28, %29 : i32
          %31 = arith.index_cast %30 : i32 to index
          %32 = arith.subi %c8192_i32, %29 : i32
          %33 = arith.subi %32, %28 : i32
          %34 = arith.minsi %33, %c128_i32 : i32
          %35 = arith.index_cast %34 : i32 to index
          %subview_10 = memref.subview %reinterpret_cast[%31, 0] [%35, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>,pace<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_11 = memref.subview %alloc[0, 0] [%35, 128] [1, 1] : memref<128x128xf16, strided<[128, 1]>, #hivm.address memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_10 : memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_spsubview_11 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
        }
        // 加载 K 分块到 L1
        %13 = arith.muli %11, %c256_i32 : i32
        %14 = arith.index_cast %13 : i32 to index
        %subview = memref.subview %reinterpret_cast_4[%14, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #he<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_spaceoc_6 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
        // C1P: S = Q × K^T
        %15 = arith.muli %1, %c128_i32 : i32
        %16 = arith.subi %c8192_i32, %15 : i32
        %17 = arith.divsi %arg16, %c32_i32 : i32
        %18 = arith.muli %17, %c3072_i32 : i32
        %19 = arith.subi %16, %18 : i32
        %20 = arith.minsi %19, %c128_i32 : i32
        %21 = arith.index_cast %20 : i32 to index
        hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_6, %true, %21, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_7 : me, strided<[256, 1]>, #hivm.address_space<cc>>)

        // ===== C1S: 将 S 写入 workspace_s =====
        %22 = arith.remsi %arg16, %c2_i32 : i32        // slot = stream_id % 2  →  槽位索引 0或1
        %23 = arith.extsi %22 : i32 to i64
        // Wait Flag0 (slot0) 或 Flag1 (slot1): Vector 已读走上次 scores，槽位可用
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_FIX>] flag = %23
        %subview_8 = memref.subview %alloc_7[0, 0] [%21, 256] [1, 1] : memref<128x256xf32, strided<[256, 1]>, #hivm.address_mref<?x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
        %24 = arith.index_cast %1 : i32 to index
        %25 = arith.index_cast %22 : i32 to index
        %subview_9 = memref.subview %reinterpret_cast_2[%24, %25, 0, 0] [1, 1, %21, 256] [1, 1, 1, 1] : memref<24x2x128x256x536, 32768, 256, 1]>, #hivm.address_space<gm>> to memref<?x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%subview_8 : memref<?x256xf321]>, #hivm.address_space<cc>>) outs(%subview_9 : memref<?x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
        %26 = arith.extsi %22 : i32 to i64
        // Set Flag0/Flag1: 通知 Vector scores 已就绪
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %26 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }

      %10 = arith.cmpi sgt, %arg16, %c0_i32 : i32
      // ========== C2 阶段: 当 stream_id > 0 时执行 (total_id = stream_id - 1) ==========
      scf.if %10 {
        // 加载 V 分块到 L1
        %11 = arith.addi %arg16, %c31_i32 : i32
        %12 = arith.remsi %11, %c32_i32 : i32
        %13 = arith.muli %12, %c256_i32 : i32
        %14 = arith.index_cast %13 : i32 to index
        %subview = memref.subview %reinterpret_cast_0[%14, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #he<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_spaceoc_6 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false

        // ===== C2L: 加载 P 到 L1 =====
        // 计算 slot_p: total_id % 2 = (stream_id - 1) % 2 = (stream_id + 1) % 2
        %15 = arith.addi %arg16, %c1_i32 : i32
        %16 = arith.remsi %15, %c2_i32 : i32           // slot_p ∈ {0,1}
        %17 = arith.addi %16, %c4_i32 : i32            // flag = 4 + slot_p  → 4 或 5
        %18 = arith.extsi %17 : i32 to i64
        // Wait Flag4 (slot_p=0) 或 Flag5 (slot_p=1): Vector 已写完 P
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %18
        %19 = arith.index_cast %1 : i32 to index
        %20 = arith.index_cast %16 : i32 to index
        %21 = arith.muli %1, %c128_i32 : i32
        %22 = arith.subi %c8192_i32, %21 : i32
        %23 = arith.subi %arg16, %c1_i32 : i32
        %24 = arith.divsi %23, %c32_i32 : i32
        %25 = arith.muli %24, %c3072_i32 : i32
        %26 = arith.subi %22, %25 : i32
        %27 = arith.minsi %26, %c128_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        %subview_8 = memref.subview %reinterpret_cast_3[%19, %20, 0, 0] [1, 1, %28, 256] [1, 1, 1, 1] : memref<24x2x128x256x536, 32768, 256, 1]>, #hivm.address_space<gm>> to memref<?x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %alloc_5[0, 0] [%28, 256] [1, 1] : memref<128x256xf16, strided<[256, 1]>, #hivm.address_memref<?x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
        hivm.hir.nd2nz {dst_continuous} ins(%subview_8 : memref<?x256xf16, strided<[256, 1], offset: ?>, #hivm.address_spaceview_9 : memref<?x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
        %29 = arith.extsi %17 : i32 to i64
        // Set Flag4/Flag5: 通知 Vector Cube 已读完 P，槽位可释放
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %29 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>

        // C2P: O = P × V
        hivm.hir.mmadL1 ins(%alloc_5, %alloc_6, %true, %28, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>, #hivm.addr, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_7 : memref<128x25656, 1]>, #hivm.address_space<cc>>)

        // ===== C2S: 将 partial O 写入 workspace_o =====
        %30 = arith.addi %16, %c2_i32 : i32            // flag = 2 + slot_p  → 2 或 3
        %31 = arith.extsi %30 : i32 to i64
        // Wait Flag2 (slot_p=0) 或 Flag3 (slot_p=1): Vector 已读完上次的 O，槽位可用
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_FIX>] flag = %31
        %subview_10 = memref.subview %alloc_7[0, 0] [%28, 256] [1, 1] : memref<128x256xf32, strided<[256, 1]>, #hivm.addressemref<?x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
        %subview_11 = memref.subview %reinterpret_cast_1[%19, %20, 0, 0] [1, 1, %28, 128] [1, 1, 1, 1] : memref<24x2x128x1282768, 16384, 128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%subview_10 : memref<?x256xf3 1]>, #hivm.address_space<cc>>) outs(%subview_11 : memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
        %32 = arith.extsi %30 : i32 to i64
        // Set Flag2/Flag3: 通知 Vector partial O 已就绪
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
    }

    // ================================================================
    // c_clear_flags(): 等待 Vector 读完最后的 scores/output 数据
    // 同步标志 0,1 (scores 槽释放) 和 2,3 (output 槽释放)
    // ================================================================
    scf.for %arg16 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %6 = arith.extsi %arg16 : i32 to i64
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_FIX>] flag = %6
      // Wait Flag0, Flag1, Flag2, Flag3
    }
    return
  }

  // ================================================================
  // AIV (Vector) 侧: 负责在线 softmax + 累加 rescale
  // ================================================================
  func.func @FlashAttnExp_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2%arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.add %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: memref<?xf16, #hivm.address_space<gm>>, %arg8: memref<?xf16, #hivm.ad, %arg9: memref<?xf16, #hivm.address_space<gm>>, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i3yncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm. #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %cst = arith.constant 0.000000e+00 : f32
    %c31_i32 = arith.constant 31 : i32
    %c128 = arith.constant 128 : index
    %c16384 = arith.constant 16384 : index
    %cst_0 = arith.constant 0.0883883461 : f32
    %c256 = arith.constant 256 : index
    %c32768 = arith.constant 32768 : index
    %c65536 = arith.constant 65536 : index
    %c3072_i32 = arith.constant 3072 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32_i32 = arith.constant 32 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg9 to offset: [0], sizes: [24, 2, 128, 128], strides: [32768, 16384, 128,16, #hivm.address_space<gm>> to memref<24x2x128x128xf16, strided<[32768, 16384, 128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [24, 2, 128, 256], strides: [65536, 32768, 25xf16, #hivm.address_space<gm>> to memref<24x2x128x256xf16, strided<[65536, 32768, 256, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [24, 2, 128, 256], strides: [65536, 32768, 25xf16, #hivm.address_space<gm>> to memref<24x2x128x256xf16, strided<[65536, 32768, 256, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [128, 1] : memref<?xf16space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32

    // ================================================================
    // v_init_flags(): 初始化标志 0,1,2,3 为“空”（允许 Cube 写入 scores/output）
    // flag = 0,1,2,3
    // ================================================================
    scf.for %arg16 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %8 = arith.extsi %arg16 : i32 to i64
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %8 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      // Set Flag0,1,2,3: 释放 scores/O 槽位
    }
    %4 = arith.subi %c87_i32, %1 : i32
    %5 = arith.divsi %4, %c24_i32 : i32
    %6 = arith.muli %5, %c32_i32 : i32
    %7 = arith.addi %6, %c1_i32 : i32

    // ================================================================
    // 主软件流水循环: stream_id = %arg16
    // ================================================================
    scf.for %arg16 = %c0_i32 to %7 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>        // logsum
      %alloc_4 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>       // scores_max
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>       // scores_max_prev
      %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>       // scores_scale
      %alloc_7 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>       // scores_sum
      %alloc_8 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>   // cross_kernel_f16
      %alloc_9 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>   // cross_kernel_f32
      %alloc_10 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>  // acc_o
      %8 = arith.subi %c87_i32, %1 : i32
      %9 = arith.divsi %8, %c24_i32 : i32
      %10 = arith.muli %9, %c32_i32 : i32
      %11 = arith.cmpi slt, %arg16, %10 : i32

      // ========== V1 阶段: 当 stream_id < num_loops_total ==========
      scf.if %11 {
        %13 = arith.remsi %arg16, %c2_i32 : i32        // slot = stream_id % 2
        %14 = arith.extsi %13 : i32 to i64
        // Wait Flag0 (slot0) 或 Flag1 (slot1): Cube 已写完 scores
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %14
        // 加载 scores 到 UB ...
        %15 = arith.index_cast %1 : i32 to index
        %16 = arith.index_cast %13 : i32 to index
        %17 = arith.muli %1, %c128_i32 : i32
        %18 = arith.subi %c8192_i32, %17 : i32
        %19 = arith.divsi %arg16, %c32_i32 : i32
        %20 = arith.muli %19, %c3072_i32 : i32
        %21 = arith.subi %18, %20 : i32
        %22 = arith.minsi %21, %c128_i32 : i32
        %23 = arith.addi %22, %c1_i32 : i32
        %24 = arith.divsi %23, %c2_i32 : i32
        %25 = arith.muli %3, %24 : i32
        %26 = arith.index_cast %25 : i32 to index
        %27 = arith.index_cast %24 : i32 to index
        %28 = arith.muli %15, %c65536 : index
        %29 = arith.muli %16, %c32768 : index
        %30 = arith.addi %28, %29 : index
        %31 = arith.muli %26, %c256 : index
        %32 = arith.addi %30, %31 : index
        %subview = memref.subview %reinterpret_cast_1[%15, %16, %26, 0] [1, 1, %27, 256] [1, 1, 1, 1] : memref<24x2x128x256x536, 32768, 256, 1]>, #hivm.address_space<gm>> to memref<1x1x?x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.ad
        %subview_11 = memref.subview %alloc_8[0, 0] [%27, 256] [1, 1] : memref<64x256xf16, strided<[256, 1]>, #hivm.address_mref<?x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
        %reinterpret_cast_12 = memref.reinterpret_cast %subview to offset: [%32], sizes: [%27, 256], strides: [256, 1] : mem6, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x256xf16, strided<[256, 1], offset: ?>, #ce<gm>>
        memref.copy %reinterpret_cast_12, %subview_11 : memref<?x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
        %33 = arith.extsi %13 : i32 to i64
        // Set Flag0/Flag1: 通知 Cube scores 已被读走，可覆盖
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %33 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>

        // V1P: 在线 softmax
        %34 = arith.remsi %arg16, %c32_i32 : i32        // block_id
        %35 = arith.cmpi eq, %34, %c0_i32 : i32
        scf.if %35 {
          // 第一个 K/V 块时初始化
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_10 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
        }
        memref.copy %alloc_4, %alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<64x1xf32, strivm.address_space<ub>>
        hivm.hir.vcast ins(%alloc_8 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memrefided<[256, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vmul ins(%alloc_9, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%allo256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vreduce <max> ins(%alloc_9 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_4 2, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
        %36 = arith.cmpi sgt, %34, %c0_i32 : i32
        scf.if %36 {
          hivm.hir.vsub ins(%alloc_5, %alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64[1, 1]>, #hivm.address_space<ub>>)
        }
        hivm.hir.vsub ins(%alloc_9, %alloc_4 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [
        hivm.hir.vexp ins(%alloc_9 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<ded<[256, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vcast ins(%alloc_9 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memrefided<[256, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vreduce <sum> ins(%alloc_9 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 2, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
        hivm.hir.vmul ins(%alloc, %alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, sthivm.address_space<ub>>) outs(%alloc : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vadd ins(%alloc, %alloc_7 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, sthivm.address_space<ub>>) outs(%alloc : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)

        // ===== V1S: 将 softmax 后的 P 写入 workspace_p =====
        %37 = arith.addi %13, %c4_i32 : i32            // flag = 4 + slot  → 4 或 5
        %38 = arith.extsi %37 : i32 to i64
        // Wait Flag4/Flag5: Cube 已读完上次 P，槽位可写
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE3>] flag = %38
        %39 = arith.muli %15, %c65536 : index
        %40 = arith.muli %16, %c32768 : index
        %41 = arith.addi %39, %40 : index
        %42 = arith.muli %26, %c256 : index
        %43 = arith.addi %41, %42 : index
        %subview_13 = memref.subview %alloc_8[0, 0] [%27, 256] [1, 1] : memref<64x256xf16, strided<[256, 1]>, #hivm.address_mref<?x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
        %subview_14 = memref.subview %reinterpret_cast_2[%15, %16, %26, 0] [1, 1, %27, 256] [1, 1, 1, 1] : memref<24x2x128x2[65536, 32768, 256, 1]>, #hivm.address_space<gm>> to memref<1x1x?x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivmm>>
        %reinterpret_cast_15 = memref.reinterpret_cast %subview_14 to offset: [%43], sizes: [%27, 256], strides: [256, 1] : xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x256xf16, strided<[256, 1], offset: ?>space<gm>>
        memref.copy %subview_13, %reinterpret_cast_15 : memref<?x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memtrided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %44 = arith.extsi %37 : i32 to i64
        // Set Flag4/Flag5: 通知 Cube P 已就绪
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %44 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }

      %12 = arith.cmpi sgt, %arg16, %c0_i32 : i32
      // ========== V2 阶段: 当 stream_id > 0 (total_id = stream_id - 1) ==========
      scf.if %12 {
        // 计算 total_id 对应的 slot_p = total_id % 2 = (stream_id + 1) % 2
        %13 = arith.addi %arg16, %c1_i32 : i32
        %14 = arith.remsi %13, %c2_i32 : i32           // slot_p
        %15 = arith.addi %14, %c2_i32 : i32            // flag = 2 + slot_p  → 2 或 3
        %16 = arith.extsi %15 : i32 to i64
        // Wait Flag2/Flag3: Cube 已写完 partial O
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %16
        // 加载 partial O 到 UB ...
        %17 = arith.index_cast %1 : i32 to index
        %18 = arith.index_cast %14 : i32 to index
        %19 = arith.muli %1, %c128_i32 : i32
        %20 = arith.subi %c8192_i32, %19 : i32
        %21 = arith.subi %arg16, %c1_i32 : i32
        %22 = arith.divsi %21, %c32_i32 : i32
        %23 = arith.muli %22, %c3072_i32 : i32
        %24 = arith.subi %20, %23 : i32
        %25 = arith.minsi %24, %c128_i32 : i32
        %26 = arith.addi %25, %c1_i32 : i32
        %27 = arith.divsi %26, %c2_i32 : i32
        %28 = arith.muli %3, %27 : i32
        %29 = arith.index_cast %28 : i32 to index
        %30 = arith.index_cast %27 : i32 to index
        %31 = arith.muli %17, %c32768 : index
        %32 = arith.muli %18, %c16384 : index
        %33 = arith.addi %31, %32 : index
        %34 = arith.muli %29, %c128 : index
        %35 = arith.addi %33, %34 : index
        %subview = memref.subview %reinterpret_cast[%17, %18, %29, 0] [1, 1, %30, 128] [1, 1, 1, 1] : memref<24x2x128x128xf18, 16384, 128, 1]>, #hivm.address_space<gm>> to memref<1x1x?x128xf16, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.addr
        %subview_11 = memref.subview %alloc_8[0, 0] [%30, 128] [1, 1] : memref<64x256xf16, strided<[256, 1]>, #hivm.address_mref<?x128xf16, strided<[256, 1]>, #hivm.address_space<ub>>
        %reinterpret_cast_12 = memref.reinterpret_cast %subview to offset: [%35], sizes: [%30, 128], strides: [128, 1] : mem6, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #ce<gm>>
        memref.copy %reinterpret_cast_12, %subview_11 : memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<x128xf16, strided<[256, 1]>, #hivm.address_space<ub>>
        %36 = arith.extsi %15 : i32 to i64
        // Set Flag2/Flag3: 通知 Cube 已读完 O，槽位可释放
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %36 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>

        // V2P: 累加、rescale、最终归一化
        hivm.hir.vcast ins(%alloc_8 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memrefided<[256, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vmul ins(%alloc_10, %alloc_6 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1x 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast =
        %subview_13 = memref.subview %alloc_9[0, 0] [64, 128] [1, 1] : memref<64x256xf32, strided<[256, 1]>, #hivm.address_sref<64x128xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        hivm.hir.vadd ins(%alloc_10, %subview_13 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64d<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
        %37 = arith.addi %arg16, %c31_i32 : i32
        %38 = arith.remsi %37, %c32_i32 : i32
        %39 = arith.cmpi eq, %38, %c31_i32 : i32
        scf.if %39 {
          // 最后一个 K/V 块：归一化并写回全局内存
          hivm.hir.vdiv ins(%alloc_10, %alloc : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1x 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast =
          %subview_14 = memref.subview %alloc_8[0, 0] [64, 128] [1, 1] : memref<64x256xf16, strided<[256, 1]>, #hivm.addressemref<64x128xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vcast ins(%alloc_10 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_14 : 6, strided<[256, 1]>, #hivm.address_space<ub>>)
          %40 = arith.muli %1, %c128_i32 : i32
          %41 = arith.subi %c8192_i32, %40 : i32
          %42 = arith.subi %arg16, %c1_i32 : i32
          %43 = arith.divsi %42, %c32_i32 : i32
          %44 = arith.muli %43, %c3072_i32 : i32
          %45 = arith.subi %41, %44 : i32
          %46 = arith.minsi %45, %c128_i32 : i32
          %47 = arith.addi %46, %c1_i32 : i32
          %48 = arith.divsi %47, %c2_i32 : i32
          %49 = arith.index_cast %48 : i32 to index
          %50 = arith.addi %44, %40 : i32
          %51 = arith.muli %3, %48 : i32
          %52 = arith.addi %50, %51 : i32
          %53 = arith.index_cast %52 : i32 to index
          %subview_15 = memref.subview %alloc_8[0, 0] [%49, 128] [1, 1] : memref<64x256xf16, strided<[256, 1]>, #hivm.addresmemref<?x128xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %subview_16 = memref.subview %reinterpret_cast_3[%53, 0] [%49, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_15, %subview_16 : memref<?x128xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x1[128, 1], offset: ?>, #hivm.address_space<gm>>
        }
      }
    }

    // ================================================================
    // v_clear_flags(): 等待 Cube 读走最后的 P 数据
    // 同步标志 4,5
    // ================================================================
    scf.for %arg16 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %8 = arith.addi %arg16, %c4_i32 : i32           // flag = 4,5
      %9 = arith.extsi %8 : i32 to i64
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE3>] flag = %9
      // Wait Flag4, Flag5
    }
    return
  }
}