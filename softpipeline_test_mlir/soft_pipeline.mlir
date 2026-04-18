// Soft Pipeline IR: 将 4 个 ScopeOp 合并为 1 个内层循环，按 j 值分发执行
// 基于 aftercvsplit.mlir 手动构造
//
// 核心变换：
//   之前: 外层循环 32 次, 内部 4 个 ScopeOp 顺序执行
//   之后: 外层循环 8 次 (=32/4), 内层循环 4 次, 每次 j 执行不同 Scope
//
// 同步策略 (Soft Pipeline):
//   j=0 (C:QK):   wait(V, j-1), [C:QK], set(C, j)       -- 等上一个 V 完成
//   j=1 (V:Soft):  wait(C, j),   [V:Softmax], set(V, j)   -- 等 C 完成
//   j=2 (C:PV):    wait(V, j),   [C:PV], set(C, j+1)      -- 等 V 完成
//   j=3 (V:Acc):   wait(C, j+1), [V:Acc], set(V, j+1)     -- 等 C 完成
//
// 跨外层迭代叠加:
//   外层 i=0: [C:QK_0] [V:Soft_0] [C:PV_0] [V:Acc_0]
//   外层 i=1:                                [C:QK_1] [V:Soft_1] [C:PV_1] [V:Acc_1]
//                                             ↑ V:Acc_0 与 C:QK_1 叠加！

module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c64_i32 = arith.constant 64 : i32
    %c1536_i32 = arith.constant 1536 : i32
    %c24_i32 = arith.constant 24 : i32
    %c151_i32 = arith.constant 151 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32

    // ===== Workspace 扩展: 添加 num_stages 维度 =====
    // 之前: memref<2x64x256xf32>
    // 之后: memref<4x2x64x256xf32>  (4 = num_stages)
    %c0 = arith.constant 0 : index
    %4 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf32, #hivm.address_space<gm>>
    %c524288 = arith.constant 524288 : index
    %5 = memref_ext.alloc_workspace() from %arg2 offset = [%c524288] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf16, #hivm.address_space<gm>>
    %c786432 = arith.constant 786432 : index
    %6 = memref_ext.alloc_workspace() from %arg2 offset = [%c786432] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x128xf32, #hivm.address_space<gm>>

    %7 = arith.subi %c151_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32

    // ===== Init 循环: 在外层循环之前初始化同步标志 =====
    // 对应 insert_cv_sync 的 InsertInitAndClearInSinglePipeline
    // 初始化 Vector 的标志，让第一个 Cube 操作不会被阻塞
    %c0_i32_init = arith.constant 0 : i32
    %c4_i32_init = arith.constant 4 : i32
    %c1_i32_init = arith.constant 1 : i32
    scf.for %arg_init = %c0_i32_init to %c4_i32_init step %c1_i32_init : i32 {
      %init_id = arith.extsi %arg_init : i32 to i64
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %init_id syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scf.for %arg13 = %c0_i32 to %8 step %c1_i32 : i32 {
      %alloc = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      %alloc_9 = memref.alloc() : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %9 = arith.muli %arg13, %c1536_i32 : i32
      %10 = arith.muli %1, %c64_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %subview = memref.subview %reinterpret_cast[%12, 0] [64, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_8 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)

      // ===== 外层循环迭代次数 = 32 / 4 = 8 =====
      %c4_i32 = arith.constant 4 : i32
      %13 = arith.divsi %c32_i32, %c4_i32 : i32

      scf.for %arg14 = %c0_i32 to %13 step %c1_i32 : i32 {
        %alloc_12 = memref.alloc() : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>>
        %22 = arith.muli %arg14, %c256_i32 : i32
        %23 = arith.index_cast %22 : i32 to index
        %24 = arith.muli %3, %c32_i32 : i32
        %25 = arith.index_cast %24 : i32 to index

        // ===== Soft Pipeline: 单个内层循环，4 次迭代 =====
        // 每次迭代 j 执行不同的 Scope，配合同步实现 CV 叠加
        %c0_i32_inner = arith.constant 0 : i32
        %c4_i32_inner = arith.constant 4 : i32
        %c1_i32_inner = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_inner to %c4_i32_inner step %c1_i32_inner : i32 {

          // ===== 计算同步 flag ID =====
          // flag_id = outerIV * num_stages + innerIV
          %c4_i32_flag = arith.constant 4 : i32
          %flag_base = arith.muli %arg14, %c4_i32_flag : i32
          %flag_id_i32 = arith.addi %flag_base, %arg15 : i32
          %flag_id_i64 = arith.extsi %flag_id_i32 : i32 to i64

          // 计算 flag_id + num_stages (用于跨外层迭代的同步)
          %c4_i64 = arith.constant 4 : i64
          %flag_id_next_i64 = arith.addi %flag_id_i64, %c4_i64 : i64

          // SYNC_FLAGS_LIMIT = 16
          %c16_i64 = arith.constant 16 : i64
          %flag_id_mod = arith.remsi %flag_id_i64, %c16_i64 : i64
          %flag_id_next_mod = arith.remsi %flag_id_next_i64, %c16_i64 : i64

          // 计算 workspace subview 的 stage index
          %stage_idx = arith.index_cast %arg15 : i32 to index

          // 计算 KV 的行偏移: (arg14 * 4 + arg15) * 256
          %c4_i32_kv = arith.constant 4 : i32
          %kv_offset_base = arith.muli %arg14, %c4_i32_kv : i32
          %kv_offset_idx = arith.addi %kv_offset_base, %arg15 : i32
          %kv_offset = arith.muli %kv_offset_idx, %c256_i32 : i32
          %kv_offset_index = arith.index_cast %kv_offset : i32 to index

          // ===== j == 0: CUBE - Load K + QK matmul =====
          %cmp_j0 = arith.cmpi eq, %arg15, %c0_i32 : i32
          scf.if %cmp_j0 {
            // wait(V, flag_id - num_stages) = wait(V, 上一个外层迭代的 V 完成)
            // 对于第一次外层迭代，init 循环已经 set 了这些 flag
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %flag_id_i64

            %alloc_13 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_14 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
            %subview_15 = memref.subview %reinterpret_cast_3[%kv_offset_index, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_15 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_13 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_13, %true, %c64, %c128, %c256 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_14 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)

            // 写入 workspace_1[stage_idx, 0, 0, 0]
            %subview_16 = memref.subview %4[%stage_idx, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_16 = memref.collapse_shape %subview_16 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_14 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_16 : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)

            // set(C, flag_id) - 通知 Cube 完成
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %flag_id_i64 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }

          // ===== j == 1: VECTOR - Softmax =====
          %c1_i32_cmp = arith.constant 1 : i32
          %cmp_j1 = arith.cmpi eq, %arg15, %c1_i32_cmp : i32
          scf.if %cmp_j1 {
            // wait(C, flag_id) - 等待 Cube 的 QK 完成
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %flag_id_i64

            %alloc_13 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_14 = memref.alloc() : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_15 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_16 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_17 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_18 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_19 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>

            // 读取 workspace_1[stage_idx, 0, vid, 0]
            %subview_20 = memref.subview %4[%stage_idx, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_20 = memref.collapse_shape %subview_20 [[0, 1, 2], [3]] : memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_20, %alloc_13 : memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>

            // Softmax 计算
            hivm.hir.vmul ins(%alloc_13, %alloc_8 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <max> ins(%alloc_13 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            hivm.hir.vmax ins(%alloc_5, %alloc_15 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_17 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_5, %alloc_17 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_19 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            %subview_correction = memref.subview %alloc_12[%stage_idx, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_correction = memref.collapse_shape %subview_correction [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vexp ins(%alloc_19 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_correction : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_13, %alloc_17 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_18 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vexp ins(%alloc_18 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <sum> ins(%alloc_13 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            %subview_correction2 = memref.subview %alloc_12[%stage_idx, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_correction2 = memref.collapse_shape %subview_correction2 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_6, %collapse_correction2 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_6, %alloc_16 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vbrc ins(%cst : f32) outs(%alloc_19 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_19, %alloc_17 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vcast ins(%alloc_13 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)

            // 写入 workspace_2[stage_idx, 0, vid, 0]
            %subview_21 = memref.subview %5[%stage_idx, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_21 = memref.collapse_shape %subview_21 [[0, 1, 2], [3]] : memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %alloc_14, %collapse_21 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>

            // set(V, flag_id + num_stages) - 通知 Vector 完成（用 flag_id_next 让下一个外层迭代的 C 等待）
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %flag_id_next_mod syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }

          // ===== j == 2: CUBE - Load V + PV matmul =====
          %c2_i32_cmp = arith.constant 2 : i32
          %cmp_j2 = arith.cmpi eq, %arg15, %c2_i32_cmp : i32
          scf.if %cmp_j2 {
            // wait(V, flag_id + num_stages) - 等待 Vector 的 Softmax 完成
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %flag_id_next_mod

            %alloc_13 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_14 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
            %alloc_15 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>

            // 读取 workspace_2[stage_idx, 0, 0, 0]
            %subview_16 = memref.subview %5[%stage_idx, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_16 = memref.collapse_shape %subview_16 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%collapse_16 : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_14 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false

            %subview_17 = memref.subview %reinterpret_cast_2[%kv_offset_index, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_17 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_13 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 ins(%alloc_14, %alloc_13, %true, %c64, %c256, %c128 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_15 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)

            // 写入 workspace_3[stage_idx, 0, 0, 0]
            %subview_18 = memref.subview %6[%stage_idx, 0, 0, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_18 = memref.collapse_shape %subview_18 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_15 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_18 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)

            // set(C, flag_id + num_stages) - 通知 Cube 完成（用 flag_id_next 让下一个 V 等待）
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %flag_id_next_mod syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }

          // ===== j == 3: VECTOR - acc_o 累加 =====
          %c3_i32_cmp = arith.constant 3 : i32
          %cmp_j3 = arith.cmpi eq, %arg15, %c3_i32_cmp : i32
          scf.if %cmp_j3 {
            // wait(C, flag_id + num_stages) - 等待 Cube 的 PV 完成
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %flag_id_next_mod

            %alloc_13 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>

            // 读取 workspace_3[stage_idx, 0, vid, 0]
            %subview_14 = memref.subview %6[%stage_idx, 0, %25, 0] [1, 1, 32, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_14 = memref.collapse_shape %subview_14 [[0, 1, 2], [3]] : memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_14, %alloc_13 : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>

            %subview_correction3 = memref.subview %alloc_12[%stage_idx, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_correction3 = memref.collapse_shape %subview_correction3 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_7, %collapse_correction3 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vadd ins(%alloc_7, %alloc_13 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)

            // set(V, flag_id + num_stages) - 通知 Vector 完成（让下一个外层迭代的 C:QK 等待）
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %flag_id_next_mod syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }

        } {tilelangir.num_stages = 4 : i32, hivm.soft_pipeline}
      } {tilelangir.num_stages = 4 : i32}

      // ===== Clear 循环: 在外层循环之后清除同步标志 =====
      %c0_i32_clear = arith.constant 0 : i32
      %c4_i32_clear = arith.constant 4 : i32
      %c1_i32_clear = arith.constant 1 : i32
      scf.for %arg_clear = %c0_i32_clear to %c4_i32_clear step %c1_i32_clear : i32 {
        %clear_id = arith.extsi %arg_clear : i32 to i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %clear_id
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}

      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %14 = arith.muli %3, %c32_i32 : i32
      %15 = arith.subi %c8192_i32, %14 : i32
      %16 = arith.subi %15, %10 : i32
      %17 = arith.subi %16, %9 : i32
      %18 = arith.minsi %17, %c32_i32 : i32
      %19 = arith.index_cast %18 : i32 to index
      %subview_10 = memref.subview %alloc_9[0, 0] [%19, 128] [1, 1] : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %20 = arith.addi %11, %14 : i32
      %21 = arith.index_cast %20 : i32 to index
      %subview_11 = memref.subview %reinterpret_cast_4[%21, 0] [%19, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_10, %subview_11 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}
