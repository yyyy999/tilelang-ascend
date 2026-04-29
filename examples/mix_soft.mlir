#map = affine_map<(d0)[s0] -> (d0 * 1048576 + s0)>
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @flash_attention_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c1048576 = arith.constant 1048576 : index
    return %c1048576 : index
  }
  func.func @flash_attention_get_kernel_num_args() -> i32 attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c4_i32 = arith.constant 4 : i32
    return %c4_i32 : i32
  }
  func.func @flash_attention_get_kernel_arg_type(%arg0: i32) -> i32 attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c-1_i32 = arith.constant -1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c129_i32 = arith.constant 129 : i32
    %0 = arith.cmpi eq, %arg0, %c3_i32 : i32
    %1 = arith.select %0, %c129_i32, %c-1_i32 : i32
    %c2_i32 = arith.constant 2 : i32
    %c129_i32_0 = arith.constant 129 : i32
    %2 = arith.cmpi eq, %arg0, %c2_i32 : i32
    %3 = arith.select %2, %c129_i32_0, %1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c129_i32_1 = arith.constant 129 : i32
    %4 = arith.cmpi eq, %arg0, %c1_i32 : i32
    %5 = arith.select %4, %c129_i32_1, %3 : i32
    %c0_i32 = arith.constant 0 : i32
    %c129_i32_2 = arith.constant 129 : i32
    %6 = arith.cmpi eq, %arg0, %c0_i32 : i32
    %7 = arith.select %6, %c129_i32_2, %5 : i32
    return %7 : i32
  }
  func.func @flash_attention_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c33_i32 = arith.constant 33 : i32
    // 修改：workspace 偏移调整，scores f16 占 256KB，P f16 偏移 256KB，partial O f16 偏移 512KB
    %c262144 = arith.constant 262144 : index
    %c524288 = arith.constant 524288 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply #map(%5)[%c0]
    // scores workspace 改为 f16 2x2x128x256
    %view = memref.view %arg2[%6][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x128x256xf16, #hivm.address_space<gm>>
    %7 = hivm.hir.get_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply #map(%8)[%c262144]  // P 偏移改为 256KB
    %view_4 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x128x256xf16, #hivm.address_space<gm>>
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply #map(%11)[%c524288] // partial O 偏移改为 512KB
    // partial O workspace 改为 f16 2x2x128x128
    %view_5 = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x128x128xf16, #hivm.address_space<gm>>
    %13 = arith.subi %c87_i32, %1 : i32
    %14 = arith.divsi %13, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %14 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %15 = arith.muli %arg13, %c3072_i32 : i32
      %16 = arith.muli %1, %c128_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview = memref.subview %reinterpret_cast[%18, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      scf.for %arg14 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %27 = arith.addi %arg14, %c4_i32 : i32
        %28 = arith.extsi %27 : i32 to i64
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %28 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
      scf.for %arg14 = %c0_i32 to %c33_i32 step %c1_i32  : i32 {
        %27 = arith.muli %arg14, %c256_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.muli %3, %c64_i32 : i32
        %30 = arith.index_cast %29 : i32 to index
        %31 = arith.cmpi slt, %arg14, %c32_i32 : i32
        %32 = arith.cmpi sgt, %arg14, %c0_i32 : i32
        %33 = arith.remsi %arg14, %c2_i32 : i32
        %34 = arith.index_cast %33 : i32 to index
        scf.if %31 {
          %alloc_6 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_7 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
          %subview_8 = memref.subview %reinterpret_cast_3[%28, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_8 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_6 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_6, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_7 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
          %35 = arith.remsi %arg14, %c2_i32 : i32
          %36 = arith.extsi %35 : i32 to i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_FIX>] flag = %36
          %subview_9 = memref.subview %view[%34, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<2x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          // C1S: 量化 f32 -> f16 写入 scores workspace
          hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%alloc_7 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
          %37 = arith.extsi %35 : i32 to i64
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scf.if %32 {
          %35 = arith.addi %arg14, %c1_i32 : i32
          %36 = arith.remsi %35, %c2_i32 : i32
          %37 = arith.addi %36, %c4_i32 : i32
          %38 = arith.extsi %37 : i32 to i64
          %39 = arith.index_cast %36 : i32 to index
          %alloc_6 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_7 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %alloc_8 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
          %subview_9 = memref.subview %view_4[%39, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<2x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %38
          %total_id = arith.subi %arg14, %c1_i32 : i32
          %v_offset = arith.muli %total_id, %c256_i32 : i32
          %v_idx = arith.index_cast %v_offset : i32 to index
          %subview_10 = memref.subview %reinterpret_cast_2[%v_idx, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_7 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.nd2nz {dst_continuous} ins(%subview_10 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_6 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          %40 = arith.extsi %37 : i32 to i64
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %40 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          hivm.hir.mmadL1 ins(%alloc_7, %alloc_6, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_8 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
          %41 = arith.addi %36, %c2_i32 : i32
          %42 = arith.extsi %41 : i32 to i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_FIX>] flag = %42
          %subview_11 = memref.subview %view_5[%39, 0, 0, 0] [1, 1, 128, 128] [1, 1, 1, 1] : memref<2x2x128x128xf16, #hivm.address_space<gm>> to memref<1x1x128x128xf16, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_12 = memref.collapse_shape %subview_11 [[0, 1, 2], [3]] : memref<1x1x128x128xf16, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          // C2S: 量化 f32 -> f16 写入 partial O workspace
          hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%alloc_8 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_12 : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
          %43 = arith.extsi %41 : i32 to i64
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %43 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %27 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_FIX>] flag = %27
      }
      %19 = arith.muli %3, %c64_i32 : i32
      %20 = arith.subi %c8192_i32, %19 : i32
      %21 = arith.subi %20, %16 : i32
      %22 = arith.subi %21, %15 : i32
      %23 = arith.minsi %22, %c64_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.addi %17, %19 : i32
      %26 = arith.index_cast %25 : i32 to index
    }
    return
  }
  func.func @flash_attention_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c33_i32 = arith.constant 33 : i32
    %c262144 = arith.constant 262144 : index
    %c524288 = arith.constant 524288 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply #map(%5)[%c0]
    // scores f16
    %view = memref.view %arg2[%6][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x128x256xf16, #hivm.address_space<gm>>
    %7 = hivm.hir.get_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply #map(%8)[%c262144]  // P 偏移
    %view_2 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x128x256xf16, #hivm.address_space<gm>>
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply #map(%11)[%c524288] // partial O 偏移，改为 f16
    %view_3 = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x128x128xf16, #hivm.address_space<gm>>
    %13 = arith.subi %c87_i32, %1 : i32
    %14 = arith.divsi %13, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %14 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_4 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_5 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %15 = arith.muli %arg13, %c3072_i32 : i32
      %16 = arith.muli %1, %c128_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %27 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %27 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
      %cst_one = arith.constant 1.0 : f32
      %alloc_8 = memref.alloc() : memref<2x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_one : f32) outs(%alloc_8 : memref<2x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>>)
      scf.for %arg14 = %c0_i32 to %c33_i32 step %c1_i32  : i32 {
        %27 = arith.muli %arg14, %c256_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.muli %3, %c64_i32 : i32
        %30 = arith.index_cast %29 : i32 to index
        %31 = arith.cmpi slt, %arg14, %c32_i32 : i32
        %32 = arith.cmpi sgt, %arg14, %c0_i32 : i32
        %33 = arith.remsi %arg14, %c2_i32 : i32
        %34 = arith.index_cast %33 : i32 to index
        scf.if %31 {
          %35 = arith.remsi %arg14, %c2_i32 : i32
          %36 = arith.extsi %35 : i32 to i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %36
          %alloc_9 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_10 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_11 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_12 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_13 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_14 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          // V1L: 加载 scores f16，再转 f32
          %alloc_scores_f16 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %subview_15 = memref.subview %view[%34, 0, %30, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<2x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_15 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_scores_f16 : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vcast ins(%alloc_scores_f16 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          %37 = arith.extsi %35 : i32 to i64
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          hivm.hir.vmul ins(%alloc_9, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_9 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_9 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc, %alloc_11 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_9, %alloc_13 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_9 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_9 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_9 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          %subview_16 = memref.subview %view_2[%34, 0, %30, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<2x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_17 = memref.collapse_shape %subview_16 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %38 = arith.addi %35, %c4_i32 : i32
          %39 = arith.extsi %38 : i32 to i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE3>] flag = %39
          memref.copy %alloc_10, %collapse_shape_17 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %40 = arith.extsi %38 : i32 to i64
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %40 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          hivm.hir.vsub ins(%alloc, %alloc_13 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %subview_18 = memref.subview %alloc_8[%34, 0, 0] [1, 64, 1] [1, 1, 1] : memref<2x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_19 = memref.collapse_shape %subview_18 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vexp ins(%alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
          hivm.hir.vmul ins(%alloc_4, %collapse_shape_19 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_4, %alloc_12 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_14, %alloc_13 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        scf.if %32 {
          %35 = arith.addi %arg14, %c1_i32 : i32
          %36 = arith.remsi %35, %c2_i32 : i32
          %37 = arith.addi %36, %c2_i32 : i32
          %38 = arith.extsi %37 : i32 to i64
          %39 = arith.index_cast %36 : i32 to index
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %38
          %alloc_9 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          // V2L: 加载 partial O f16，转 f32
          %alloc_o_f16 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
          %subview_10 = memref.subview %view_3[%39, 0, %30, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<2x2x128x128xf16, #hivm.address_space<gm>> to memref<1x1x64x128xf16, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_10 [[0, 1, 2], [3]] : memref<1x1x64x128xf16, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_o_f16 : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
          hivm.hir.vcast ins(%alloc_o_f16 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
          %40 = arith.extsi %37 : i32 to i64
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %40 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          %subview_11 = memref.subview %alloc_8[%39, 0, 0] [1, 64, 1] [1, 1, 1] : memref<2x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_12 = memref.collapse_shape %subview_11 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_5, %collapse_shape_12 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_5, %alloc_9 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg14 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %27 = arith.addi %arg14, %c4_i32 : i32
        %28 = arith.extsi %27 : i32 to i64
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE3>] flag = %28
      }
      hivm.hir.vdiv ins(%alloc_5, %alloc_4 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %19 = arith.muli %3, %c64_i32 : i32
      %20 = arith.subi %c8192_i32, %19 : i32
      %21 = arith.subi %20, %16 : i32
      %22 = arith.subi %21, %15 : i32
      %23 = arith.minsi %22, %c64_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %alloc_6[0, 0] [%24, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %25 = arith.addi %17, %19 : i32
      %26 = arith.index_cast %25 : i32 to index
      %subview_7 = memref.subview %reinterpret_cast[%26, 0] [%24, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %subview_7 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}