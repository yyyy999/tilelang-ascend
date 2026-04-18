// -----// IR Dump After TileLangIRInsertCVSync (tilelangir-insert-cv-sync) ('builtin.module' operation) //----- //
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
    %c0 = arith.constant 0 : index
    %4 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf32, #hivm.address_space<gm>>
    %c524288 = arith.constant 524288 : index
    %5 = memref_ext.alloc_workspace() from %arg2 offset = [%c524288] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf16, #hivm.address_space<gm>>
    %c786432 = arith.constant 786432 : index
    %6 = memref_ext.alloc_workspace() from %arg2 offset = [%c786432] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x128xf32, #hivm.address_space<gm>>
    %7 = arith.subi %c151_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
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
      %c4_i32 = arith.constant 4 : i32
      %13 = arith.divsi %c32_i32, %c4_i32 : i32
      %c0_i32_10 = arith.constant 0 : i32
      %c1_i32_11 = arith.constant 1 : i32
      %c4_i32_12 = arith.constant 4 : i32
      scf.for %arg14 = %c0_i32_10 to %c4_i32_12 step %c1_i32_11  : i32 {
        %22 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %22 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scf.for %arg14 = %c0_i32 to %13 step %c1_i32  : i32 {
        %alloc_15 = memref.alloc() : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>>
        %22 = arith.muli %arg14, %c256_i32 : i32
        %23 = arith.index_cast %22 : i32 to index
        %24 = arith.muli %3, %c32_i32 : i32
        %25 = arith.index_cast %24 : i32 to index
        %c0_i32_16 = arith.constant 0 : i32
        %c4_i32_17 = arith.constant 4 : i32
        %c1_i32_18 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_16 to %c4_i32_17 step %c1_i32_18  : i32 {
          %26 = arith.extsi %arg15 : i32 to i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %26
          %c4_i32_28 = arith.constant 4 : i32
          %27 = arith.muli %arg14, %c4_i32_28 : i32
          %28 = arith.addi %27, %arg15 : i32
          %29 = arith.index_cast %28 : i32 to index
          %30 = arith.index_cast %arg15 : i32 to index
          %c256_i32_29 = arith.constant 256 : i32
          %31 = arith.muli %28, %c256_i32_29 : i32
          %32 = arith.index_cast %31 : i32 to index
          %subview_30 = memref.subview %reinterpret_cast_3[%32, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_31 = memref.subview %4[%30, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_31 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_32 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_33 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_30 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_32 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_32, %true, %c64, %c128, %c256 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_33 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_33 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %26 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_19 = arith.constant 0 : i32
        %c4_i32_20 = arith.constant 4 : i32
        %c1_i32_21 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_19 to %c4_i32_20 step %c1_i32_21  : i32 {
          %26 = arith.extsi %arg15 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c4_i64 = arith.constant 4 : i64
          %27 = arith.addi %26, %c4_i64 : i64
          %28 = arith.remsi %27, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %26
          %c4_i32_28 = arith.constant 4 : i32
          %29 = arith.muli %arg14, %c4_i32_28 : i32
          %30 = arith.addi %29, %arg15 : i32
          %31 = arith.index_cast %30 : i32 to index
          %32 = arith.index_cast %arg15 : i32 to index
          %subview_29 = memref.subview %4[%32, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_29 [[0, 1, 2], [3]] : memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_30 = memref.subview %5[%32, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_31 = memref.collapse_shape %subview_30 [[0, 1, 2], [3]] : memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_32 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_33 = memref.alloc() : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_34 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_35 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_36 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_37 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_38 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_32 : memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_32, %alloc_8 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_32 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_34 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc_5, %alloc_34 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_36 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_5, %alloc_36 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_38 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %33 = arith.index_cast %arg15 : i32 to index
          %subview_39 = memref.subview %alloc_15[%33, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_40 = memref.collapse_shape %subview_39 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vexp ins(%alloc_38 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_40 : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_32, %alloc_36 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_37 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_37 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_32 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_35 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          %34 = arith.index_cast %arg15 : i32 to index
          %subview_41 = memref.subview %alloc_15[%34, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_42 = memref.collapse_shape %subview_41 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_6, %collapse_shape_42 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_6, %alloc_35 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_38 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_38, %alloc_36 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_32 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_33 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
          memref.copy %alloc_33, %collapse_shape_31 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %28 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_22 = arith.constant 0 : i32
        %c4_i32_23 = arith.constant 4 : i32
        %c1_i32_24 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_22 to %c4_i32_23 step %c1_i32_24  : i32 {
          %26 = arith.extsi %arg15 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c4_i64 = arith.constant 4 : i64
          %27 = arith.addi %26, %c4_i64 : i64
          %28 = arith.remsi %27, %c16_i64 : i64
          %c4_i64_28 = arith.constant 4 : i64
          %29 = arith.addi %26, %c4_i64_28 : i64
          %30 = arith.remsi %29, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %28
          %c4_i32_29 = arith.constant 4 : i32
          %31 = arith.muli %arg14, %c4_i32_29 : i32
          %32 = arith.addi %31, %arg15 : i32
          %33 = arith.index_cast %32 : i32 to index
          %34 = arith.index_cast %arg15 : i32 to index
          %c256_i32_30 = arith.constant 256 : i32
          %35 = arith.muli %32, %c256_i32_30 : i32
          %36 = arith.index_cast %35 : i32 to index
          %subview_31 = memref.subview %reinterpret_cast_2[%36, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_32 = memref.subview %5[%34, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_32 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_33 = memref.subview %6[%34, 0, 0, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_34 = memref.collapse_shape %subview_33 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_35 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_36 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %alloc_37 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_36 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.nd2nz {dst_continuous} ins(%subview_31 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_35 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_36, %alloc_35, %true, %c64, %c256, %c128 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_37 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_37 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_34 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %30 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_25 = arith.constant 0 : i32
        %c4_i32_26 = arith.constant 4 : i32
        %c1_i32_27 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_25 to %c4_i32_26 step %c1_i32_27  : i32 {
          %26 = arith.extsi %arg15 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c4_i64 = arith.constant 4 : i64
          %27 = arith.addi %26, %c4_i64 : i64
          %28 = arith.remsi %27, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %28
          %c4_i32_28 = arith.constant 4 : i32
          %29 = arith.muli %arg14, %c4_i32_28 : i32
          %30 = arith.addi %29, %arg15 : i32
          %31 = arith.index_cast %30 : i32 to index
          %32 = arith.index_cast %arg15 : i32 to index
          %subview_29 = memref.subview %6[%32, 0, %25, 0] [1, 1, 32, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_29 [[0, 1, 2], [3]] : memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_30 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_30 : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          %33 = arith.index_cast %arg15 : i32 to index
          %subview_31 = memref.subview %alloc_15[%33, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_32 = memref.collapse_shape %subview_31 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_7, %collapse_shape_32 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_7, %alloc_30 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %26 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 4 : i32}
      scf.for %arg14 = %c0_i32_10 to %c4_i32_12 step %c1_i32_11  : i32 {
        %22 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %22
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %14 = arith.muli %3, %c32_i32 : i32
      %15 = arith.subi %c8192_i32, %14 : i32
      %16 = arith.subi %15, %10 : i32
      %17 = arith.subi %16, %9 : i32
      %18 = arith.minsi %17, %c32_i32 : i32
      %19 = arith.index_cast %18 : i32 to index
      %subview_13 = memref.subview %alloc_9[0, 0] [%19, 128] [1, 1] : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %20 = arith.addi %11, %14 : i32
      %21 = arith.index_cast %20 : i32 to index
      %subview_14 = memref.subview %reinterpret_cast_4[%21, 0] [%19, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_13, %subview_14 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}