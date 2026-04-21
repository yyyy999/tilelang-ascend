// -----// IR Dump After TileLangIREnableSoftPipeline (tilelangir-enable-soft-pipeline) ('builtin.module' operation) //----- //
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
    %4 = memref_ext.alloc_workspace() : memref<4x2x64x256xf32, #hivm.address_space<gm>>
    %5 = memref_ext.alloc_workspace() : memref<4x2x64x256xf16, #hivm.address_space<gm>>
    %6 = memref_ext.alloc_workspace() : memref<4x2x64x128xf32, #hivm.address_space<gm>>
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
      memref.copy %subview, %alloc : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_8 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
      %c4_i32 = arith.constant 4 : i32
      %13 = arith.divsi %c32_i32, %c4_i32 : i32
      %c0_i32_10 = arith.constant 0 : i32
      %c4_i32_11 = arith.constant 4 : i32
      %c1_i32_12 = arith.constant 1 : i32
      scf.for %arg14 = %c0_i32_10 to %c4_i32_11 step %c1_i32_12  : i32 {
        %22 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %22 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scf.for %arg14 = %c0_i32 to %13 step %c1_i32  : i32 {
        %alloc_15 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        annotation.mark %alloc_15 {hivm.multi_buffer = 4 : i32} : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %22 = arith.muli %arg14, %c256_i32 : i32
        %23 = arith.index_cast %22 : i32 to index
        %24 = arith.muli %3, %c32_i32 : i32
        %25 = arith.index_cast %24 : i32 to index
        %c0_i32_16 = arith.constant 0 : i32
        %c4_i32_17 = arith.constant 4 : i32
        %c1_i32_18 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_16 to %c4_i32_17 step %c1_i32_18  : i32 {
          %26 = arith.extsi %arg14 : i32 to i64
          %27 = arith.extsi %arg15 : i32 to i64
          %c4_i64 = arith.constant 4 : i64
          %c16_i64 = arith.constant 16 : i64
          %28 = arith.muli %26, %c4_i64 : i64
          %29 = arith.addi %28, %27 : i64
          %30 = arith.remsi %29, %c16_i64 : i64
          %c1_i64 = arith.constant 1 : i64
          %31 = arith.addi %29, %c1_i64 : i64
          %32 = arith.remsi %31, %c16_i64 : i64
          %33 = arith.index_cast %arg15 : i32 to index
          %c4_i32_19 = arith.constant 4 : i32
          %34 = arith.muli %arg14, %c4_i32_19 : i32
          %35 = arith.addi %34, %arg15 : i32
          %36 = arith.index_cast %35 : i32 to index
          %c0_i32_20 = arith.constant 0 : i32
          %37 = arith.cmpi eq, %arg15, %c0_i32_20 : i32
          scf.if %37 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_23 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
            %subview_24 = memref.subview %reinterpret_cast_3[%23, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %subview_24, %alloc_22 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_22, %true, %c64, %c128, %c256 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_23 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_25 = memref.subview %4[%41, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %alloc_23, %collapse_shape : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>> to memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c1_i32_21 = arith.constant 1 : i32
          %38 = arith.cmpi eq, %arg15, %c1_i32_21 : i32
          scf.if %38 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_23 = memref.alloc() : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_24 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_25 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_26 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_27 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_28 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_29 = memref.subview %4[%41, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_29 [[0, 1, 2], [3]] : memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_22 : memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_22, %alloc_8 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <max> ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_24 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            hivm.hir.vmax ins(%alloc_5, %alloc_24 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_5, %alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vexp ins(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_22, %alloc_26 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vexp ins(%alloc_27 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <sum> ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            hivm.hir.vmul ins(%alloc_6, %alloc_15 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_6, %alloc_25 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vbrc ins(%cst : f32) outs(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_28, %alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vcast ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_23 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_30 = memref.subview %5[%42, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_31 = memref.collapse_shape %subview_30 [[0, 1, 2], [3]] : memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %alloc_23, %collapse_shape_31 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c2_i32 = arith.constant 2 : i32
          %39 = arith.cmpi eq, %arg15, %c2_i32 : i32
          scf.if %39 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_23 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
            %alloc_24 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_25 = memref.subview %5[%41, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_23 : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
            %subview_26 = memref.subview %reinterpret_cast_2[%23, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %subview_26, %alloc_22 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            hivm.hir.mmadL1 ins(%alloc_23, %alloc_22, %true, %c64, %c256, %c128 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_24 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_27 = memref.subview %6[%42, 0, 0, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %alloc_24, %collapse_shape_28 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>> to memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c3_i32 = arith.constant 3 : i32
          %40 = arith.cmpi eq, %arg15, %c3_i32 : i32
          scf.if %40 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_23 = memref.subview %6[%41, 0, %25, 0] [1, 1, 32, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_23 [[0, 1, 2], [3]] : memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_22 : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_7, %alloc_15 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vadd ins(%alloc_7, %alloc_22 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
        } {hivm.soft_pipeline}
      } {tilelangir.num_stages = 4 : i32}
      scf.for %arg14 = %c0_i32_10 to %c4_i32_11 step %c1_i32_12  : i32 {
        %c4_i64 = arith.constant 4 : i64
        %22 = arith.extsi %13 : i32 to i64
        %23 = arith.muli %22, %c4_i64 : i64
        %c16_i64 = arith.constant 16 : i64
        %24 = arith.remsi %23, %c16_i64 : i64
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %24
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
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

// -----// IR Dump After TileLangIREnableLocalBuffer (tilelangir-enable-local-buffer) ('builtin.module' operation) //----- //
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
    %4 = memref_ext.alloc_workspace() : memref<4x2x64x256xf32, #hivm.address_space<gm>>
    %5 = memref_ext.alloc_workspace() : memref<4x2x64x256xf16, #hivm.address_space<gm>>
    %6 = memref_ext.alloc_workspace() : memref<4x2x64x128xf32, #hivm.address_space<gm>>
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
      memref.copy %subview, %alloc : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_8 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
      %c4_i32 = arith.constant 4 : i32
      %13 = arith.divsi %c32_i32, %c4_i32 : i32
      %c0_i32_10 = arith.constant 0 : i32
      %c4_i32_11 = arith.constant 4 : i32
      %c1_i32_12 = arith.constant 1 : i32
      scf.for %arg14 = %c0_i32_10 to %c4_i32_11 step %c1_i32_12  : i32 {
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
          %26 = arith.extsi %arg14 : i32 to i64
          %27 = arith.extsi %arg15 : i32 to i64
          %c4_i64 = arith.constant 4 : i64
          %c16_i64 = arith.constant 16 : i64
          %28 = arith.muli %26, %c4_i64 : i64
          %29 = arith.addi %28, %27 : i64
          %30 = arith.remsi %29, %c16_i64 : i64
          %c1_i64 = arith.constant 1 : i64
          %31 = arith.addi %29, %c1_i64 : i64
          %32 = arith.remsi %31, %c16_i64 : i64
          %33 = arith.index_cast %arg15 : i32 to index
          %c4_i32_19 = arith.constant 4 : i32
          %34 = arith.muli %arg14, %c4_i32_19 : i32
          %35 = arith.addi %34, %arg15 : i32
          %36 = arith.index_cast %35 : i32 to index
          %c0_i32_20 = arith.constant 0 : i32
          %37 = arith.cmpi eq, %arg15, %c0_i32_20 : i32
          scf.if %37 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_23 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
            %subview_24 = memref.subview %reinterpret_cast_3[%23, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %subview_24, %alloc_22 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_22, %true, %c64, %c128, %c256 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_23 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_25 = memref.subview %4[%41, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %alloc_23, %collapse_shape : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>> to memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c1_i32_21 = arith.constant 1 : i32
          %38 = arith.cmpi eq, %arg15, %c1_i32_21 : i32
          scf.if %38 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_23 = memref.alloc() : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_24 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_25 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_26 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_27 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_28 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_29 = memref.subview %4[%41, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_29 [[0, 1, 2], [3]] : memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_22 : memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_22, %alloc_8 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <max> ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_24 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            hivm.hir.vmax ins(%alloc_5, %alloc_24 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_5, %alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_30 = memref.subview %alloc_15[%42, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_31 = memref.collapse_shape %subview_30 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vexp ins(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_31 : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_22, %alloc_26 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vexp ins(%alloc_27 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <sum> ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_32 = memref.subview %alloc_15[%43, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_33 = memref.collapse_shape %subview_32 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_6, %collapse_shape_33 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_6, %alloc_25 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vbrc ins(%cst : f32) outs(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_28, %alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vcast ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_23 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
            %44 = arith.index_cast %arg15 : i32 to index
            %subview_34 = memref.subview %5[%44, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_35 = memref.collapse_shape %subview_34 [[0, 1, 2], [3]] : memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %alloc_23, %collapse_shape_35 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c2_i32 = arith.constant 2 : i32
          %39 = arith.cmpi eq, %arg15, %c2_i32 : i32
          scf.if %39 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_23 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
            %alloc_24 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_25 = memref.subview %5[%41, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_23 : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
            %subview_26 = memref.subview %reinterpret_cast_2[%23, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %subview_26, %alloc_22 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            hivm.hir.mmadL1 ins(%alloc_23, %alloc_22, %true, %c64, %c256, %c128 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_24 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_27 = memref.subview %6[%42, 0, 0, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %alloc_24, %collapse_shape_28 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>> to memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c3_i32 = arith.constant 3 : i32
          %40 = arith.cmpi eq, %arg15, %c3_i32 : i32
          scf.if %40 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_23 = memref.subview %6[%41, 0, %25, 0] [1, 1, 32, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_23 [[0, 1, 2], [3]] : memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_22 : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_24 = memref.subview %alloc_15[%42, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_25 = memref.collapse_shape %subview_24 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_7, %collapse_shape_25 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vadd ins(%alloc_7, %alloc_22 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
        } {hivm.soft_pipeline}
      } {tilelangir.num_stages = 4 : i32}
      scf.for %arg14 = %c0_i32_10 to %c4_i32_11 step %c1_i32_12  : i32 {
        %c4_i64 = arith.constant 4 : i64
        %22 = arith.extsi %13 : i32 to i64
        %23 = arith.muli %22, %c4_i64 : i64
        %c16_i64 = arith.constant 16 : i64
        %24 = arith.remsi %23, %c16_i64 : i64
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %24
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
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

// -----// IR Dump After TileLangIRSpecializeCube (tilelangir-specialize-cube) ('func.func' operation: @flash_attention) //----- //
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
    %4 = memref_ext.alloc_workspace() : memref<4x2x64x256xf32, #hivm.address_space<gm>>
    %5 = memref_ext.alloc_workspace() : memref<4x2x64x256xf16, #hivm.address_space<gm>>
    %6 = memref_ext.alloc_workspace() : memref<4x2x64x128xf32, #hivm.address_space<gm>>
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
      %c4_i32_11 = arith.constant 4 : i32
      %c1_i32_12 = arith.constant 1 : i32
      scf.for %arg14 = %c0_i32_10 to %c4_i32_11 step %c1_i32_12  : i32 {
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
          %26 = arith.extsi %arg14 : i32 to i64
          %27 = arith.extsi %arg15 : i32 to i64
          %c4_i64 = arith.constant 4 : i64
          %c16_i64 = arith.constant 16 : i64
          %28 = arith.muli %26, %c4_i64 : i64
          %29 = arith.addi %28, %27 : i64
          %30 = arith.remsi %29, %c16_i64 : i64
          %c1_i64 = arith.constant 1 : i64
          %31 = arith.addi %29, %c1_i64 : i64
          %32 = arith.remsi %31, %c16_i64 : i64
          %33 = arith.index_cast %arg15 : i32 to index
          %c4_i32_19 = arith.constant 4 : i32
          %34 = arith.muli %arg14, %c4_i32_19 : i32
          %35 = arith.addi %34, %arg15 : i32
          %36 = arith.index_cast %35 : i32 to index
          %c0_i32_20 = arith.constant 0 : i32
          %37 = arith.cmpi eq, %arg15, %c0_i32_20 : i32
          scf.if %37 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_23 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
            %subview_24 = memref.subview %reinterpret_cast_3[%23, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_24 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_22 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_22, %true, %c64, %c128, %c256 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_23 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_25 = memref.subview %4[%41, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_23 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c1_i32_21 = arith.constant 1 : i32
          %38 = arith.cmpi eq, %arg15, %c1_i32_21 : i32
          scf.if %38 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_23 = memref.alloc() : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_24 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_25 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_26 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_27 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_28 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_29 = memref.subview %4[%41, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_29 [[0, 1, 2], [3]] : memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_22 : memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_22, %alloc_8 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <max> ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_24 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            hivm.hir.vmax ins(%alloc_5, %alloc_24 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_5, %alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_30 = memref.subview %alloc_15[%42, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_31 = memref.collapse_shape %subview_30 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vexp ins(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_31 : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_22, %alloc_26 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vexp ins(%alloc_27 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <sum> ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_32 = memref.subview %alloc_15[%43, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_33 = memref.collapse_shape %subview_32 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_6, %collapse_shape_33 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_6, %alloc_25 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vbrc ins(%cst : f32) outs(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_28, %alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vcast ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_23 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
            %44 = arith.index_cast %arg15 : i32 to index
            %subview_34 = memref.subview %5[%44, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_35 = memref.collapse_shape %subview_34 [[0, 1, 2], [3]] : memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %alloc_23, %collapse_shape_35 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c2_i32 = arith.constant 2 : i32
          %39 = arith.cmpi eq, %arg15, %c2_i32 : i32
          scf.if %39 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_23 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
            %alloc_24 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_25 = memref.subview %5[%41, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_23 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            %subview_26 = memref.subview %reinterpret_cast_2[%23, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_26 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_22 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 ins(%alloc_23, %alloc_22, %true, %c64, %c256, %c128 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_24 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_27 = memref.subview %6[%42, 0, 0, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_24 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_28 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c3_i32 = arith.constant 3 : i32
          %40 = arith.cmpi eq, %arg15, %c3_i32 : i32
          scf.if %40 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_23 = memref.subview %6[%41, 0, %25, 0] [1, 1, 32, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_23 [[0, 1, 2], [3]] : memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_22 : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_24 = memref.subview %alloc_15[%42, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_25 = memref.collapse_shape %subview_24 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_7, %collapse_shape_25 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vadd ins(%alloc_7, %alloc_22 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
        } {hivm.soft_pipeline}
      } {tilelangir.num_stages = 4 : i32}
      scf.for %arg14 = %c0_i32_10 to %c4_i32_11 step %c1_i32_12  : i32 {
        %c4_i64 = arith.constant 4 : i64
        %22 = arith.extsi %13 : i32 to i64
        %23 = arith.muli %22, %c4_i64 : i64
        %c16_i64 = arith.constant 16 : i64
        %24 = arith.remsi %23, %c16_i64 : i64
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %24
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
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

// -----// IR Dump After BindWorkSpaceArg (hivm-bind-workspace-arg) ('func.func' operation: @flash_attention) //----- //
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
    %4 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf32, #hivm.address_space<gm>>
    %5 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf16, #hivm.address_space<gm>>
    %6 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x128xf32, #hivm.address_space<gm>>
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
      %c4_i32_11 = arith.constant 4 : i32
      %c1_i32_12 = arith.constant 1 : i32
      scf.for %arg14 = %c0_i32_10 to %c4_i32_11 step %c1_i32_12  : i32 {
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
          %26 = arith.extsi %arg14 : i32 to i64
          %27 = arith.extsi %arg15 : i32 to i64
          %c4_i64 = arith.constant 4 : i64
          %c16_i64 = arith.constant 16 : i64
          %28 = arith.muli %26, %c4_i64 : i64
          %29 = arith.addi %28, %27 : i64
          %30 = arith.remsi %29, %c16_i64 : i64
          %c1_i64 = arith.constant 1 : i64
          %31 = arith.addi %29, %c1_i64 : i64
          %32 = arith.remsi %31, %c16_i64 : i64
          %33 = arith.index_cast %arg15 : i32 to index
          %c4_i32_19 = arith.constant 4 : i32
          %34 = arith.muli %arg14, %c4_i32_19 : i32
          %35 = arith.addi %34, %arg15 : i32
          %36 = arith.index_cast %35 : i32 to index
          %c0_i32_20 = arith.constant 0 : i32
          %37 = arith.cmpi eq, %arg15, %c0_i32_20 : i32
          scf.if %37 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_23 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
            %subview_24 = memref.subview %reinterpret_cast_3[%23, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_24 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_22 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_22, %true, %c64, %c128, %c256 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_23 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_25 = memref.subview %4[%41, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_23 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c1_i32_21 = arith.constant 1 : i32
          %38 = arith.cmpi eq, %arg15, %c1_i32_21 : i32
          scf.if %38 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_23 = memref.alloc() : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_24 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_25 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_26 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_27 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_28 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_29 = memref.subview %4[%41, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_29 [[0, 1, 2], [3]] : memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_22 : memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_22, %alloc_8 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <max> ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_24 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            hivm.hir.vmax ins(%alloc_5, %alloc_24 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_5, %alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_30 = memref.subview %alloc_15[%42, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_31 = memref.collapse_shape %subview_30 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vexp ins(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_31 : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_22, %alloc_26 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vexp ins(%alloc_27 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <sum> ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_32 = memref.subview %alloc_15[%43, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_33 = memref.collapse_shape %subview_32 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_6, %collapse_shape_33 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_6, %alloc_25 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vbrc ins(%cst : f32) outs(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_28, %alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vcast ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_23 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
            %44 = arith.index_cast %arg15 : i32 to index
            %subview_34 = memref.subview %5[%44, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_35 = memref.collapse_shape %subview_34 [[0, 1, 2], [3]] : memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %alloc_23, %collapse_shape_35 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c2_i32 = arith.constant 2 : i32
          %39 = arith.cmpi eq, %arg15, %c2_i32 : i32
          scf.if %39 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_23 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
            %alloc_24 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_25 = memref.subview %5[%41, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_23 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            %subview_26 = memref.subview %reinterpret_cast_2[%23, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_26 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_22 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 ins(%alloc_23, %alloc_22, %true, %c64, %c256, %c128 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_24 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_27 = memref.subview %6[%42, 0, 0, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_24 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_28 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c3_i32 = arith.constant 3 : i32
          %40 = arith.cmpi eq, %arg15, %c3_i32 : i32
          scf.if %40 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_23 = memref.subview %6[%41, 0, %25, 0] [1, 1, 32, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_23 [[0, 1, 2], [3]] : memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_22 : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_24 = memref.subview %alloc_15[%42, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_25 = memref.collapse_shape %subview_24 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_7, %collapse_shape_25 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vadd ins(%alloc_7, %alloc_22 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
        } {hivm.soft_pipeline}
      } {tilelangir.num_stages = 4 : i32}
      scf.for %arg14 = %c0_i32_10 to %c4_i32_11 step %c1_i32_12  : i32 {
        %c4_i64 = arith.constant 4 : i64
        %22 = arith.extsi %13 : i32 to i64
        %23 = arith.muli %22, %c4_i64 : i64
        %c16_i64 = arith.constant 16 : i64
        %24 = arith.remsi %23, %c16_i64 : i64
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %24
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
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

// -----// IR Dump After TileLangIRPlanWorkspaceMemory (tilelangir-plan-workspace-memory) ('func.func' operation: @flash_attention) //----- //
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
      %c4_i32_11 = arith.constant 4 : i32
      %c1_i32_12 = arith.constant 1 : i32
      scf.for %arg14 = %c0_i32_10 to %c4_i32_11 step %c1_i32_12  : i32 {
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
          %26 = arith.extsi %arg14 : i32 to i64
          %27 = arith.extsi %arg15 : i32 to i64
          %c4_i64 = arith.constant 4 : i64
          %c16_i64 = arith.constant 16 : i64
          %28 = arith.muli %26, %c4_i64 : i64
          %29 = arith.addi %28, %27 : i64
          %30 = arith.remsi %29, %c16_i64 : i64
          %c1_i64 = arith.constant 1 : i64
          %31 = arith.addi %29, %c1_i64 : i64
          %32 = arith.remsi %31, %c16_i64 : i64
          %33 = arith.index_cast %arg15 : i32 to index
          %c4_i32_19 = arith.constant 4 : i32
          %34 = arith.muli %arg14, %c4_i32_19 : i32
          %35 = arith.addi %34, %arg15 : i32
          %36 = arith.index_cast %35 : i32 to index
          %c0_i32_20 = arith.constant 0 : i32
          %37 = arith.cmpi eq, %arg15, %c0_i32_20 : i32
          scf.if %37 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_23 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
            %subview_24 = memref.subview %reinterpret_cast_3[%23, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_24 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_22 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_22, %true, %c64, %c128, %c256 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_23 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_25 = memref.subview %4[%41, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_23 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c1_i32_21 = arith.constant 1 : i32
          %38 = arith.cmpi eq, %arg15, %c1_i32_21 : i32
          scf.if %38 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_23 = memref.alloc() : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_24 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_25 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_26 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_27 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_28 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_29 = memref.subview %4[%41, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_29 [[0, 1, 2], [3]] : memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_22 : memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_22, %alloc_8 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <max> ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_24 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            hivm.hir.vmax ins(%alloc_5, %alloc_24 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_5, %alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_30 = memref.subview %alloc_15[%42, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_31 = memref.collapse_shape %subview_30 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vexp ins(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_31 : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_22, %alloc_26 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vexp ins(%alloc_27 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <sum> ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_32 = memref.subview %alloc_15[%43, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_33 = memref.collapse_shape %subview_32 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_6, %collapse_shape_33 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_6, %alloc_25 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vbrc ins(%cst : f32) outs(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_28, %alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vcast ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_23 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
            %44 = arith.index_cast %arg15 : i32 to index
            %subview_34 = memref.subview %5[%44, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_35 = memref.collapse_shape %subview_34 [[0, 1, 2], [3]] : memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %alloc_23, %collapse_shape_35 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c2_i32 = arith.constant 2 : i32
          %39 = arith.cmpi eq, %arg15, %c2_i32 : i32
          scf.if %39 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_23 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
            %alloc_24 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_25 = memref.subview %5[%41, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_23 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            %subview_26 = memref.subview %reinterpret_cast_2[%23, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_26 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_22 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 ins(%alloc_23, %alloc_22, %true, %c64, %c256, %c128 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_24 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_27 = memref.subview %6[%42, 0, 0, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_24 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_28 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c3_i32 = arith.constant 3 : i32
          %40 = arith.cmpi eq, %arg15, %c3_i32 : i32
          scf.if %40 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_23 = memref.subview %6[%41, 0, %25, 0] [1, 1, 32, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_23 [[0, 1, 2], [3]] : memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_22 : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_24 = memref.subview %alloc_15[%42, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_25 = memref.collapse_shape %subview_24 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_7, %collapse_shape_25 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vadd ins(%alloc_7, %alloc_22 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
        } {hivm.soft_pipeline}
      } {tilelangir.num_stages = 4 : i32}
      scf.for %arg14 = %c0_i32_10 to %c4_i32_11 step %c1_i32_12  : i32 {
        %c4_i64 = arith.constant 4 : i64
        %22 = arith.extsi %13 : i32 to i64
        %23 = arith.muli %22, %c4_i64 : i64
        %c16_i64 = arith.constant 16 : i64
        %24 = arith.remsi %23, %c16_i64 : i64
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %24
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
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
      %c4_i32_11 = arith.constant 4 : i32
      %c1_i32_12 = arith.constant 1 : i32
      scf.for %arg14 = %c0_i32_10 to %c4_i32_11 step %c1_i32_12  : i32 {
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
          %26 = arith.extsi %arg14 : i32 to i64
          %27 = arith.extsi %arg15 : i32 to i64
          %c4_i64 = arith.constant 4 : i64
          %c16_i64 = arith.constant 16 : i64
          %28 = arith.muli %26, %c4_i64 : i64
          %29 = arith.addi %28, %27 : i64
          %30 = arith.remsi %29, %c16_i64 : i64
          %c1_i64 = arith.constant 1 : i64
          %31 = arith.addi %29, %c1_i64 : i64
          %32 = arith.remsi %31, %c16_i64 : i64
          %33 = arith.index_cast %arg15 : i32 to index
          %c4_i32_19 = arith.constant 4 : i32
          %34 = arith.muli %arg14, %c4_i32_19 : i32
          %35 = arith.addi %34, %arg15 : i32
          %36 = arith.index_cast %35 : i32 to index
          %c0_i32_20 = arith.constant 0 : i32
          %37 = arith.cmpi eq, %arg15, %c0_i32_20 : i32
          scf.if %37 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_23 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
            %subview_24 = memref.subview %reinterpret_cast_3[%23, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_24 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_22 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_22, %true, %c64, %c128, %c256 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_23 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_25 = memref.subview %4[%41, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_23 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c1_i32_21 = arith.constant 1 : i32
          %38 = arith.cmpi eq, %arg15, %c1_i32_21 : i32
          scf.if %38 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_23 = memref.alloc() : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_24 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_25 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_26 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_27 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_28 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_29 = memref.subview %4[%41, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_29 [[0, 1, 2], [3]] : memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_22 : memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_22, %alloc_8 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <max> ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_24 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            hivm.hir.vmax ins(%alloc_5, %alloc_24 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_5, %alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_30 = memref.subview %alloc_15[%42, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_31 = memref.collapse_shape %subview_30 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vexp ins(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_31 : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_22, %alloc_26 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vexp ins(%alloc_27 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <sum> ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_32 = memref.subview %alloc_15[%43, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_33 = memref.collapse_shape %subview_32 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_6, %collapse_shape_33 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_6, %alloc_25 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vbrc ins(%cst : f32) outs(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_28, %alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vcast ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_23 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
            %44 = arith.index_cast %arg15 : i32 to index
            %subview_34 = memref.subview %5[%44, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_35 = memref.collapse_shape %subview_34 [[0, 1, 2], [3]] : memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %alloc_23, %collapse_shape_35 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c2_i32 = arith.constant 2 : i32
          %39 = arith.cmpi eq, %arg15, %c2_i32 : i32
          scf.if %39 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_23 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
            %alloc_24 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_25 = memref.subview %5[%41, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_23 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            %subview_26 = memref.subview %reinterpret_cast_2[%23, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_26 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_22 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 ins(%alloc_23, %alloc_22, %true, %c64, %c256, %c128 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_24 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_27 = memref.subview %6[%42, 0, 0, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_24 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_28 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c3_i32 = arith.constant 3 : i32
          %40 = arith.cmpi eq, %arg15, %c3_i32 : i32
          scf.if %40 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_23 = memref.subview %6[%41, 0, %25, 0] [1, 1, 32, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_23 [[0, 1, 2], [3]] : memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_22 : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_24 = memref.subview %alloc_15[%42, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_25 = memref.collapse_shape %subview_24 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_7, %collapse_shape_25 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vadd ins(%alloc_7, %alloc_22 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
        } {hivm.soft_pipeline}
      } {tilelangir.num_stages = 4 : i32}
      scf.for %arg14 = %c0_i32_10 to %c4_i32_11 step %c1_i32_12  : i32 {
        %c4_i64 = arith.constant 4 : i64
        %22 = arith.extsi %13 : i32 to i64
        %23 = arith.muli %22, %c4_i64 : i64
        %c16_i64 = arith.constant 16 : i64
        %24 = arith.remsi %23, %c16_i64 : i64
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %24
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
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

// -----// IR Dump After InsertInferWorkSpaceSizeFunc (hivm-insert-infer-workspace-size-func) ('func.func' operation: @flash_attention) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c1048576 = arith.constant 1048576 : index
    return %c1048576 : index
  }
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
      %c4_i32_11 = arith.constant 4 : i32
      %c1_i32_12 = arith.constant 1 : i32
      scf.for %arg14 = %c0_i32_10 to %c4_i32_11 step %c1_i32_12  : i32 {
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
          %26 = arith.extsi %arg14 : i32 to i64
          %27 = arith.extsi %arg15 : i32 to i64
          %c4_i64 = arith.constant 4 : i64
          %c16_i64 = arith.constant 16 : i64
          %28 = arith.muli %26, %c4_i64 : i64
          %29 = arith.addi %28, %27 : i64
          %30 = arith.remsi %29, %c16_i64 : i64
          %c1_i64 = arith.constant 1 : i64
          %31 = arith.addi %29, %c1_i64 : i64
          %32 = arith.remsi %31, %c16_i64 : i64
          %33 = arith.index_cast %arg15 : i32 to index
          %c4_i32_19 = arith.constant 4 : i32
          %34 = arith.muli %arg14, %c4_i32_19 : i32
          %35 = arith.addi %34, %arg15 : i32
          %36 = arith.index_cast %35 : i32 to index
          %c0_i32_20 = arith.constant 0 : i32
          %37 = arith.cmpi eq, %arg15, %c0_i32_20 : i32
          scf.if %37 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_23 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
            %subview_24 = memref.subview %reinterpret_cast_3[%23, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_24 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_22 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_22, %true, %c64, %c128, %c256 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_23 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_25 = memref.subview %4[%41, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_23 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c1_i32_21 = arith.constant 1 : i32
          %38 = arith.cmpi eq, %arg15, %c1_i32_21 : i32
          scf.if %38 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_23 = memref.alloc() : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_24 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_25 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_26 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_27 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_28 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_29 = memref.subview %4[%41, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_29 [[0, 1, 2], [3]] : memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_22 : memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_22, %alloc_8 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <max> ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_24 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            hivm.hir.vmax ins(%alloc_5, %alloc_24 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_5, %alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_30 = memref.subview %alloc_15[%42, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_31 = memref.collapse_shape %subview_30 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vexp ins(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_31 : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_22, %alloc_26 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vexp ins(%alloc_27 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <sum> ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_32 = memref.subview %alloc_15[%43, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_33 = memref.collapse_shape %subview_32 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_6, %collapse_shape_33 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_6, %alloc_25 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vbrc ins(%cst : f32) outs(%alloc_28 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_28, %alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vcast ins(%alloc_22 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_23 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
            %44 = arith.index_cast %arg15 : i32 to index
            %subview_34 = memref.subview %5[%44, 0, %25, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_35 = memref.collapse_shape %subview_34 [[0, 1, 2], [3]] : memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %alloc_23, %collapse_shape_35 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c2_i32 = arith.constant 2 : i32
          %39 = arith.cmpi eq, %arg15, %c2_i32 : i32
          scf.if %39 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_23 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
            %alloc_24 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_25 = memref.subview %5[%41, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_23 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            %subview_26 = memref.subview %reinterpret_cast_2[%23, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_26 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_22 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 ins(%alloc_23, %alloc_22, %true, %c64, %c256, %c128 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_24 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_27 = memref.subview %6[%42, 0, 0, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_24 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_28 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %c3_i32 = arith.constant 3 : i32
          %40 = arith.cmpi eq, %arg15, %c3_i32 : i32
          scf.if %40 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %30
            %alloc_22 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %41 = arith.index_cast %arg15 : i32 to index
            %subview_23 = memref.subview %6[%41, 0, %25, 0] [1, 1, 32, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_23 [[0, 1, 2], [3]] : memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_22 : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_24 = memref.subview %alloc_15[%42, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_25 = memref.collapse_shape %subview_24 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_7, %collapse_shape_25 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vadd ins(%alloc_7, %alloc_22 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
        } {hivm.soft_pipeline}
      } {tilelangir.num_stages = 4 : i32}
      scf.for %arg14 = %c0_i32_10 to %c4_i32_11 step %c1_i32_12  : i32 {
        %c4_i64 = arith.constant 4 : i64
        %22 = arith.extsi %13 : i32 to i64
        %23 = arith.muli %22, %c4_i64 : i64
        %c16_i64 = arith.constant 16 : i64
        %24 = arith.remsi %23, %c16_i64 : i64
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %24
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
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

// -----// IR Dump After LowerMemRefExt (lower-memref-ext) ('builtin.module' operation) //----- //
#map = affine_map<(d0)[s0] -> (d0 * 1048576 + s0)>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c1048576 = arith.constant 1048576 : index
    return %c1048576 : index
  }
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c0_i64 = arith.constant 0 : i64
    %c8_i32 = arith.constant 8 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %c4_i32 = arith.constant 4 : i32
    %c786432 = arith.constant 786432 : index
    %c524288 = arith.constant 524288 : index
    %c0 = arith.constant 0 : index
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
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply #map(%5)[%c0]
    %view = memref.view %arg2[%6][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf32, #hivm.address_space<gm>>
    %7 = hivm.hir.get_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply #map(%8)[%c524288]
    %view_5 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf16, #hivm.address_space<gm>>
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply #map(%11)[%c786432]
    %view_6 = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x128xf32, #hivm.address_space<gm>>
    %13 = arith.subi %c151_i32, %1 : i32
    %14 = arith.divsi %13, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %14 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %alloc_7 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_9 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %15 = arith.muli %arg13, %c1536_i32 : i32
      %16 = arith.muli %1, %c64_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview = memref.subview %reinterpret_cast[%18, 0] [64, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_9 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_8 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_7 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %27 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %27 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %alloc_14 = memref.alloc() : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>>
        %27 = arith.muli %arg14, %c256_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.muli %3, %c32_i32 : i32
        %30 = arith.index_cast %29 : i32 to index
        scf.for %arg15 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
          %31 = arith.extsi %arg14 : i32 to i64
          %32 = arith.extsi %arg15 : i32 to i64
          %33 = arith.muli %31, %c4_i64 : i64
          %34 = arith.addi %33, %32 : i64
          %35 = arith.remsi %34, %c16_i64 : i64
          %36 = arith.addi %34, %c1_i64 : i64
          %37 = arith.remsi %36, %c16_i64 : i64
          %38 = arith.cmpi eq, %arg15, %c0_i32 : i32
          scf.if %38 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %35
            %alloc_15 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_16 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
            %subview_17 = memref.subview %reinterpret_cast_3[%28, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_17 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_15 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_15, %true, %c64, %c128, %c256 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_16 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_18 = memref.subview %view[%42, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_18 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %39 = arith.cmpi eq, %arg15, %c1_i32 : i32
          scf.if %39 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %35
            %alloc_15 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_16 = memref.alloc() : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_17 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_18 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_19 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_20 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_21 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_22 = memref.subview %view[%42, 0, %30, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_22 [[0, 1, 2], [3]] : memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_15 : memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_15, %alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <max> ins(%alloc_15 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_17 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            hivm.hir.vmax ins(%alloc_7, %alloc_17 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_19 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_7, %alloc_19 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_21 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_23 = memref.subview %alloc_14[%43, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_24 = memref.collapse_shape %subview_23 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vexp ins(%alloc_21 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_24 : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_15, %alloc_19 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_20 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vexp ins(%alloc_20 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <sum> ins(%alloc_15 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_18 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            %44 = arith.index_cast %arg15 : i32 to index
            %subview_25 = memref.subview %alloc_14[%44, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_26 = memref.collapse_shape %subview_25 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_8, %collapse_shape_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_8, %alloc_18 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vbrc ins(%cst : f32) outs(%alloc_21 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_21, %alloc_19 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vcast ins(%alloc_15 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
            %45 = arith.index_cast %arg15 : i32 to index
            %subview_27 = memref.subview %view_5[%45, 0, %30, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %alloc_16, %collapse_shape_28 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %40 = arith.cmpi eq, %arg15, %c2_i32 : i32
          scf.if %40 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %35
            %alloc_15 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_16 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
            %alloc_17 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_18 = memref.subview %view_5[%42, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_18 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_16 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            %subview_19 = memref.subview %reinterpret_cast_2[%28, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_19 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_15 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 ins(%alloc_16, %alloc_15, %true, %c64, %c256, %c128 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_17 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_20 = memref.subview %view_6[%43, 0, 0, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_21 = memref.collapse_shape %subview_20 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_17 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_21 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %41 = arith.cmpi eq, %arg15, %c3_i32 : i32
          scf.if %41 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %35
            %alloc_15 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_16 = memref.subview %view_6[%42, 0, %30, 0] [1, 1, 32, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_16 [[0, 1, 2], [3]] : memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_15 : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_17 = memref.subview %alloc_14[%43, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_18 = memref.collapse_shape %subview_17 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_9, %collapse_shape_18 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vadd ins(%alloc_9, %alloc_15 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
        } {hivm.soft_pipeline}
      } {tilelangir.num_stages = 4 : i32}
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %c0_i64
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      hivm.hir.vdiv ins(%alloc_9, %alloc_8 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_9 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %19 = arith.muli %3, %c32_i32 : i32
      %20 = arith.subi %c8192_i32, %19 : i32
      %21 = arith.subi %20, %16 : i32
      %22 = arith.subi %21, %15 : i32
      %23 = arith.minsi %22, %c32_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview_12 = memref.subview %alloc_11[0, 0] [%24, 128] [1, 1] : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %25 = arith.addi %17, %19 : i32
      %26 = arith.index_cast %25 : i32 to index
      %subview_13 = memref.subview %reinterpret_cast_4[%26, 0] [%24, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_12, %subview_13 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}

// -----// IR Dump After TileLangIRSplitMixKernel (tilelangir-split-mix-kernel) ('builtin.module' operation) //----- //
#map = affine_map<(d0)[s0] -> (d0 * 1048576 + s0)>
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @flash_attention_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c1048576 = arith.constant 1048576 : index
    return %c1048576 : index
  }
  func.func @flash_attention_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0_i64 = arith.constant 0 : i64
    %c8_i32 = arith.constant 8 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %c4_i32 = arith.constant 4 : i32
    %c786432 = arith.constant 786432 : index
    %c524288 = arith.constant 524288 : index
    %c0 = arith.constant 0 : index
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
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply #map(%5)[%c0]
    %view = memref.view %arg2[%6][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf32, #hivm.address_space<gm>>
    %7 = hivm.hir.get_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply #map(%8)[%c524288]
    %view_4 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf16, #hivm.address_space<gm>>
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply #map(%11)[%c786432]
    %view_5 = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x128xf32, #hivm.address_space<gm>>
    %13 = arith.subi %c151_i32, %1 : i32
    %14 = arith.divsi %13, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %14 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %15 = arith.muli %arg13, %c1536_i32 : i32
      %16 = arith.muli %1, %c64_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview = memref.subview %reinterpret_cast[%18, 0] [64, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %27 = arith.muli %arg14, %c256_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.muli %3, %c32_i32 : i32
        %30 = arith.index_cast %29 : i32 to index
        scf.for %arg15 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
          %31 = arith.extsi %arg14 : i32 to i64
          %32 = arith.extsi %arg15 : i32 to i64
          %33 = arith.muli %31, %c4_i64 : i64
          %34 = arith.addi %33, %32 : i64
          %35 = arith.remsi %34, %c16_i64 : i64
          %36 = arith.addi %34, %c1_i64 : i64
          %37 = arith.remsi %36, %c16_i64 : i64
          %38 = arith.cmpi eq, %arg15, %c0_i32 : i32
          scf.if %38 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %35
            %alloc_6 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_7 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
            %subview_8 = memref.subview %reinterpret_cast_3[%28, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_8 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_6 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_6, %true, %c64, %c128, %c256 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_7 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_9 = memref.subview %view[%42, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_7 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %39 = arith.cmpi eq, %arg15, %c1_i32 : i32
          scf.if %39 {
          }
          %40 = arith.cmpi eq, %arg15, %c2_i32 : i32
          scf.if %40 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %35
            %alloc_6 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_7 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
            %alloc_8 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_9 = memref.subview %view_4[%42, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_7 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            %subview_10 = memref.subview %reinterpret_cast_2[%28, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_10 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_6 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 ins(%alloc_7, %alloc_6, %true, %c64, %c256, %c128 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_8 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_11 = memref.subview %view_5[%43, 0, 0, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_12 = memref.collapse_shape %subview_11 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_8 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_12 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %41 = arith.cmpi eq, %arg15, %c3_i32 : i32
          scf.if %41 {
          }
        } {hivm.soft_pipeline}
      } {tilelangir.num_stages = 4 : i32}
      %19 = arith.muli %3, %c32_i32 : i32
      %20 = arith.subi %c8192_i32, %19 : i32
      %21 = arith.subi %20, %16 : i32
      %22 = arith.subi %21, %15 : i32
      %23 = arith.minsi %22, %c32_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.addi %17, %19 : i32
      %26 = arith.index_cast %25 : i32 to index
    }
    return
  }
  func.func @flash_attention_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0_i64 = arith.constant 0 : i64
    %c8_i32 = arith.constant 8 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %c4_i32 = arith.constant 4 : i32
    %c786432 = arith.constant 786432 : index
    %c524288 = arith.constant 524288 : index
    %c0 = arith.constant 0 : index
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
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply #map(%5)[%c0]
    %view = memref.view %arg2[%6][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf32, #hivm.address_space<gm>>
    %7 = hivm.hir.get_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply #map(%8)[%c524288]
    %view_2 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf16, #hivm.address_space<gm>>
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply #map(%11)[%c786432]
    %view_3 = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x128xf32, #hivm.address_space<gm>>
    %13 = arith.subi %c151_i32, %1 : i32
    %14 = arith.divsi %13, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %14 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_4 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_5 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %15 = arith.muli %arg13, %c1536_i32 : i32
      %16 = arith.muli %1, %c64_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_5 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_4 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_6 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %27 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %27 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %alloc_9 = memref.alloc() : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>>
        %27 = arith.muli %arg14, %c256_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.muli %3, %c32_i32 : i32
        %30 = arith.index_cast %29 : i32 to index
        scf.for %arg15 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
          %31 = arith.extsi %arg14 : i32 to i64
          %32 = arith.extsi %arg15 : i32 to i64
          %33 = arith.muli %31, %c4_i64 : i64
          %34 = arith.addi %33, %32 : i64
          %35 = arith.remsi %34, %c16_i64 : i64
          %36 = arith.addi %34, %c1_i64 : i64
          %37 = arith.remsi %36, %c16_i64 : i64
          %38 = arith.cmpi eq, %arg15, %c0_i32 : i32
          scf.if %38 {
          }
          %39 = arith.cmpi eq, %arg15, %c1_i32 : i32
          scf.if %39 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %35
            %alloc_10 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_11 = memref.alloc() : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_12 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_13 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_14 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_15 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_16 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_17 = memref.subview %view[%42, 0, %30, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_17 [[0, 1, 2], [3]] : memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_10 : memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_10, %alloc_6 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <max> ins(%alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            hivm.hir.vmax ins(%alloc, %alloc_12 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc, %alloc_14 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_18 = memref.subview %alloc_9[%43, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_19 = memref.collapse_shape %subview_18 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vexp ins(%alloc_16 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_10, %alloc_14 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vexp ins(%alloc_15 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <sum> ins(%alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            %44 = arith.index_cast %arg15 : i32 to index
            %subview_20 = memref.subview %alloc_9[%44, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_21 = memref.collapse_shape %subview_20 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_4, %collapse_shape_21 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_4, %alloc_13 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vbrc ins(%cst : f32) outs(%alloc_16 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_16, %alloc_14 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vcast ins(%alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
            %45 = arith.index_cast %arg15 : i32 to index
            %subview_22 = memref.subview %view_2[%45, 0, %30, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_23 = memref.collapse_shape %subview_22 [[0, 1, 2], [3]] : memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %alloc_11, %collapse_shape_23 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %40 = arith.cmpi eq, %arg15, %c2_i32 : i32
          scf.if %40 {
          }
          %41 = arith.cmpi eq, %arg15, %c3_i32 : i32
          scf.if %41 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %35
            %alloc_10 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_11 = memref.subview %view_3[%42, 0, %30, 0] [1, 1, 32, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_11 [[0, 1, 2], [3]] : memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_10 : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_12 = memref.subview %alloc_9[%43, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_13 = memref.collapse_shape %subview_12 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_5, %collapse_shape_13 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vadd ins(%alloc_5, %alloc_10 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
        } {hivm.soft_pipeline}
      } {tilelangir.num_stages = 4 : i32}
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %c0_i64
      }
      hivm.hir.vdiv ins(%alloc_5, %alloc_4 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_5 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %19 = arith.muli %3, %c32_i32 : i32
      %20 = arith.subi %c8192_i32, %19 : i32
      %21 = arith.subi %20, %16 : i32
      %22 = arith.subi %21, %15 : i32
      %23 = arith.minsi %22, %c32_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %alloc_7[0, 0] [%24, 128] [1, 1] : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %25 = arith.addi %17, %19 : i32
      %26 = arith.index_cast %25 : i32 to index
      %subview_8 = memref.subview %reinterpret_cast[%26, 0] [%24, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %subview_8 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}

// -----// IR Dump After TileLangIRWrapHostFunction (tilelangir-wrap-host-function) ('builtin.module' operation) //----- //
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
    %c0_i64 = arith.constant 0 : i64
    %c8_i32 = arith.constant 8 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %c4_i32 = arith.constant 4 : i32
    %c786432 = arith.constant 786432 : index
    %c524288 = arith.constant 524288 : index
    %c0 = arith.constant 0 : index
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
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply #map(%5)[%c0]
    %view = memref.view %arg2[%6][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf32, #hivm.address_space<gm>>
    %7 = hivm.hir.get_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply #map(%8)[%c524288]
    %view_4 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf16, #hivm.address_space<gm>>
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply #map(%11)[%c786432]
    %view_5 = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x128xf32, #hivm.address_space<gm>>
    %13 = arith.subi %c151_i32, %1 : i32
    %14 = arith.divsi %13, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %14 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %15 = arith.muli %arg13, %c1536_i32 : i32
      %16 = arith.muli %1, %c64_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview = memref.subview %reinterpret_cast[%18, 0] [64, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %27 = arith.muli %arg14, %c256_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.muli %3, %c32_i32 : i32
        %30 = arith.index_cast %29 : i32 to index
        scf.for %arg15 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
          %31 = arith.extsi %arg14 : i32 to i64
          %32 = arith.extsi %arg15 : i32 to i64
          %33 = arith.muli %31, %c4_i64 : i64
          %34 = arith.addi %33, %32 : i64
          %35 = arith.remsi %34, %c16_i64 : i64
          %36 = arith.addi %34, %c1_i64 : i64
          %37 = arith.remsi %36, %c16_i64 : i64
          %38 = arith.cmpi eq, %arg15, %c0_i32 : i32
          scf.if %38 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %35
            %alloc_6 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_7 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
            %subview_8 = memref.subview %reinterpret_cast_3[%28, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_8 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_6 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_6, %true, %c64, %c128, %c256 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_7 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_9 = memref.subview %view[%42, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_7 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %39 = arith.cmpi eq, %arg15, %c1_i32 : i32
          scf.if %39 {
          }
          %40 = arith.cmpi eq, %arg15, %c2_i32 : i32
          scf.if %40 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %35
            %alloc_6 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_7 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
            %alloc_8 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_9 = memref.subview %view_4[%42, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_7 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            %subview_10 = memref.subview %reinterpret_cast_2[%28, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_10 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_6 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 ins(%alloc_7, %alloc_6, %true, %c64, %c256, %c128 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_8 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_11 = memref.subview %view_5[%43, 0, 0, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_12 = memref.collapse_shape %subview_11 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_8 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_12 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %41 = arith.cmpi eq, %arg15, %c3_i32 : i32
          scf.if %41 {
          }
        } {hivm.soft_pipeline}
      } {tilelangir.num_stages = 4 : i32}
      %19 = arith.muli %3, %c32_i32 : i32
      %20 = arith.subi %c8192_i32, %19 : i32
      %21 = arith.subi %20, %16 : i32
      %22 = arith.subi %21, %15 : i32
      %23 = arith.minsi %22, %c32_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.addi %17, %19 : i32
      %26 = arith.index_cast %25 : i32 to index
    }
    return
  }
  func.func @flash_attention_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0_i64 = arith.constant 0 : i64
    %c8_i32 = arith.constant 8 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %c4_i32 = arith.constant 4 : i32
    %c786432 = arith.constant 786432 : index
    %c524288 = arith.constant 524288 : index
    %c0 = arith.constant 0 : index
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
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply #map(%5)[%c0]
    %view = memref.view %arg2[%6][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf32, #hivm.address_space<gm>>
    %7 = hivm.hir.get_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply #map(%8)[%c524288]
    %view_2 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf16, #hivm.address_space<gm>>
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply #map(%11)[%c786432]
    %view_3 = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x128xf32, #hivm.address_space<gm>>
    %13 = arith.subi %c151_i32, %1 : i32
    %14 = arith.divsi %13, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %14 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_4 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_5 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %15 = arith.muli %arg13, %c1536_i32 : i32
      %16 = arith.muli %1, %c64_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_5 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_4 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_6 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %27 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %27 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %alloc_9 = memref.alloc() : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>>
        %27 = arith.muli %arg14, %c256_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.muli %3, %c32_i32 : i32
        %30 = arith.index_cast %29 : i32 to index
        scf.for %arg15 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
          %31 = arith.extsi %arg14 : i32 to i64
          %32 = arith.extsi %arg15 : i32 to i64
          %33 = arith.muli %31, %c4_i64 : i64
          %34 = arith.addi %33, %32 : i64
          %35 = arith.remsi %34, %c16_i64 : i64
          %36 = arith.addi %34, %c1_i64 : i64
          %37 = arith.remsi %36, %c16_i64 : i64
          %38 = arith.cmpi eq, %arg15, %c0_i32 : i32
          scf.if %38 {
          }
          %39 = arith.cmpi eq, %arg15, %c1_i32 : i32
          scf.if %39 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %35
            %alloc_10 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_11 = memref.alloc() : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_12 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_13 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_14 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_15 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_16 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_17 = memref.subview %view[%42, 0, %30, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_17 [[0, 1, 2], [3]] : memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_10 : memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_10, %alloc_6 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <max> ins(%alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            hivm.hir.vmax ins(%alloc, %alloc_12 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc, %alloc_14 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_18 = memref.subview %alloc_9[%43, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_19 = memref.collapse_shape %subview_18 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vexp ins(%alloc_16 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_10, %alloc_14 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vexp ins(%alloc_15 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <sum> ins(%alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            %44 = arith.index_cast %arg15 : i32 to index
            %subview_20 = memref.subview %alloc_9[%44, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_21 = memref.collapse_shape %subview_20 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_4, %collapse_shape_21 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_4, %alloc_13 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vbrc ins(%cst : f32) outs(%alloc_16 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_16, %alloc_14 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vcast ins(%alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
            %45 = arith.index_cast %arg15 : i32 to index
            %subview_22 = memref.subview %view_2[%45, 0, %30, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_23 = memref.collapse_shape %subview_22 [[0, 1, 2], [3]] : memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %alloc_11, %collapse_shape_23 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %40 = arith.cmpi eq, %arg15, %c2_i32 : i32
          scf.if %40 {
          }
          %41 = arith.cmpi eq, %arg15, %c3_i32 : i32
          scf.if %41 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %35
            %alloc_10 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_11 = memref.subview %view_3[%42, 0, %30, 0] [1, 1, 32, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_11 [[0, 1, 2], [3]] : memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_10 : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_12 = memref.subview %alloc_9[%43, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_13 = memref.collapse_shape %subview_12 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_5, %collapse_shape_13 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vadd ins(%alloc_5, %alloc_10 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
        } {hivm.soft_pipeline}
      } {tilelangir.num_stages = 4 : i32}
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %c0_i64
      }
      hivm.hir.vdiv ins(%alloc_5, %alloc_4 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_5 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %19 = arith.muli %3, %c32_i32 : i32
      %20 = arith.subi %c8192_i32, %19 : i32
      %21 = arith.subi %20, %16 : i32
      %22 = arith.subi %21, %15 : i32
      %23 = arith.minsi %22, %c32_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %alloc_7[0, 0] [%24, 128] [1, 1] : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %25 = arith.addi %17, %19 : i32
      %26 = arith.index_cast %25 : i32 to index
      %subview_8 = memref.subview %reinterpret_cast[%26, 0] [%24, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %subview_8 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}

====== final npuir ======
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
    %c0_i64 = arith.constant 0 : i64
    %c8_i32 = arith.constant 8 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %c4_i32 = arith.constant 4 : i32
    %c786432 = arith.constant 786432 : index
    %c524288 = arith.constant 524288 : index
    %c0 = arith.constant 0 : index
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
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply affine_map<(d0)[s0] -> (d0 * 1048576 + s0)>(%5)[%c0]
    %view = memref.view %arg2[%6][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf32, #hivm.address_space<gm>>
    %7 = hivm.hir.get_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply affine_map<(d0)[s0] -> (d0 * 1048576 + s0)>(%8)[%c524288]
    %view_4 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf16, #hivm.address_space<gm>>
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply affine_map<(d0)[s0] -> (d0 * 1048576 + s0)>(%11)[%c786432]
    %view_5 = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x128xf32, #hivm.address_space<gm>>
    %13 = arith.subi %c151_i32, %1 : i32
    %14 = arith.divsi %13, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %14 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %15 = arith.muli %arg13, %c1536_i32 : i32
      %16 = arith.muli %1, %c64_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview = memref.subview %reinterpret_cast[%18, 0] [64, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %27 = arith.muli %arg14, %c256_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.muli %3, %c32_i32 : i32
        %30 = arith.index_cast %29 : i32 to index
        scf.for %arg15 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
          %31 = arith.extsi %arg14 : i32 to i64
          %32 = arith.extsi %arg15 : i32 to i64
          %33 = arith.muli %31, %c4_i64 : i64
          %34 = arith.addi %33, %32 : i64
          %35 = arith.remsi %34, %c16_i64 : i64
          %36 = arith.addi %34, %c1_i64 : i64
          %37 = arith.remsi %36, %c16_i64 : i64
          %38 = arith.cmpi eq, %arg15, %c0_i32 : i32
          scf.if %38 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %35
            %alloc_6 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_7 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
            %subview_8 = memref.subview %reinterpret_cast_3[%28, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_8 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_6 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_6, %true, %c64, %c128, %c256 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_7 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_9 = memref.subview %view[%42, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_7 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %39 = arith.cmpi eq, %arg15, %c1_i32 : i32
          scf.if %39 {
          }
          %40 = arith.cmpi eq, %arg15, %c2_i32 : i32
          scf.if %40 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %35
            %alloc_6 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
            %alloc_7 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
            %alloc_8 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_9 = memref.subview %view_4[%42, 0, 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_7 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            %subview_10 = memref.subview %reinterpret_cast_2[%28, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.nd2nz {dst_continuous} ins(%subview_10 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_6 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
            hivm.hir.mmadL1 ins(%alloc_7, %alloc_6, %true, %c64, %c256, %c128 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_8 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_11 = memref.subview %view_5[%43, 0, 0, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_12 = memref.collapse_shape %subview_11 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_8 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_12 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %41 = arith.cmpi eq, %arg15, %c3_i32 : i32
          scf.if %41 {
          }
        } {hivm.soft_pipeline}
      } {tilelangir.num_stages = 4 : i32}
      %19 = arith.muli %3, %c32_i32 : i32
      %20 = arith.subi %c8192_i32, %19 : i32
      %21 = arith.subi %20, %16 : i32
      %22 = arith.subi %21, %15 : i32
      %23 = arith.minsi %22, %c32_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.addi %17, %19 : i32
      %26 = arith.index_cast %25 : i32 to index
    }
    return
  }
  func.func @flash_attention_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0_i64 = arith.constant 0 : i64
    %c8_i32 = arith.constant 8 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %c4_i32 = arith.constant 4 : i32
    %c786432 = arith.constant 786432 : index
    %c524288 = arith.constant 524288 : index
    %c0 = arith.constant 0 : index
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
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply affine_map<(d0)[s0] -> (d0 * 1048576 + s0)>(%5)[%c0]
    %view = memref.view %arg2[%6][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf32, #hivm.address_space<gm>>
    %7 = hivm.hir.get_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply affine_map<(d0)[s0] -> (d0 * 1048576 + s0)>(%8)[%c524288]
    %view_2 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x256xf16, #hivm.address_space<gm>>
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply affine_map<(d0)[s0] -> (d0 * 1048576 + s0)>(%11)[%c786432]
    %view_3 = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x2x64x128xf32, #hivm.address_space<gm>>
    %13 = arith.subi %c151_i32, %1 : i32
    %14 = arith.divsi %13, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %14 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_4 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_5 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %15 = arith.muli %arg13, %c1536_i32 : i32
      %16 = arith.muli %1, %c64_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_5 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_4 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_6 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %27 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %27 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %alloc_9 = memref.alloc() : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>>
        %27 = arith.muli %arg14, %c256_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.muli %3, %c32_i32 : i32
        %30 = arith.index_cast %29 : i32 to index
        scf.for %arg15 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
          %31 = arith.extsi %arg14 : i32 to i64
          %32 = arith.extsi %arg15 : i32 to i64
          %33 = arith.muli %31, %c4_i64 : i64
          %34 = arith.addi %33, %32 : i64
          %35 = arith.remsi %34, %c16_i64 : i64
          %36 = arith.addi %34, %c1_i64 : i64
          %37 = arith.remsi %36, %c16_i64 : i64
          %38 = arith.cmpi eq, %arg15, %c0_i32 : i32
          scf.if %38 {
          }
          %39 = arith.cmpi eq, %arg15, %c1_i32 : i32
          scf.if %39 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %35
            %alloc_10 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_11 = memref.alloc() : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_12 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_13 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_14 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %alloc_15 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            %alloc_16 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_17 = memref.subview %view[%42, 0, %30, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf32, #hivm.address_space<gm>> to memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_17 [[0, 1, 2], [3]] : memref<1x1x32x256xf32, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_10 : memref<32x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_10, %alloc_6 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <max> ins(%alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            hivm.hir.vmax ins(%alloc, %alloc_12 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc, %alloc_14 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_18 = memref.subview %alloc_9[%43, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_19 = memref.collapse_shape %subview_18 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vexp ins(%alloc_16 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
            hivm.hir.vsub ins(%alloc_10, %alloc_14 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vexp ins(%alloc_15 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vreduce <sum> ins(%alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
            %44 = arith.index_cast %arg15 : i32 to index
            %subview_20 = memref.subview %alloc_9[%44, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_21 = memref.collapse_shape %subview_20 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_4, %collapse_shape_21 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_4, %alloc_13 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vbrc ins(%cst : f32) outs(%alloc_16 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vadd ins(%alloc_16, %alloc_14 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
            hivm.hir.vcast ins(%alloc_10 : memref<32x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
            %45 = arith.index_cast %arg15 : i32 to index
            %subview_22 = memref.subview %view_2[%45, 0, %30, 0] [1, 1, 32, 256] [1, 1, 1, 1] : memref<4x2x64x256xf16, #hivm.address_space<gm>> to memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_23 = memref.collapse_shape %subview_22 [[0, 1, 2], [3]] : memref<1x1x32x256xf16, strided<[32768, 16384, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %alloc_11, %collapse_shape_23 : memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<32x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
          %40 = arith.cmpi eq, %arg15, %c2_i32 : i32
          scf.if %40 {
          }
          %41 = arith.cmpi eq, %arg15, %c3_i32 : i32
          scf.if %41 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %35
            %alloc_10 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %42 = arith.index_cast %arg15 : i32 to index
            %subview_11 = memref.subview %view_3[%42, 0, %30, 0] [1, 1, 32, 128] [1, 1, 1, 1] : memref<4x2x64x128xf32, #hivm.address_space<gm>> to memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_11 [[0, 1, 2], [3]] : memref<1x1x32x128xf32, strided<[16384, 8192, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            memref.copy %collapse_shape, %alloc_10 : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %43 = arith.index_cast %arg15 : i32 to index
            %subview_12 = memref.subview %alloc_9[%43, 0, 0] [1, 32, 1] [1, 1, 1] : memref<4x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
            %collapse_shape_13 = memref.collapse_shape %subview_12 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.vmul ins(%alloc_5, %collapse_shape_13 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
            hivm.hir.vadd ins(%alloc_5, %alloc_10 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
          }
        } {hivm.soft_pipeline}
      } {tilelangir.num_stages = 4 : i32}
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %c0_i64
      }
      hivm.hir.vdiv ins(%alloc_5, %alloc_4 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_5 : memref<32x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %19 = arith.muli %3, %c32_i32 : i32
      %20 = arith.subi %c8192_i32, %19 : i32
      %21 = arith.subi %20, %16 : i32
      %22 = arith.subi %21, %15 : i32
      %23 = arith.minsi %22, %c32_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %alloc_7[0, 0] [%24, 128] [1, 1] : memref<32x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %25 = arith.addi %17, %19 : i32
      %26 = arith.index_cast %25 : i32 to index
      %subview_8 = memref.subview %reinterpret_cast[%26, 0] [%24, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %subview_8 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}