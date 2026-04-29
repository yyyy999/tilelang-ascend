module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
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
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32

    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>

    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32

    %4 = memref_ext.alloc_workspace() : memref<2x2x128x256xf32, #hivm.address_space<gm>>
    %5 = memref_ext.alloc_workspace() : memref<2x2x128x256xf16, #hivm.address_space<gm>>
    %6 = memref_ext.alloc_workspace() : memref<2x2x128x128xf32, #hivm.address_space<gm>>

    %7 = arith.subi %c87_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32

    scf.for %arg13 = %c0_i32 to %8 step %c1_i32 : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>

      %9 = arith.muli %arg13, %c3072_i32 : i32
      %10 = arith.muli %1, %c128_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %loop_inner_bound = arith.addi %c32_i32, %c1_i32 : i32

      %subview = memref.subview %reinterpret_cast[%12, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>

      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)

      scf.for %arg14 = %c0_i32 to %loop_inner_bound step %c1_i32 : i32 {
        %alloc_11 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        // 移除引发维度错误的 multi_buffer
        // annotation.mark %alloc_11 {hivm.multi_buffer = 8 : i32} : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>

        %21 = arith.muli %arg14, %c256_i32 : i32
        %22 = arith.index_cast %21 : i32 to index
        %23 = arith.muli %3, %c64_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %25 = arith.cmpi slt, %arg14, %c32_i32 : i32
        %26 = arith.cmpi sgt, %arg14, %c0_i32 : i32
        %27 = arith.remsi %arg14, %c2_i32 : i32
        %28 = arith.index_cast %27 : i32 to index

        // ====================== CUBE: Q @ K^T ======================
        scf.if %25 {
          %alloc_12 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_13 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
          %subview_14 = memref.subview %reinterpret_cast_3[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_14, %alloc_12 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>

          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_12, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_13 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)

          %subview_15 = memref.subview %4[%28, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<2x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_15 [[0, 1, 2], [3]] : memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_13, %collapse_shape : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>> to memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}

        // ====================== VECTOR: Scale + Softmax ======================
        scf.if %25 {
          %alloc_12 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_13 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_14 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_15 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_16 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_17 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>

          %subview_18 = memref.subview %4[%28, 0, %24, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<2x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_18 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_19 = memref.subview %5[%28, 0, %24, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<2x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_1 = memref.collapse_shape %subview_19 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>

          memref.copy %collapse_shape, %alloc_12 : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_12, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc_5, %alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_12, %alloc_16 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]

          memref.copy %alloc_13, %collapse_shape_1 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.vsub ins(%alloc_5, %alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          // 修复：输入输出均为 2维，rank 严格一致
          hivm.hir.vexp ins(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vmul ins(%alloc_6, %alloc_11 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_6, %alloc_15 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_17, %alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

        // ====================== CUBE: Attn @ V ======================
        scf.if %26 {
          %alloc_12 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_13 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %alloc_14 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>

          %subview_15 = memref.subview %5[%28, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<2x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_15 [[0, 1, 2], [3]] : memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_13 : memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>

          %subview_16 = memref.subview %reinterpret_cast_2[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_16, %alloc_12 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>

          hivm.hir.mmadL1 ins(%alloc_13, %alloc_12, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_14 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)

          %subview_17 = memref.subview %6[%28, 0, 0, 0] [1, 1, 128, 128] [1, 1, 1, 1] : memref<2x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_18 = memref.collapse_shape %subview_17 [[0, 1, 2], [3]] : memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_14, %collapse_shape_18 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>> to memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}

        // ====================== VECTOR: Accumulate ======================
        scf.if %26 {
          %alloc_12 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          %subview_13 = memref.subview %6[%28, 0, %24, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<2x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_13 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>

          memref.copy %collapse_shape, %alloc_12 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_7, %alloc_11 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_7, %alloc_12 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 2 : i32}

      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)

      %sub_val = arith.muli %3, %c64_i32 : i32
      %14 = arith.subi %c8192_i32, %sub_val : i32
      %15 = arith.subi %14, %10 : i32
      %16 = arith.subi %15, %9 : i32
      %17 = arith.minsi %16, %c64_i32 : i32
      %18 = arith.index_cast %17 : i32 to index

      %subview_9 = memref.subview %alloc_8[0, 0] [%18, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %19 = arith.addi %11, %sub_val : i32
      %20 = arith.index_cast %19 : i32 to index
      %subview_10 = memref.subview %reinterpret_cast_4[%20, 0] [%18, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_9, %subview_10 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }

    return
  }
}