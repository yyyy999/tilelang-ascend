// -----// IR Dump After TileLangIRCVSplit (tilelangir-cv-split) ('func.func' operation: @flash_attention) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
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
    %4 = memref_ext.alloc_workspace() : memref<2x64x256xf32>
    annotation.mark %4 {hivm.multi_buffer = 4 : i32} : memref<2x64x256xf32>
    %5 = memref_ext.alloc_workspace() : memref<2x64x256xf16>
    annotation.mark %5 {hivm.multi_buffer = 4 : i32} : memref<2x64x256xf16>
    %6 = memref_ext.alloc_workspace() : memref<2x64x128xf32>
    annotation.mark %6 {hivm.multi_buffer = 4 : i32} : memref<2x64x128xf32>
    %7 = arith.subi %c151_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
      %alloc_5 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>>
      %alloc_6 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>>
      %alloc_7 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>>
      %alloc_8 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>>
      %alloc_9 = memref.alloc() : memref<32x128xf16, strided<[128, 1]>>
      %9 = arith.muli %arg13, %c1536_i32 : i32
      %10 = arith.muli %1, %c64_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %subview = memref.subview %reinterpret_cast[%12, 0] [64, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1]>>
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<32x1xf32, strided<[1, 1]>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_8 : memref<32x256xf32, strided<[256, 1]>>)
      scf.for %arg14 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        %alloc_12 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>>
        annotation.mark %alloc_12 {hivm.multi_buffer = 4 : i32} : memref<32x1xf32, strided<[1, 1]>>
        %21 = arith.muli %arg14, %c256_i32 : i32
        %22 = arith.index_cast %21 : i32 to index
        %23 = arith.muli %3, %c32_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        scope.scope : () -> () {
          %alloc_13 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>>
          %alloc_14 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>>
          %subview_15 = memref.subview %reinterpret_cast_3[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_15, %alloc_13 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>>
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_13, %true, %c64, %c128, %c256 : memref<64x128xf16, strided<[128, 1]>>, memref<256x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_14 : memref<64x256xf32, strided<[256, 1]>>)
          %subview_16 = memref.subview %4[0, 0, 0] [1, 64, 256] [1, 1, 1] : memref<2x64x256xf32> to memref<64x256xf32, strided<[256, 1]>>
          memref.copy %alloc_14, %subview_16 : memref<64x256xf32, strided<[256, 1]>> to memref<64x256xf32, strided<[256, 1]>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          %alloc_13 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>>
          %alloc_14 = memref.alloc() : memref<32x256xf16, strided<[256, 1]>>
          %alloc_15 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>>
          %alloc_16 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>>
          %alloc_17 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>>
          %alloc_18 = memref.alloc() : memref<32x256xf32, strided<[256, 1]>>
          %alloc_19 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>>
          %subview_20 = memref.subview %4[0, %24, 0] [1, 32, 256] [1, 1, 1] : memref<2x64x256xf32> to memref<32x256xf32, strided<[256, 1], offset: ?>>
          memref.copy %subview_20, %alloc_13 : memref<32x256xf32, strided<[256, 1], offset: ?>> to memref<32x256xf32, strided<[256, 1]>>
          hivm.hir.vmul ins(%alloc_13, %alloc_8 : memref<32x256xf32, strided<[256, 1]>>, memref<32x256xf32, strided<[256, 1]>>) outs(%alloc_13 : memref<32x256xf32, strided<[256, 1]>>)
          hivm.hir.vreduce <max> ins(%alloc_13 : memref<32x256xf32, strided<[256, 1]>>) outs(%alloc_15 : memref<32x1xf32, strided<[1, 1]>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc_5, %alloc_15 : memref<32x1xf32, strided<[1, 1]>>, memref<32x1xf32, strided<[1, 1]>>) outs(%alloc_17 : memref<32x1xf32, strided<[1, 1]>>)
          hivm.hir.vsub ins(%alloc_5, %alloc_17 : memref<32x1xf32, strided<[1, 1]>>, memref<32x1xf32, strided<[1, 1]>>) outs(%alloc_19 : memref<32x1xf32, strided<[1, 1]>>)
          hivm.hir.vexp ins(%alloc_19 : memref<32x1xf32, strided<[1, 1]>>) outs(%alloc_12 : memref<32x1xf32, strided<[1, 1]>>)
          hivm.hir.vsub ins(%alloc_13, %alloc_17 : memref<32x256xf32, strided<[256, 1]>>, memref<32x1xf32, strided<[1, 1]>>) outs(%alloc_18 : memref<32x256xf32, strided<[256, 1]>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_18 : memref<32x256xf32, strided<[256, 1]>>) outs(%alloc_13 : memref<32x256xf32, strided<[256, 1]>>)
          hivm.hir.vreduce <sum> ins(%alloc_13 : memref<32x256xf32, strided<[256, 1]>>) outs(%alloc_16 : memref<32x1xf32, strided<[1, 1]>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_6, %alloc_12 : memref<32x1xf32, strided<[1, 1]>>, memref<32x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>>)
          hivm.hir.vadd ins(%alloc_6, %alloc_16 : memref<32x1xf32, strided<[1, 1]>>, memref<32x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<32x1xf32, strided<[1, 1]>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_19 : memref<32x1xf32, strided<[1, 1]>>)
          hivm.hir.vadd ins(%alloc_19, %alloc_17 : memref<32x1xf32, strided<[1, 1]>>, memref<32x1xf32, strided<[1, 1]>>) outs(%alloc_5 : memref<32x1xf32, strided<[1, 1]>>)
          hivm.hir.vcast ins(%alloc_13 : memref<32x256xf32, strided<[256, 1]>>) outs(%alloc_14 : memref<32x256xf16, strided<[256, 1]>>)
          %subview_21 = memref.subview %5[0, %24, 0] [1, 32, 256] [1, 1, 1] : memref<2x64x256xf16> to memref<32x256xf16, strided<[256, 1], offset: ?>>
          memref.copy %alloc_14, %subview_21 : memref<32x256xf16, strided<[256, 1]>> to memref<32x256xf16, strided<[256, 1], offset: ?>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        scope.scope : () -> () {
          %alloc_13 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>>
          %alloc_14 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>>
          %alloc_15 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
          %subview_16 = memref.subview %5[0, 0, 0] [1, 64, 256] [1, 1, 1] : memref<2x64x256xf16> to memref<64x256xf16, strided<[256, 1]>>
          memref.copy %subview_16, %alloc_14 : memref<64x256xf16, strided<[256, 1]>> to memref<64x256xf16, strided<[256, 1]>>
          %subview_17 = memref.subview %reinterpret_cast_2[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_17, %alloc_13 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>>
          hivm.hir.mmadL1 ins(%alloc_14, %alloc_13, %true, %c64, %c256, %c128 : memref<64x256xf16, strided<[256, 1]>>, memref<256x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_15 : memref<64x128xf32, strided<[128, 1]>>)
          %subview_18 = memref.subview %6[0, 0, 0] [1, 64, 128] [1, 1, 1] : memref<2x64x128xf32> to memref<64x128xf32, strided<[128, 1]>>
          memref.copy %alloc_15, %subview_18 : memref<64x128xf32, strided<[128, 1]>> to memref<64x128xf32, strided<[128, 1]>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          %alloc_13 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>>
          %subview_14 = memref.subview %6[0, %24, 0] [1, 32, 128] [1, 1, 1] : memref<2x64x128xf32> to memref<32x128xf32, strided<[128, 1], offset: ?>>
          memref.copy %subview_14, %alloc_13 : memref<32x128xf32, strided<[128, 1], offset: ?>> to memref<32x128xf32, strided<[128, 1]>>
          hivm.hir.vmul ins(%alloc_7, %alloc_12 : memref<32x128xf32, strided<[128, 1]>>, memref<32x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_7, %alloc_13 : memref<32x128xf32, strided<[128, 1]>>, memref<32x128xf32, strided<[128, 1]>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>>)
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 4 : i32}
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<32x128xf32, strided<[128, 1]>>, memref<32x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<32x128xf32, strided<[128, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<32x128xf32, strided<[128, 1]>>) outs(%alloc_9 : memref<32x128xf16, strided<[128, 1]>>)
      %13 = arith.muli %3, %c32_i32 : i32
      %14 = arith.subi %c8192_i32, %13 : i32
      %15 = arith.subi %14, %10 : i32
      %16 = arith.subi %15, %9 : i32
      %17 = arith.minsi %16, %c32_i32 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview_10 = memref.subview %alloc_9[0, 0] [%18, 128] [1, 1] : memref<32x128xf16, strided<[128, 1]>> to memref<?x128xf16, strided<[128, 1]>>
      %19 = arith.addi %11, %13 : i32
      %20 = arith.index_cast %19 : i32 to index
      %subview_11 = memref.subview %reinterpret_cast_4[%20, 0] [%18, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_10, %subview_11 : memref<?x128xf16, strided<[128, 1]>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}