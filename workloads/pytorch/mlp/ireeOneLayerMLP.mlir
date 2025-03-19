#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
module {
  module {
    func.func @main(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[3],f32>, %arg2: !torch.vtensor<[10,5],f32>) -> (!torch.vtensor<[10,3],f32>, !torch.vtensor<[10,5],f32>, !torch.vtensor<[10,3],f32>) {
      %int0 = torch.constant.int 0
      %int1 = torch.constant.int 1
      %0 = torch.aten.transpose.int %arg0, %int0, %int1 : !torch.vtensor<[3,5],f32>, !torch.int, !torch.int -> !torch.vtensor<[5,3],f32>
      %1 = torch.aten.mm %arg2, %0 : !torch.vtensor<[10,5],f32>, !torch.vtensor<[5,3],f32> -> !torch.vtensor<[10,3],f32>
      %int1_0 = torch.constant.int 1
      %2 = torch.aten.mul.Scalar %1, %int1_0 : !torch.vtensor<[10,3],f32>, !torch.int -> !torch.vtensor<[10,3],f32>
      %int1_1 = torch.constant.int 1
      %3 = torch.aten.mul.Scalar %arg1, %int1_1 : !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[3],f32>
      %int1_2 = torch.constant.int 1
      %4 = torch.aten.add.Tensor %2, %3, %int1_2 : !torch.vtensor<[10,3],f32>, !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[10,3],f32>
      %int1_3 = torch.constant.int 1
      %false = torch.constant.bool false
      %5 = torch.aten._softmax %4, %int1_3, %false : !torch.vtensor<[10,3],f32>, !torch.int, !torch.bool -> !torch.vtensor<[10,3],f32>
      return %5, %arg2, %5 : !torch.vtensor<[10,3],f32>, !torch.vtensor<[10,5],f32>, !torch.vtensor<[10,3],f32>
    }
  }
  module {
    util.func public @main$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
      %cst = arith.constant 0.000000e+00 : f32
      %c0_i64 = arith.constant 0 : i64
      %cst_0 = arith.constant 0xFF800000 : f32
      %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<3x5xf32>
      %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<3xf32>
      %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<10x5xf32>
      %3 = tensor.empty() : tensor<5x3xf32>
      %transposed = linalg.transpose ins(%0 : tensor<3x5xf32>) outs(%3 : tensor<5x3xf32>) permutation = [1, 0] 
      %4 = tensor.empty() : tensor<10x3xf32>
      %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<10x3xf32>) -> tensor<10x3xf32>
      %6 = linalg.matmul ins(%2, %transposed : tensor<10x5xf32>, tensor<5x3xf32>) outs(%5 : tensor<10x3xf32>) -> tensor<10x3xf32>
      %7 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %1 : tensor<10x3xf32>, tensor<3xf32>) outs(%4 : tensor<10x3xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %23 = arith.addf %in, %in_1 : f32
        linalg.yield %23 : f32
      } -> tensor<10x3xf32>
      %8 = tensor.empty() : tensor<10xi64>
      %9 = linalg.fill ins(%c0_i64 : i64) outs(%8 : tensor<10xi64>) -> tensor<10xi64>
      %10 = tensor.empty() : tensor<10xf32>
      %11 = linalg.fill ins(%cst_0 : f32) outs(%10 : tensor<10xf32>) -> tensor<10xf32>
      %12:2 = linalg.generic {indexing_maps = [#map, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%7 : tensor<10x3xf32>) outs(%11, %9 : tensor<10xf32>, tensor<10xi64>) {
      ^bb0(%in: f32, %out: f32, %out_1: i64):
        %23 = linalg.index 1 : index
        %24 = arith.index_cast %23 : index to i64
        %25 = arith.maximumf %in, %out : f32
        %26 = arith.cmpf ogt, %in, %out : f32
        %27 = arith.select %26, %24, %out_1 : i64
        linalg.yield %25, %27 : f32, i64
      } -> (tensor<10xf32>, tensor<10xi64>)
      %expanded = tensor.expand_shape %12#0 [[0, 1]] output_shape [10, 1] : tensor<10xf32> into tensor<10x1xf32>
      %13 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %expanded : tensor<10x3xf32>, tensor<10x1xf32>) outs(%4 : tensor<10x3xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %23 = arith.subf %in, %in_1 : f32
        linalg.yield %23 : f32
      } -> tensor<10x3xf32>
      %14 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<10x3xf32>) outs(%4 : tensor<10x3xf32>) {
      ^bb0(%in: f32, %out: f32):
        %23 = math.exp %in : f32
        linalg.yield %23 : f32
      } -> tensor<10x3xf32>
      %15 = tensor.empty() : tensor<10x1xf32>
      %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<10x1xf32>) -> tensor<10x1xf32>
      %17 = linalg.generic {indexing_maps = [#map, #map3], iterator_types = ["parallel", "reduction"]} ins(%14 : tensor<10x3xf32>) outs(%16 : tensor<10x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %23 = arith.addf %in, %out : f32
        linalg.yield %23 : f32
      } -> tensor<10x1xf32>
      %18 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%14, %17 : tensor<10x3xf32>, tensor<10x1xf32>) outs(%4 : tensor<10x3xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %23 = arith.divf %in, %in_1 : f32
        linalg.yield %23 : f32
      } -> tensor<10x3xf32>
      %19:2 = hal.tensor.barrier join(%18, %2 : tensor<10x3xf32>, tensor<10x5xf32>) => %arg4 : !hal.fence
      %20 = hal.tensor.export %19#0 : tensor<10x3xf32> -> !hal.buffer_view
      %21 = hal.tensor.export %19#1 : tensor<10x5xf32> -> !hal.buffer_view
      %22 = hal.tensor.export %19#0 : tensor<10x3xf32> -> !hal.buffer_view
      util.return %20, %21, %22 : !hal.buffer_view, !hal.buffer_view, !hal.buffer_view
    }
    util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) attributes {iree.abi.stub} {
      %0 = util.null : !hal.fence
      %c-1_i32 = arith.constant -1 : i32
      %c0 = arith.constant 0 : index
      %device_0 = hal.devices.get %c0 : !hal.device
      %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
      %1:3 = util.call @main$async(%arg0, %arg1, %arg2, %0, %fence) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view)
      %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
      util.return %1#0, %1#1, %1#2 : !hal.buffer_view, !hal.buffer_view, !hal.buffer_view
    }
  }
}

