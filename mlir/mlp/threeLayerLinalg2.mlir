#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @mlp(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?x?xf32>, %arg4: tensor<?xf32>, %arg5: tensor<?x?xf32>, %arg6: tensor<?xf32>) -> tensor<?xf32> {
  %cst0 = constant 0.000000e+00 : f32
  %cst1 = constant 1.000000e+00 : f32

  // Layer 1: Fully Connected
  %matmul_1 = linalg.matmul mkl_blas_gemm_f32 ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%cst0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %add_1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = [affine_map<(d0) -> (d0)>]} ins(%matmul_1, %arg2 : tensor<?x?xf32>, tensor<?xf32>) outs(%matmul_1 : tensor<?x?xf32>) {
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %add = addf %in1, %in2 : f32
    linalg.yield %add : f32
  } : tensor<?x?xf32>

  // Activation Function (ReLU)
  %relu_1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = [affine_map<(d0, d1) -> (d0, d1)>]} ins(%add_1 : tensor<?x?xf32>) outs(%add_1 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = math.maxf %in, %cst0 : f32
    linalg.yield %max : f32
  } : tensor<?x?xf32>


  // Layer 2: Fully Connected
  %matmul_2 = linalg.matmul mkl_blas_gemm_f32 ins(%relu_1, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%cst0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %add_2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = [affine_map<(d0) -> (d0)>]} ins(%matmul_2, %arg4 : tensor<?x?xf32>, tensor<?xf32>) outs(%matmul_2 : tensor<?x?xf32>) {
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %add = addf %in1, %in2 : f32
    linalg.yield %add : f32
  } : tensor<?x?xf32>

  // Activation Function (ReLU)
  %relu_2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = [affine_map<(d0, d1) -> (d0, d1)>]} ins(%add_2 : tensor<?x?xf32>) outs(%add_2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = math.maxf %in, %cst0 : f32
    linalg.yield %max : f32
  } : tensor<?x?xf32>

  // Layer 3: Fully Connected
  %matmul_3 = linalg.matmul mkl_blas_gemm_f32 ins(%relu_2, %arg5 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%cst0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %add_3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = [affine_map<(d0) -> (d0)>]} ins(%matmul_3, %arg6 : tensor<?x?xf32>, tensor<?xf32>) outs(%matmul_3 : tensor<?x?xf32>) {
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %add = addf %in1, %in2 : f32
    linalg.yield %add : f32
  } : tensor<?x?xf32>

  // Softmax
  %softmax = linalg.generic {indexing_maps = [#map, #map], iterator_types = [affine_map<(d0, d1) -> (d0, d1)>]} ins(%add_3 : tensor<?x?xf32>) outs(%add_3 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    // This is a simplified softmax.  A proper softmax requires
    // finding the maximum value in each row and subtracting it before exp.
    // This example omits that for simplicity and focuses on the Linalg structure.
    %exp = math.expf %in : f32
    linalg.yield %exp : f32
  } : tensor<?x?xf32>

  return %softmax : tensor<?x?xf32>
}