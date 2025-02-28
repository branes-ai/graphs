module {
  func.func @main(%arg0: tensor<1x4xf32> {ml_program.identifier = "inputs"}) -> (tensor<1x2xf32> {ml_program.identifier = "Identity"}) {
    %0 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2xf32>}> : () -> tensor<2xf32>
    %1 = "tosa.const"() <{value = dense<[[[[-0.469961405, -0.881487846, 0.810493708, 0.179342747]]], [[[0.921667575, 0.855973243, -0.76720643, -0.537564039]]]]> : tensor<2x1x1x4xf32>}> : () -> tensor<2x1x1x4xf32>
    %2 = tosa.reshape %arg0 {new_shape = array<i64: 1, 1, 1, 4>} : (tensor<1x4xf32>) -> tensor<1x1x1x4xf32>
    %3 = tosa.conv2d %2, %1, %0 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x4xf32>, tensor<2x1x1x4xf32>, tensor<2xf32>) -> tensor<1x1x1x2xf32>
    %4 = tosa.reshape %3 {new_shape = array<i64: 1, 2>} : (tensor<1x1x1x2xf32>) -> tensor<1x2xf32>
    return %4 : tensor<1x2xf32>
  }
}

