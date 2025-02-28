module {
  func.func @main(%arg0: tensor<1x4xf32> {ml_program.identifier = "inputs"}) -> (tensor<1x2xf32> {ml_program.identifier = "Identity"}) {
    %0 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2xf32>}> : () -> tensor<2xf32>
    %1 = "tosa.const"() <{value = dense<[[[[5.871910e-01, 0.0841061472, 0.0830870867, -0.19283694, 0.126194894, -0.40210861, -0.668491303, 0.733679533]]], [[[0.113784492, 0.342509627, 0.201501906, 0.104041457, -7.561900e-01, -0.0533513427, -0.513203859, -0.67047435]]]]> : tensor<2x1x1x8xf32>}> : () -> tensor<2x1x1x8xf32>
    %2 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<8xf32>}> : () -> tensor<8xf32>
    %3 = "tosa.const"() <{value = dense<[[[[-0.134873271, -0.648677588, 0.648867547, 0.388438284]]], [[[-0.643203855, -0.288629383, -0.328942984, -0.208592176]]], [[[0.131848991, 0.0770228505, -0.571867466, 0.0129542351]]], [[[-0.564357579, 0.404789269, -0.640489101, 0.1039235]]], [[[0.0434161425, -0.39589563, 0.586398065, -0.00939017534]]], [[[-0.146747589, -0.326729953, -0.486392498, -0.0906570553]]], [[[-3.440400e-01, -0.254968196, -0.350782484, -0.413751215]]], [[[-0.669178307, -0.556354225, 0.0384556055, 0.302319825]]]]> : tensor<8x1x1x4xf32>}> : () -> tensor<8x1x1x4xf32>
    %4 = tosa.reshape %arg0 {new_shape = array<i64: 1, 1, 1, 4>} : (tensor<1x4xf32>) -> tensor<1x1x1x4xf32>
    %5 = tosa.conv2d %4, %3, %2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x4xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>) -> tensor<1x1x1x8xf32>
    %6 = tosa.reshape %5 {new_shape = array<i64: 1, 8>} : (tensor<1x1x1x8xf32>) -> tensor<1x8xf32>
    %7 = tosa.clamp %6 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x8xf32>) -> tensor<1x8xf32>
    %8 = tosa.reshape %7 {new_shape = array<i64: 1, 1, 1, 8>} : (tensor<1x8xf32>) -> tensor<1x1x1x8xf32>
    %9 = tosa.conv2d %8, %1, %0 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x8xf32>, tensor<2x1x1x8xf32>, tensor<2xf32>) -> tensor<1x1x1x2xf32>
    %10 = tosa.reshape %9 {new_shape = array<i64: 1, 2>} : (tensor<1x1x1x2xf32>) -> tensor<1x2xf32>
    return %10 : tensor<1x2xf32>
  }
}

