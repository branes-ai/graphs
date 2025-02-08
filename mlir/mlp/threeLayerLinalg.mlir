module {
  func.func @mlp(%arg0: tensor<128x784xf32>) -> tensor<128x10xf32> {
    // Constants for weights and biases
    %w1 = arith.constant dense<...> : tensor<784x512xf32>
    %b1 = arith.constant dense<...> : tensor<512xf32>
    %w2 = arith.constant dense<...> : tensor<512x256xf32>
    %b2 = arith.constant dense<...> : tensor<256xf32>
    %w3 = arith.constant dense<...> : tensor<256x10xf32>
    %b3 = arith.constant dense<...> : tensor<10xf32>

    // First layer: Dense + ReLU
    %1 = linalg.matmul ins(%arg0, %w1 : tensor<128x784xf32>, tensor<784x512xf32>)
                       outs(tensor<128x512xf32>)
    %2 = linalg.broadcast_in_dim ins(%b1 : tensor<512xf32>)
                                 outs(tensor<128x512xf32>)
                                 {broadcast_dimensions = [1]}
    %3 = linalg.add ins(%1, %2 : tensor<128x512xf32>, tensor<128x512xf32>)
                    outs(tensor<128x512xf32>)
    %4 = linalg.map { math.maximum_f }
         ins(%3 : tensor<128x512xf32>)
         outs(tensor<128x512xf32>)
         reducing_init(%cst : f32)

    // Second layer: Dense + ReLU
    %5 = linalg.matmul ins(%4, %w2 : tensor<128x512xf32>, tensor<512x256xf32>)
                       outs(tensor<128x256xf32>)
    %6 = linalg.broadcast_in_dim ins(%b2 : tensor<256xf32>)
                                 outs(tensor<128x256xf32>)
                                 {broadcast_dimensions = [1]}
    %7 = linalg.add ins(%5, %6 : tensor<128x256xf32>, tensor<128x256xf32>)
                    outs(tensor<128x256xf32>)
    %8 = linalg.map { math.maximum_f }
         ins(%7 : tensor<128x256xf32>)
         outs(tensor<128x256xf32>)
         reducing_init(%cst : f32)

    // Third layer: Dense
    %9 = linalg.matmul ins(%8, %w3 : tensor<128x256xf32>, tensor<256x10xf32>)
                       outs(tensor<128x10xf32>)
    %10 = linalg.broadcast_in_dim ins(%b3 : tensor<10xf32>)
                                  outs(tensor<128x10xf32>)
                                  {broadcast_dimensions = [1]}
    %11 = linalg.add ins(%9, %10 : tensor<128x10xf32>, tensor<128x10xf32>)
                     outs(tensor<128x10xf32>)

    // Softmax
    // First compute max for numerical stability
    %12 = linalg.reduce { arith.maximum }
          ins(%11 : tensor<128x10xf32>)
          outs(tensor<128x1xf32>)
          dimensions = [1]
    
    // Subtract max and exp
    %13 = linalg.broadcast_in_dim ins(%12 : tensor<128x1xf32>)
                                  outs(tensor<128x10xf32>)
                                  {broadcast_dimensions = [0]}
    %14 = linalg.sub ins(%11, %13 : tensor<128x10xf32>, tensor<128x10xf32>)
                     outs(tensor<128x10xf32>)
    %15 = linalg.map { math.exp }
          ins(%14 : tensor<128x10xf32>)
          outs(tensor<128x10xf32>)

    // Sum exp values
    %16 = linalg.reduce { arith.addf }
          ins(%15 : tensor<128x10xf32>)
          outs(tensor<128x1xf32>)
          dimensions = [1]

    // Divide by sum
    %17 = linalg.broadcast_in_dim ins(%16 : tensor<128x1xf32>)
                                  outs(tensor<128x10xf32>)
                                  {broadcast_dimensions = [0]}
    %18 = linalg.div ins(%15, %17 : tensor<128x10xf32>, tensor<128x10xf32>)
                     outs(tensor<128x10xf32>)

    return %18 : tensor<128x10xf32>
  }
}
