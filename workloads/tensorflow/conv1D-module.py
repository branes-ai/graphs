import tensorflow as tf
import numpy as np

# run:
#
# $ python conv1D-module.py
# $ iree-import-tf --tf-import-type=savedmodel_v1 --tf-savedmodel-exported-names=entry saved/ -o convtest.mlirbc
# $ iree-compile --iree-hal-target-backends=llvm-cpu conf.mlirbc --iree-llvmcpu-target-cpu=host  -o conv.vmfb
# $ <path-to-iree-build-tools>/iree-run-module --device=local-task --module=conv.vmfb --function=entry --input="1x10x1xf32=[1 2 3 4 5 6 7 8 9 10]"
#
# produces:
#
# EXEC @entry
# result[0]: hal.buffer_view
# 1x10x1xf32=[[-2][-2][-2][-2][-2][-2][-2][-2][-2][9]]

class ConvTestModule(tf.Module):
    def __init__(self, filter):
        self.filter_tf = tf.constant(filter)
    @tf.function
    def __call__(self, signal):
        return tf.nn.conv1d(signal, self.filter_tf, stride=1, padding='SAME')

# Define a simple filter (kernel) for convolution
filter = np.array([1, 0, -1], dtype=np.float32).reshape(-1, 1, 1)  # shape: (filter_length, in_channels, out_channels)

mod = ConvTestModule(filter)

# saved module
# create signature for entry point
call = mod.__call__.get_concrete_function(np.zeros((1, 10, 1), dtype=np.float32))
signatures = {'entry': call}
tf.saved_model.save(mod, 'saved/', signatures=signatures)
