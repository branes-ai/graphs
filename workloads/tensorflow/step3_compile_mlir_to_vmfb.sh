iree-compile --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host mlir/iree_onelayer.mlir -o ./vmfb/iree_onelayer.vmfb

iree-compile --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host mlir/iree_twolayer.mlir -o ./vmfb/iree_twolayer.vmfb --module=forward

iree-compile --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host mlir/iree_threelayer.mlir -o ./vmfb/iree_threelayer.vmfb --module=forward
