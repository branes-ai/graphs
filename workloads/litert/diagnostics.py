import iree.runtime as iree_rt

print("Starting script")
print("IREE runtime version:", iree_rt.version())

try:
    print("Creating VM instance")
    instance = iree_rt.VmInstance()
    print("VM instance created:", instance)

    print("Attempting to create HAL module with local-task")
    hal_module = iree_rt.create_hal_module(instance)
    print("HAL module created:", hal_module)

except Exception as e:
    print("Exception caught:", str(e))
    import traceback
    traceback.print_exc()

print("Script completed")