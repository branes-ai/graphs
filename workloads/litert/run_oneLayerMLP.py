import numpy as np
import iree.runtime as iree_rt

print("Starting script execution")

# Get the driver and create a device
print("Getting local-task driver")
driver = iree_rt.get_driver("local-task")  # Fetch the driver
print("Driver obtained:", driver)
print("Creating device")
device = driver.create_default_device()    # Create a device from the driver
print("Device created:", device)

# Create a VM instance and context
print("Creating VM instance")
instance = iree_rt.VmInstance()
print("VM instance created:", instance)
print("Creating VM context")
context = iree_rt.VmContext(instance=instance)
print("VM context created:", context)

# Create and register the HAL module
print("Creating HAL module")
hal_module = iree_rt.create_hal_module(instance, device)
print("HAL module created:", hal_module)
print("Registering HAL module")
context.register_modules([hal_module])
print("HAL module registered")

# Load and register the user module
print("Loading VM module")
with open("oneLayerMLP.vmfb", "rb") as f:
    vm_module = iree_rt.VmModule.from_flatbuffer(instance, f.read())
print("Module loaded:", vm_module)
print("Registering user module")
context.register_modules([vm_module])
print("User module registered")

# Prepare input data
print("Preparing input data")
input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
print("Input data prepared:", input_data)

# Get the main function from the module
print("Looking up main function")
main_func = vm_module.lookup_function("main")
print("Main function retrieved:", main_func is not None)

if main_func is None:
    print("Error: Could not retrieve main function")
else:
    # Execute the main function using context.invoke
    print("Executing main function via context.invoke")
    try:
        # Prepare input as a list for invoke (IREE expects a list of args)
        output = context.invoke(main_func, [input_data])
        print("Output type:", type(output))
        print("Output value:", output)

        # Convert output to numpy if needed
        if hasattr(output, 'to_host'):
            output_np = output.to_host()
            print("Output as numpy:", output_np)
        else:
            print("Output (direct):", output)
    except AttributeError:
        print("context.invoke not available, trying alternative invocation")
        # Fallback: Use context to call the function directly if invoke isn’t available
        output = context.call(main_func, [input_data])
        print("Output type:", type(output))
        print("Output value:", output)

        if hasattr(output, 'to_host'):
            output_np = output.to_host()
            print("Output as numpy:", output_np)
        else:
            print("Output (direct):", output)

print("Script execution completed")