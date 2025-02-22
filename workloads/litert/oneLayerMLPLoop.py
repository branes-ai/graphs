import iree.runtime as iree_rt
import numpy as np

# ANSI escape codes for colors
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"  # Resets color to default

# Paths and names
vmfb_path = "oneLayerMLP.vmfb"
function_name = "main"
input_file = "testdata.txt"
driver_name = "local-task"  # or "cuda" if GPU is intended

# Create config and context
config = iree_rt.Config(driver_name)
context = iree_rt.SystemContext(config=config)

# Load VMFB
with open(vmfb_path, "rb") as f:
    vmfb_data = f.read()
vm_module = iree_rt.VmModule.from_flatbuffer(context.instance, vmfb_data)
context.add_vm_module(vm_module)

# Debug: Check available functions
#print("Loaded modules:", context.modules)
#print(f"Functions in 'module': {dir(context.modules['module'])}")

# Parse input function
def parse_input(line):
    try:
        line = line.strip().strip('"')
        values = line.split("=")[1].strip("[]").split()
        return np.array(values, dtype=np.float32).reshape(1, 4)
    except Exception as e:
        print(f"Error parsing input: {line}, Error: {e}")
        return None

# Process inputs
with open(input_file, "r") as file:
    for line in file:
        input_data = parse_input(line)
        if input_data is None:
            continue
        print(f"Parsed input shape: {input_data.shape}, data: {input_data}")
        try:
            result = context.modules['module'][function_name](input_data)
            result_np = np.asarray(result)  # Convert to NumPy array
            print(f"Input: {line.strip()}\nOutput: {result_np}\n{'-' * 40}")
            
            # Check output dimensions with colored, "larger" output
            if result_np[0, 0] < -0.2:  # First dimension (index 0)
                print(f"{RED}********************\n* FOUND SUSPICIOUS *\n*     OBJECTS      *\n********************{RESET}")
            if result_np[0, 1] < 0.2:   # Second dimension (index 1)
                print(f"{GREEN}**********************\n* ALL CLEAR FOR NOW  *\n*   GOOD TO GO       *\n**********************{RESET}")
            print("-" * 40)  # Separator after messages
            
        except Exception as e:
            print(f"Error executing: {line.strip()}, Error: {e}\n{'-' * 40}")