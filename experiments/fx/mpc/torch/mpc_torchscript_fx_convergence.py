#!/usr/bin/env python3
"""
Complete Model Predictive Control (MPC) implementation with TorchScript and FX graph analysis.
This example demonstrates a full closed-loop MPC controller that drives a chemical reactor
to setpoint over multiple time steps, then analyzes the computational graphs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.fx as fx
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend for plotting

# =============================================================================
# Chemical Reactor Process Model
# =============================================================================

class ChemicalReactor:
    """Chemical reactor model with realistic dynamics."""
    
    def __init__(self):
        # Process parameters
        self.dt = 0.1  # Sampling time (seconds)
        self.tau = 5.0  # Time constant (seconds)
        self.K = 2.0    # Process gain
        self.theta = 0.5  # Time delay (seconds)
        self.delay_steps = int(self.theta / self.dt)
        
        # State variables
        self.temperature = 25.0  # Current temperature (°C)
        self.heat_capacity = 1000.0  # Heat capacity (J/K)
        self.heat_loss_coeff = 50.0  # Heat loss coefficient
        
        # Input history for delay
        self.input_history = [0.0] * max(self.delay_steps, 1)
        
        # Disturbance
        self.ambient_temp = 20.0
        
    def step(self, heat_input: float, disturbance: float = 0.0) -> float:
        """Simulate one time step of the reactor."""
        # Get delayed input
        delayed_input = self.input_history[0] if self.input_history else 0.0
        
        # First-order dynamics with heat loss and disturbance
        heat_in = self.K * delayed_input
        heat_loss = self.heat_loss_coeff * (self.temperature - self.ambient_temp)
        
        # Temperature rate of change
        dT_dt = (heat_in - heat_loss + disturbance) / (self.heat_capacity / 100)
        
        # Update temperature
        self.temperature += dT_dt * self.dt
        
        # Update input history
        self.input_history.append(heat_input)
        if len(self.input_history) > max(self.delay_steps, 1):
            self.input_history.pop(0)
        
        return self.temperature
    
    def get_state(self) -> np.ndarray:
        """Get current state vector."""
        return np.array([self.temperature])
    
    def reset(self, initial_temp: float = 25.0):
        """Reset reactor to initial conditions."""
        self.temperature = initial_temp
        self.input_history = [0.0] * max(self.delay_steps, 1)

# =============================================================================
# Complete MPC Controller Implementation
# =============================================================================

class MPCController:
    """Complete Model Predictive Controller with optimization."""
    
    def __init__(self, prediction_horizon: int = 15, control_horizon: int = 5):
        self.N = prediction_horizon  # Prediction horizon
        self.M = control_horizon     # Control horizon
        self.dt = 0.1               # Sampling time
        
        # MPC tuning weights
        self.Q = 100.0  # State tracking weight
        self.R = 1.0    # Control effort weight
        self.S = 200.0  # Terminal state weight
        self.R_delta = 10.0  # Control rate weight
        
        # Physical constraints
        self.u_min = 0.0     # Minimum heat input (kW)
        self.u_max = 150.0   # Maximum heat input (kW)
        self.du_max = 20.0   # Maximum control rate (kW/step)
        
        # Process model for prediction (linearized around operating point)
        self.A = np.array([[0.95]])  # Discrete state transition
        self.B = np.array([[0.1]])   # Discrete control input
        self.C = np.array([[1.0]])   # Output matrix
        
        # State estimator
        self.x_est = np.array([25.0])  # Estimated state
        self.P = np.array([[1.0]])     # Estimation covariance
        
        # Control history
        self.u_prev = 0.0
        self.control_history = []
        self.error_history = []
        
    def predict_states(self, x0: np.ndarray, u_sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict future states and outputs given control sequence."""
        x = x0.copy().reshape(-1, 1)
        states = []
        outputs = []
        
        for k in range(self.N):
            # Get control input (hold last value if sequence is shorter)
            if k < len(u_sequence):
                u = u_sequence[k]
            else:
                u = u_sequence[-1] if len(u_sequence) > 0 else 0.0
            
            # State prediction
            x = self.A @ x + self.B * u
            y = self.C @ x
            
            states.append(x.flatten()[0])
            outputs.append(y.flatten()[0])
        
        return np.array(states), np.array(outputs)
    
    def cost_function(self, u_sequence: np.ndarray, x0: np.ndarray, 
                     setpoint: float, u_prev: float) -> float:
        """MPC quadratic cost function."""
        u_seq = u_sequence.reshape(-1)
        
        # Extend control sequence for full prediction horizon
        u_extended = np.zeros(self.N)
        for k in range(self.N):
            if k < len(u_seq):
                u_extended[k] = u_seq[k]
            else:
                u_extended[k] = u_seq[-1] if len(u_seq) > 0 else u_prev
        
        # Predict future behavior
        states, outputs = self.predict_states(x0, u_extended)
        
        # Initialize cost
        J = 0.0
        
        # State tracking cost
        for k in range(self.N):
            error = outputs[k] - setpoint
            if k == self.N - 1:
                J += self.S * error**2  # Terminal cost
            else:
                J += self.Q * error**2
        
        # Control effort cost
        for k in range(len(u_seq)):
            J += self.R * u_seq[k]**2
        
        # Control rate cost
        for k in range(len(u_seq)):
            if k == 0:
                du = u_seq[k] - u_prev
            else:
                du = u_seq[k] - u_seq[k-1]
            J += self.R_delta * du**2
        
        return J
    
    def solve_mpc(self, current_state: np.ndarray, setpoint: float) -> Tuple[float, bool]:
        """Solve MPC optimization problem."""
        # Initial guess - smooth transition from previous control
        u_init = np.ones(self.M) * self.u_prev
        
        # Constraints
        bounds = [(self.u_min, self.u_max) for _ in range(self.M)]
        
        # Control rate constraints
        constraints = []
        for k in range(self.M):
            if k == 0:
                # First control move
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda u, k=k: self.du_max - abs(u[k] - self.u_prev)
                })
            else:
                # Subsequent control moves
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda u, k=k: self.du_max - abs(u[k] - u[k-1])
                })
        
        # Solve optimization
        try:
            result = minimize(
                self.cost_function,
                u_init,
                args=(current_state, setpoint, self.u_prev),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if result.success:
                optimal_u = result.x[0]
                self.u_prev = optimal_u
                return optimal_u, True
            else:
                return self.u_prev, False
                
        except Exception as e:
            print(f"MPC optimization failed: {e}")
            return self.u_prev, False
    
    def update_estimator(self, measurement: float):
        """Simple state estimator update."""
        # For this example, use direct measurement
        self.x_est = np.array([measurement])
    
    def control_step(self, measurement: float, setpoint: float) -> Tuple[float, Dict[str, float]]:
        """Execute one complete MPC control step."""
        # Update state estimate
        self.update_estimator(measurement)
        
        # Solve MPC
        control_action, success = self.solve_mpc(self.x_est, setpoint)
        
        # Track error and control
        error = setpoint - measurement
        self.error_history.append(error)
        self.control_history.append(control_action)
        
        # Limit history length
        if len(self.error_history) > 100:
            self.error_history.pop(0)
        if len(self.control_history) > 100:
            self.control_history.pop(0)
        
        # Return control action and diagnostics
        diagnostics = {
            'error': error,
            'control': control_action,
            'success': success,
            'estimated_state': self.x_est[0]
        }
        
        return control_action, diagnostics

# =============================================================================
# Complete Closed-Loop MPC Simulation
# =============================================================================

def run_mpc_simulation(duration: float = 30.0, setpoint_changes: Optional[List[Tuple[float, float]]] = None) -> Dict[str, List[float]]:
    """Run complete closed-loop MPC simulation."""
    
    # Initialize system
    reactor = ChemicalReactor()
    mpc = MPCController()
    
    # Simulation parameters
    dt = 0.1
    num_steps = int(duration / dt)
    
    # Setpoint profile
    if setpoint_changes is None:
        setpoint_changes = [(0.0, 50.0), (10.0, 75.0), (20.0, 40.0)]
    
    # Storage for results
    results = {
        'time': [],
        'temperature': [],
        'setpoint': [],
        'control': [],
        'error': []
    }
    
    # Reset system
    reactor.reset(25.0)
    
    print("Running MPC closed-loop simulation...")
    print(f"Duration: {duration}s, Steps: {num_steps}")
    
    # Main simulation loop
    for step in range(num_steps):
        current_time = step * dt
        
        # Determine current setpoint
        current_setpoint = setpoint_changes[0][1]  # Default
        for change_time, setpoint_value in setpoint_changes:
            if current_time >= change_time:
                current_setpoint = setpoint_value
        
        # Get current measurement
        current_temp = reactor.temperature
        
        # MPC control step
        control_action, diagnostics = mpc.control_step(current_temp, current_setpoint)
        
        # Apply control to reactor with some disturbance
        disturbance = 0.5 * np.sin(0.1 * current_time)  # Small periodic disturbance
        new_temp = reactor.step(control_action, disturbance)
        
        # Store results
        results['time'].append(current_time)
        results['temperature'].append(current_temp)
        results['setpoint'].append(current_setpoint)
        results['control'].append(control_action)
        results['error'].append(diagnostics['error'])
        
        # Print progress
        if step % 50 == 0:
            print(f"t={current_time:.1f}s: T={current_temp:.1f}°C, SP={current_setpoint:.1f}°C, "
                  f"u={control_action:.1f}kW, e={diagnostics['error']:.2f}")
    
    return results

# =============================================================================
# Neural Network Enhanced MPC (for PyTorch graph analysis)
# =============================================================================

class NeuralMPCController(nn.Module):
    """Neural network enhanced MPC controller for graph analysis."""
    
    def __init__(self):
        super().__init__()
        
        # Neural network for model correction/enhancement
        self.state_encoder = nn.Sequential(
            nn.Linear(4, 32),  # [temp, setpoint, error, control_prev]
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        self.control_predictor = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output 0-1, will be scaled
        )
        
        # MPC parameters as buffers
        self.register_buffer('control_scale', torch.tensor(150.0))
        self.register_buffer('temp_ref', torch.tensor(25.0))
        
    def forward(self, current_temp: torch.Tensor, setpoint: torch.Tensor, 
                error_integral: torch.Tensor, prev_control: torch.Tensor) -> torch.Tensor:
        """Enhanced MPC forward pass with neural network correction."""
        
        # Normalize inputs
        temp_norm = (current_temp - self.temp_ref) / 50.0
        setpoint_norm = (setpoint - self.temp_ref) / 50.0
        error = setpoint_norm - temp_norm
        control_norm = prev_control / self.control_scale
        
        # Create input vector
        state_input = torch.stack([temp_norm, setpoint_norm, error, control_norm])
        
        # Neural network processing
        encoded_state = self.state_encoder(state_input)
        control_output = self.control_predictor(encoded_state)
        
        # Scale output and add classical control component
        neural_control = control_output.squeeze() * self.control_scale
        
        # Classical PID component
        proportional = 2.0 * error * 50.0  # Scale back error
        integral = 0.1 * error_integral
        
        # Combine neural and classical control
        total_control = neural_control + proportional + integral
        
        # Apply constraints
        total_control = torch.clamp(total_control, 0.0, self.control_scale)
        
        return total_control

class ClosedLoopMPCSimulator(nn.Module):
    """Complete closed-loop MPC simulation for graph analysis."""
    
    def __init__(self, num_steps: int = 50):
        super().__init__()
        self.controller = NeuralMPCController()
        self.num_steps = num_steps
        
        # Process model parameters
        self.register_buffer('dt', torch.tensor(0.1))
        self.register_buffer('tau', torch.tensor(5.0))
        self.register_buffer('K', torch.tensor(2.0))
        self.register_buffer('ambient_temp', torch.tensor(20.0))
        
    def plant_step(self, temperature: torch.Tensor, heat_input: torch.Tensor) -> torch.Tensor:
        """Simplified plant model for one time step."""
        heat_in = self.K * heat_input
        heat_loss = 10.0 * (temperature - self.ambient_temp)
        dT_dt = (heat_in - heat_loss) / 100.0
        new_temperature = temperature + dT_dt * self.dt
        return new_temperature
    
    def forward(self, initial_temp: torch.Tensor, setpoint: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run complete closed-loop simulation."""
        
        # Initialize states
        temperature = initial_temp
        control = torch.tensor(0.0)
        error_integral = torch.tensor(0.0)
        
        # Storage for trajectory
        temp_trajectory = [temperature]
        control_trajectory = [control]
        error_trajectory = []
        
        # Simulation loop
        for step in range(self.num_steps):
            # Calculate error
            error = setpoint - temperature
            error_trajectory.append(error)
            
            # Update integral
            error_integral += error * self.dt
            
            # MPC control step
            control = self.controller(temperature, setpoint, error_integral, control)
            
            # Apply control to plant
            temperature = self.plant_step(temperature, control)
            
            # Store results
            temp_trajectory.append(temperature)
            control_trajectory.append(control)
        
        return (torch.stack(temp_trajectory), 
                torch.stack(control_trajectory), 
                torch.stack(error_trajectory))

# =============================================================================
# Graph Analysis Functions
# =============================================================================

def analyze_torchscript_graph(model: torch.jit.ScriptModule) -> Dict[str, Any]:
    """Comprehensive TorchScript graph analysis."""
    print("=== TorchScript Graph Analysis ===")
    
    graph = model.graph
    
    # Extract detailed information
    info = {
        'nodes': [],
        'inputs': [],
        'outputs': [],
        'parameters': [],
        'node_count': 0,
        'operation_types': {},
        'graph_str': str(graph)
    }
    
    # Analyze inputs
    for i, input_node in enumerate(graph.inputs()):
        info['inputs'].append({
            'index': i,
            'name': input_node.debugName(),
            'type': str(input_node.type())
        })
    
    # Analyze outputs
    for i, output_node in enumerate(graph.outputs()):
        info['outputs'].append({
            'index': i,
            'name': output_node.debugName(),
            'type': str(output_node.type())
        })
    
    # Analyze nodes and count operation types
    for i, node in enumerate(graph.nodes()):
        operation = node.kind()
        
        # Count operation types
        if operation in info['operation_types']:
            info['operation_types'][operation] += 1
        else:
            info['operation_types'][operation] = 1
        
        node_info = {
            'index': i,
            'kind': operation,
            'inputs': [inp.debugName() for inp in node.inputs()],
            'outputs': [out.debugName() for out in node.outputs()],
            'attributes': {}
        }
        
        # Extract attributes
        for attr_name in node.attributeNames():
            try:
                attr_value = node[attr_name]
                node_info['attributes'][attr_name] = str(attr_value)
            except:
                node_info['attributes'][attr_name] = "Unable to retrieve"
        
        info['nodes'].append(node_info)
        info['node_count'] += 1
    
    # Print summary
    print(f"Graph Summary:")
    print(f"  Nodes: {info['node_count']}")
    print(f"  Inputs: {len(info['inputs'])}")
    print(f"  Outputs: {len(info['outputs'])}")
    print(f"  Operation types: {len(info['operation_types'])}")
    
    print(f"\nOperation Distribution:")
    for op_type, count in sorted(info['operation_types'].items()):
        print(f"  {op_type}: {count}")
    
    print(f"\nFirst 10 nodes:")
    for i, node in enumerate(info['nodes'][:10]):
        print(f"  {i}: {node['kind']} -> {node['outputs']}")
    
    return info

def analyze_fx_graph(model: fx.GraphModule) -> Dict[str, Any]:
    """Comprehensive FX graph analysis."""
    print("\n=== FX Graph Analysis ===")
    
    info = {
        'nodes': [],
        'node_count': 0,
        'operation_types': {},
        'graph_code': str(model.code),
        'module_hierarchy': {}
    }
    
    # Analyze each node
    for i, node in enumerate(model.graph.nodes):
        operation = node.op
        target = str(node.target)
        
        # Count operation types
        if operation in info['operation_types']:
            info['operation_types'][operation] += 1
        else:
            info['operation_types'][operation] = 1
        
        node_info = {
            'index': i,
            'name': node.name,
            'op': operation,
            'target': target,
            'args': [str(arg) for arg in node.args],
            'kwargs': {k: str(v) for k, v in node.kwargs.items()},
            'users': [user.name for user in node.users.keys()],
            'meta': node.meta if hasattr(node, 'meta') else {}
        }
        
        info['nodes'].append(node_info)
        info['node_count'] += 1
    
    # Print summary
    print(f"Graph Summary:")
    print(f"  Nodes: {info['node_count']}")
    print(f"  Operation types: {len(info['operation_types'])}")
    
    print(f"\nOperation Distribution:")
    for op_type, count in sorted(info['operation_types'].items()):
        print(f"  {op_type}: {count}")
    
    print(f"\nFirst 10 nodes:")
    for i, node in enumerate(info['nodes'][:10]):
        print(f"  {i}: {node['name']} ({node['op']}) -> {node['target']}")
    
    return info

def walk_fx_graph_execution(model: fx.GraphModule, *inputs) -> Dict[str, Any]:
    """Walk through FX graph execution with detailed tracking."""
    print("\n=== Walking FX Graph Execution ===")
    
    intermediate_values = {}
    execution_order = []
    
    class DetailedTracker(fx.Interpreter):
        def run_node(self, n: fx.Node) -> Any:
            result = super().run_node(n)
            
            # Store intermediate value
            intermediate_values[n.name] = {
                'value': result,
                'type': type(result).__name__,
                'shape': getattr(result, 'shape', None),
                'op': n.op,
                'target': str(n.target)
            }
            
            execution_order.append(n.name)
            
            # Print execution info
            shape_info = f"shape={result.shape}" if hasattr(result, 'shape') else f"type={type(result).__name__}"
            print(f"  {len(execution_order):2d}. {n.name} ({n.op}): {shape_info}")
            
            return result
    
    # Execute with tracking
    tracker = DetailedTracker(model)
    final_result = tracker.run(*inputs)
    
    print(f"\nExecution completed: {len(execution_order)} operations")
    print(f"Final result shapes: {[getattr(r, 'shape', 'scalar') for r in final_result]}")
    
    return {
        'intermediate_values': intermediate_values,
        'execution_order': execution_order,
        'final_result': final_result
    }

# =============================================================================
# Main Demonstration
# =============================================================================

def main():
    """Main demonstration of complete MPC with graph analysis."""
    
    print("Complete Model Predictive Control with Graph Analysis")
    print("=" * 60)
    
    # =============================================================================
    # 1. Run Classical MPC Simulation
    # =============================================================================
    
    print("\n1. Running Classical MPC Closed-Loop Simulation...")
    
    # Run simulation
    simulation_results = run_mpc_simulation(
        duration=20.0,
        setpoint_changes=[(0.0, 50.0), (8.0, 75.0), (15.0, 40.0)]
    )
    
    # Calculate performance metrics
    final_errors = simulation_results['error'][-50:]  # Last 50 steps
    mae = np.mean(np.abs(final_errors))
    rmse = np.sqrt(np.mean(np.array(final_errors)**2))
    
    print(f"\nClassical MPC Performance:")
    print(f"  Final MAE: {mae:.2f}°C")
    print(f"  Final RMSE: {rmse:.2f}°C")
    print(f"  Setpoint tracking: {'Good' if mae < 2.0 else 'Needs tuning'}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(simulation_results['time'], simulation_results['temperature'], label='Temperature (°C)')
    plt.plot(simulation_results['time'], simulation_results['setpoint'], label='Setpoint (°C)', linestyle='--')
    plt.plot(simulation_results['time'], simulation_results['control'], label='Control Action (kW)', linestyle=':')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.title('MPC Closed-Loop Simulation Results')
    plt.legend()
    plt.grid()
    plt.show()
    input("Press Enter to continue...")  # Pause for user to view plot
    print("Simulation complete. Press Enter to continue to Neural Enhanced MPC...")

    # =============================================================================
    # 2. Neural Enhanced MPC for Graph Analysis
    # =============================================================================
    
    print("\n2. Creating Neural Enhanced MPC for Graph Analysis...")
    
    # Create models
    enhanced_controller = NeuralMPCController()
    closed_loop_simulator = ClosedLoopMPCSimulator(num_steps=20)
    
    # Test inputs
    initial_temp = torch.tensor(25.0)
    setpoint = torch.tensor(50.0)
    error_integral = torch.tensor(0.0)
    prev_control = torch.tensor(0.0)
    
    # Test the enhanced controller
    with torch.no_grad():
        control_output = enhanced_controller(initial_temp, setpoint, error_integral, prev_control)
        print(f"Neural enhanced control: {control_output.item():.2f}")
        
        # Test full simulation
        temps, controls, errors = closed_loop_simulator(initial_temp, setpoint)
        print(f"Simulation: {temps.shape[0]} steps, final temp: {temps[-1].item():.2f}°C")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(temps.numpy(), label='Neural Sim Temperature (°C)')
    plt.xlabel('Step')
    plt.ylabel('Temperature (°C)')
    plt.title('Neural Enhanced MPC Simulation')
    plt.legend()
    plt.grid()
    plt.show()
    input("Press Enter to continue...")

    #exit(0)  # Exit early for demonstration purposes
    
    # =============================================================================
    # 3. TorchScript Analysis
    # =============================================================================
    
    print("\n3. TorchScript Graph Analysis...")
    
    try:
        # Trace the enhanced controller
        traced_controller = torch.jit.trace(
            enhanced_controller,
            (initial_temp, setpoint, error_integral, prev_control)
        )
        
        print("✓ Enhanced controller tracing successful")
        controller_info = analyze_torchscript_graph(traced_controller)
        
        # Trace the full simulation
        traced_simulator = torch.jit.trace(
            closed_loop_simulator,
            (initial_temp, setpoint)
        )
        
        print("\n✓ Full simulation tracing successful")
        simulator_info = analyze_torchscript_graph(traced_simulator)
        
    except Exception as e:
        print(f"TorchScript tracing failed: {e}")
        print("Attempting scripting instead...")
        
        try:
            scripted_controller = torch.jit.script(enhanced_controller)
            scripted_simulator = torch.jit.script(closed_loop_simulator)
            print("✓ Scripting successful")
            
            controller_info = analyze_torchscript_graph(scripted_controller)
            simulator_info = analyze_torchscript_graph(scripted_simulator)
            
        except Exception as e2:
            print(f"Scripting also failed: {e2}")
    
    # =============================================================================
    # 4. FX Graph Analysis
    # =============================================================================
    
    print("\n4. FX Graph Analysis...")
    
    try:
        # Trace enhanced controller with FX
        fx_controller = fx.symbolic_trace(enhanced_controller)
        print("✓ Enhanced controller FX tracing successful")
        
        controller_fx_info = analyze_fx_graph(fx_controller)
        
        # Walk through controller execution
        controller_execution = walk_fx_graph_execution(
            fx_controller, initial_temp, setpoint, error_integral, prev_control
        )
        
        # Trace full simulation with FX
        fx_simulator = fx.symbolic_trace(closed_loop_simulator)
        print("\n✓ Full simulation FX tracing successful")
        
        simulator_fx_info = analyze_fx_graph(fx_simulator)
        
        # Walk through simulation execution (first few steps only to save space)
        print("\nWalking through simulation execution (abbreviated)...")
        simulation_execution = walk_fx_graph_execution(fx_simulator, initial_temp, setpoint)
        
    except Exception as e:
        print(f"FX tracing failed: {e}")
        print("This can happen with complex control flow in the simulation loop.")
    
    # =============================================================================
    # 5. Comparison and Performance Analysis
    # =============================================================================
    
    print("\n5. Performance Comparison...")
    
    # Time the different approaches
    import time
    
    # Original model
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = enhanced_controller(initial_temp, setpoint, error_integral, prev_control)
    original_time = time.time() - start_time
    
    # TorchScript model (if available)
    try:
        start_time = time.time()
        for _ in range(100):
            _ = traced_controller(initial_temp, setpoint, error_integral, prev_control)
        torchscript_time = time.time() - start_time
        print(f"TorchScript speedup: {original_time/torchscript_time:.2f}x")
    except:
        print("TorchScript timing not available")
    
    # FX model (if available)
    try:
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = fx_controller(initial_temp, setpoint, error_integral, prev_control)
        fx_time = time.time() - start_time
        print(f"FX overhead: {fx_time/original_time:.2f}x")
    except:
        print("FX timing not available")
    


if __name__ == "__main__":
    main()
