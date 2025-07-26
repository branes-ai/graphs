#!/usr/bin/env python3
"""
Model Predictive Control (MPC) example with TorchScript and FX graph analysis.
This example demonstrates:
1. MPC controller for a chemical reactor temperature control
2. TorchScript graph generation and analysis
3. FX graph generation and analysis
4. Comparison between both approaches
"""

import numpy as np
import torch
import torch.nn as nn
import torch.fx as fx
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import solve_discrete_are
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Chemical Reactor Process Model
# =============================================================================

class ChemicalReactor:
    """Simplified chemical reactor model for temperature control."""
    
    def __init__(self):
        # Process parameters
        self.dt = 0.1  # Sampling time
        self.tau = 5.0  # Time constant
        self.K = 2.0    # Process gain
        self.delay = 2  # Process delay steps
        
        # State variables
        self.temp = 25.0  # Current temperature
        self.temp_history = [25.0] * self.delay
        self.disturbance = 0.0
        
    def step(self, heat_input: float, disturbance: float = 0.0) -> float:
        """Simulate one step of the reactor."""
        # First-order plus delay model
        if len(self.temp_history) >= self.delay:
            delayed_input = self.temp_history[-self.delay]
        else:
            delayed_input = 0.0
            
        # Temperature dynamics: dT/dt = (K*u - T + d) / tau
        temp_dot = (self.K * delayed_input - self.temp + disturbance) / self.tau
        self.temp += temp_dot * self.dt
        
        # Update history
        self.temp_history.append(heat_input)
        if len(self.temp_history) > self.delay + 10:
            self.temp_history.pop(0)
            
        return self.temp

# =============================================================================
# MPC Controller Implementation
# =============================================================================

class MPCController:
    """Model Predictive Controller for temperature regulation."""
    
    def __init__(self, prediction_horizon: int = 10, control_horizon: int = 3):
        self.N = prediction_horizon  # Prediction horizon
        self.M = control_horizon     # Control horizon
        
        # MPC weights
        self.Q = 10.0  # State weight
        self.R = 1.0   # Control weight
        self.S = 5.0   # Terminal weight
        
        # Constraints
        self.u_min = 0.0   # Minimum heat input
        self.u_max = 100.0 # Maximum heat input
        self.du_max = 10.0 # Maximum control rate
        
        # Process model parameters (simplified for MPC)
        self.A = np.array([[0.9]])  # State transition
        self.B = np.array([[0.2]])  # Control input
        self.C = np.array([[1.0]])  # Output
        
    def predict(self, x0: np.ndarray, u_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict future states and outputs."""
        x = x0.copy()
        states = [x]
        outputs = []
        
        for k in range(self.N):
            if k < len(u_seq):
                u = u_seq[k]
            else:
                u = u_seq[-1]  # Hold last control input
                
            x = self.A @ x + self.B * u
            y = self.C @ x
            
            states.append(x)
            outputs.append(y[0])
            
        return np.array(states[1:]), np.array(outputs)
    
    def cost_function(self, u_seq: np.ndarray, x0: np.ndarray, setpoint: float) -> float:
        """MPC cost function."""
        # Reshape control sequence
        u_seq = u_seq.reshape(-1)
        
        # Extend control sequence for prediction horizon
        u_extended = np.zeros(self.N)
        u_extended[:min(len(u_seq), self.N)] = u_seq[:self.N]
        if len(u_seq) < self.N:
            u_extended[len(u_seq):] = u_seq[-1]
        
        # Predict future states
        _, outputs = self.predict(x0, u_extended)
        
        # Calculate cost
        cost = 0.0
        
        # State tracking cost
        for k in range(self.N):
            error = outputs[k] - setpoint
            if k == self.N - 1:
                cost += self.S * error**2  # Terminal cost
            else:
                cost += self.Q * error**2
        
        # Control effort cost
        for k in range(len(u_seq)):
            cost += self.R * u_seq[k]**2
            
        # Control rate cost
        for k in range(1, len(u_seq)):
            du = u_seq[k] - u_seq[k-1]
            cost += 0.1 * du**2
            
        return cost
    
    def solve(self, current_state: np.ndarray, setpoint: float, last_u: float = 0.0) -> float:
        """Solve MPC optimization problem."""
        # Initial guess
        u0 = np.ones(self.M) * last_u
        
        # Constraints
        bounds = [(self.u_min, self.u_max) for _ in range(self.M)]
        
        # Control rate constraints
        constraints = []
        for k in range(self.M):
            if k == 0:
                # First control move constraint
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda u, k=k, last_u=last_u: self.du_max - abs(u[k] - last_u)
                })
            else:
                # Subsequent control move constraints
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda u, k=k: self.du_max - abs(u[k] - u[k-1])
                })
        
        # Solve optimization
        result = minimize(
            self.cost_function,
            u0,
            args=(current_state, setpoint),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'disp': False}
        )
        
        return result.x[0] if result.success else last_u

# =============================================================================
# Neural Network MPC Enhancement
# =============================================================================

class NeuralMPCPredictor(nn.Module):
    """Neural network to enhance MPC predictions."""
    
    def __init__(self, input_size: int = 4, hidden_size: int = 32, output_size: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict temperature correction."""
        return self.network(x)

class EnhancedMPCController(nn.Module):
    """MPC Controller enhanced with neural network."""
    
    def __init__(self):
        super().__init__()
        self.predictor = NeuralMPCPredictor()
        self.mpc = MPCController()
        
        # MPC parameters as tensors for TorchScript
        self.register_buffer('Q', torch.tensor(10.0))
        self.register_buffer('R', torch.tensor(1.0))
        self.register_buffer('N', torch.tensor(10))
        
    def predict_correction(self, state_info: torch.Tensor) -> torch.Tensor:
        """Predict model correction using neural network."""
        return self.predictor(state_info)
    
    def forward(self, current_temp: torch.Tensor, setpoint: torch.Tensor, 
                control_history: torch.Tensor) -> torch.Tensor:
        """Simplified MPC forward pass for TorchScript."""
        # Prepare input for neural predictor
        state_info = torch.cat([
            current_temp.unsqueeze(0),
            setpoint.unsqueeze(0), 
            control_history[-2:]  # Last 2 control inputs
        ])
        
        # Get model correction
        correction = self.predict_correction(state_info)
        
        # Simplified control law (for demonstration)
        error = setpoint - current_temp
        proportional = 2.0 * error
        integral = 0.1 * torch.sum(control_history)
        
        # Apply correction
        control_output = proportional + integral + correction.squeeze()
        
        # Apply constraints
        control_output = torch.clamp(control_output, 0.0, 100.0)
        
        return control_output

# =============================================================================
# Graph Analysis Functions
# =============================================================================

def analyze_torchscript_graph(model: torch.jit.ScriptModule) -> Dict[str, Any]:
    """Analyze TorchScript computational graph."""
    print("=== TorchScript Graph Analysis ===")
    
    # Get the graph
    graph = model.graph
    
    # Basic graph information
    info = {
        'nodes': [],
        'inputs': [],
        'outputs': [],
        'node_count': 0,
        'parameter_count': 0
    }
    
    # Analyze inputs
    for input_node in graph.inputs():
        info['inputs'].append({
            'name': input_node.debugName(),
            'type': str(input_node.type())
        })
    
    # Analyze outputs  
    for output_node in graph.outputs():
        info['outputs'].append({
            'name': output_node.debugName(),
            'type': str(output_node.type())
        })
    
    # Analyze nodes
    for node in graph.nodes():
        node_info = {
            'kind': node.kind(),
            'inputs': [inp.debugName() for inp in node.inputs()],
            'outputs': [out.debugName() for out in node.outputs()],
            'attributes': {}
        }
        
        # Get node attributes
        for attr_name in node.attributeNames():
            try:
                attr_value = node[attr_name]
                node_info['attributes'][attr_name] = str(attr_value)
            except:
                node_info['attributes'][attr_name] = "Unable to retrieve"
        
        info['nodes'].append(node_info)
        info['node_count'] += 1
    
    # Print graph structure
    print(f"Graph has {info['node_count']} nodes")
    print(f"Inputs: {len(info['inputs'])}")
    print(f"Outputs: {len(info['outputs'])}")
    
    print("\nGraph Structure:")
    for i, node in enumerate(info['nodes']):
        print(f"  Node {i}: {node['kind']}")
        if node['inputs']:
            print(f"    Inputs: {node['inputs']}")
        if node['outputs']:
            print(f"    Outputs: {node['outputs']}")
    
    return info

def analyze_fx_graph(model: fx.GraphModule) -> Dict[str, Any]:
    """Analyze FX computational graph."""
    print("\n=== FX Graph Analysis ===")
    
    info = {
        'nodes': [],
        'node_count': 0,
        'graph_code': str(model.code)
    }
    
    # Analyze each node
    for node in model.graph.nodes:
        node_info = {
            'name': node.name,
            'op': node.op,
            'target': str(node.target),
            'args': [str(arg) for arg in node.args],
            'kwargs': {k: str(v) for k, v in node.kwargs.items()},
            'users': [user.name for user in node.users.keys()]
        }
        info['nodes'].append(node_info)
        info['node_count'] += 1
    
    print(f"Graph has {info['node_count']} nodes")
    
    print("\nFX Graph Structure:")
    for i, node in enumerate(info['nodes']):
        print(f"  Node {i}: {node['name']} ({node['op']})")
        print(f"    Target: {node['target']}")
        if node['args']:
            print(f"    Args: {node['args']}")
        if node['kwargs']:
            print(f"    Kwargs: {node['kwargs']}")
        if node['users']:
            print(f"    Users: {node['users']}")
    
    print(f"\nGenerated Code:\n{info['graph_code']}")
    
    return info

def walk_fx_graph(model: fx.GraphModule, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Walk through FX graph execution and capture intermediate values."""
    print("\n=== Walking FX Graph Execution ===")
    
    intermediate_values = {}
    
    # Create a custom interpreter
    class ValueTracker(fx.Interpreter):
        def run_node(self, n: fx.Node) -> Any:
            result = super().run_node(n)
            intermediate_values[n.name] = result
            print(f"Node '{n.name}' ({n.op}): {type(result)} - Shape: {getattr(result, 'shape', 'N/A')}")
            return result
    
    # Run the tracker
    tracker = ValueTracker(model)
    final_result = tracker.run(input_data)
    
    print(f"\nFinal result: {final_result}")
    return intermediate_values

# =============================================================================
# Main Demonstration
# =============================================================================

def main():
    """Main demonstration of MPC with TorchScript and FX."""
    
    print("Model Predictive Control with TorchScript and FX Graph Analysis")
    print("=" * 70)
    
    # Create and test basic MPC
    reactor = ChemicalReactor()
    mpc = MPCController()
    
    print("\n1. Testing Classical MPC Controller...")
    current_state = np.array([reactor.temp])
    setpoint = 50.0
    control_action = mpc.solve(current_state, setpoint)
    print(f"Current temperature: {reactor.temp:.2f}°C")
    print(f"Setpoint: {setpoint}°C") 
    print(f"Control action: {control_action:.2f}")
    
    # Create enhanced neural MPC
    print("\n2. Creating Enhanced Neural MPC...")
    enhanced_mpc = EnhancedMPCController()
    
    # Prepare sample data
    current_temp = torch.tensor(25.0)
    setpoint_tensor = torch.tensor(50.0)
    control_history = torch.tensor([0.0, 5.0])
    
    # Test the model
    with torch.no_grad():
        neural_control = enhanced_mpc(current_temp, setpoint_tensor, control_history)
        print(f"Neural enhanced control action: {neural_control.item():.2f}")
    
    # =============================================================================
    # TorchScript Analysis
    # =============================================================================
    
    print("\n3. Converting to TorchScript...")
    try:
        # Create traced model
        traced_model = torch.jit.trace(
            enhanced_mpc,
            (current_temp, setpoint_tensor, control_history)
        )
        
        print("TorchScript tracing successful!")
        
        # Analyze TorchScript graph
        torchscript_info = analyze_torchscript_graph(traced_model)
        
        # Test traced model
        with torch.no_grad():
            traced_output = traced_model(current_temp, setpoint_tensor, control_history)
            print(f"TorchScript output: {traced_output.item():.2f}")
            
    except Exception as e:
        print(f"TorchScript tracing failed: {e}")
        print("This is common with complex control logic. Using scripting instead...")
        
        # Try scripting instead
        try:
            scripted_model = torch.jit.script(enhanced_mpc)
            print("TorchScript scripting successful!")
            torchscript_info = analyze_torchscript_graph(scripted_model)
        except Exception as e2:
            print(f"TorchScript scripting also failed: {e2}")
            scripted_model = None
    
    # =============================================================================
    # FX Analysis  
    # =============================================================================
    
    print("\n4. Converting to FX Graph...")
    try:
        # Create FX graph
        fx_model = fx.symbolic_trace(enhanced_mpc)
        print("FX symbolic tracing successful!")
        
        # Analyze FX graph
        fx_info = analyze_fx_graph(fx_model)
        
        # Walk through FX graph execution
        intermediate_values = walk_fx_graph(fx_model, (current_temp, setpoint_tensor, control_history))
        
        # Test FX model
        with torch.no_grad():
            fx_output = fx_model(current_temp, setpoint_tensor, control_history)
            print(f"FX output: {fx_output.item():.2f}")
            
    except Exception as e:
        print(f"FX tracing failed: {e}")
        print("FX may have issues with complex control flow and external dependencies.")
    
    # =============================================================================
    # Simple Function Analysis (Non-PyTorch)
    # =============================================================================
    
    print("\n5. Analyzing Simple Numerical Function...")
    
    def simple_mpc_step(temp: float, setpoint: float, kp: float = 2.0) -> float:
        """Simple proportional controller."""
        error = setpoint - temp
        control = kp * error
        return max(0.0, min(100.0, control))  # Apply constraints
    
    # Convert to PyTorch function for analysis
    class SimpleMPC(nn.Module):
        def __init__(self):
            super().__init__()
            self.kp = nn.Parameter(torch.tensor(2.0))
            
        def forward(self, temp: torch.Tensor, setpoint: torch.Tensor) -> torch.Tensor:
            error = setpoint - temp
            control = self.kp * error
            return torch.clamp(control, 0.0, 100.0)
    
    simple_model = SimpleMPC()
    
    # TorchScript for simple model
    simple_traced = torch.jit.trace(simple_model, (current_temp, setpoint_tensor))
    print("\nSimple Model TorchScript Analysis:")
    analyze_torchscript_graph(simple_traced)
    
    # FX for simple model
    simple_fx = fx.symbolic_trace(simple_model)
    print("\nSimple Model FX Analysis:")
    analyze_fx_graph(simple_fx)
    

if __name__ == "__main__":
    main()
