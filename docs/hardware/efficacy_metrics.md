# Efficacy Metrics

In ASIC design we use the Energy-Delay product, but we also use Energy/Delay to calculate power. Why is the Energy-Delay product a good metric for evaluating the energy effectiveness of a circuit. And can we use the Energy-Delay product to compare complete engines like CPUs vs GPUs

---

## Why the Energy-Delay Product (EDP) is a Good Metric

The **Energy-Delay Product (EDP)**, defined as $EDP = E \times T$ (where $E$ is Energy and $T$ is Delay/Time), is a highly effective metric because it simultaneously penalizes inefficiency in *both* speed and energy consumption.

1.  **It Penalizes Undesirable Trade-offs:**
    * A circuit designer can achieve very low **Energy** ($E$) by dramatically slowing down the circuit ($T$ increases significantly). While $E$ is great, the poor **latency** ($T$) makes the circuit impractical.
    * Alternatively, a designer can achieve very low **Delay** ($T$) by significantly increasing the supply voltage, which quadratically increases energy ($E$). While $T$ is great, the high $E$ makes the circuit too hot or power-hungry.
    * **EDP** minimizes the *product* of these two competing factors, pushing the designer toward the optimal operating point where there is a judicious **balance** between high speed and low energy. A lower EDP is always better.

2.  **It is an Energy-of-Computation Metric:**
    * Your other metric, **Energy/Delay** ($E/T$), is simply **Power** ($P$), because $P = E/T$. Power is energy consumed *per unit of time*.
    * **Energy** ($E$) is Power $\times$ Time, or $E = P \times T$. Energy measures the total cost to complete an operation.
    * **EDP** is Energy $\times$ Time, or $EDP = E \times T = (P \times T) \times T = P \times T^2$. This is sometimes referred to as the **Power-Delay Squared Product** to highlight the quadratic penalty on delay. It essentially measures the **energy required for a fixed time-to-solution**.

3.  **Inherently Models Dynamic Power:**
    * In CMOS logic, the dynamic switching energy ($E_{sw}$) is proportional to the square of the supply voltage ($V_{DD}$), $E_{sw} \propto C_{L}V_{DD}^2$.
    * The delay ($T$) is inversely proportional to $V_{DD}$, $T \propto 1/V_{DD}$.
    * Therefore, $EDP \propto V_{DD}^2 \times (1/V_{DD})^2 = \text{constant}$. This classic relationship shows that EDP is generally **invariant** to voltage scaling (in ideal logic). Minimizing EDP forces a search for architectural and structural innovations, not just voltage tuning.

---
## EDP and the Pareto Optimal Front

When we plot the **Energy ($E$)** and **Delay ($T$)** of many different circuits/designs on a 2D scatter plot, the resulting boundary traced by the non-dominated points will be the **Pareto Frontier** (or Pareto Optimal Front).

This is a classic application of **multi-objective optimization** in circuit design.

---

## 📈 The Energy-Delay Pareto Frontier

In the context of circuit optimization, the two competing objectives are:

1.  **Minimize Energy ($E$):** The total energy consumed to complete the task (Joules).
2.  **Minimize Delay ($T$):** The time taken to complete the task (seconds).

### Defining the Pareto Frontier

The Pareto Frontier is a curve composed of **Pareto-Optimal** solutions. A design point (a specific $(E, T)$ pair) is considered **Pareto-Optimal** (or non-dominated) if it cannot improve one objective (e.g., lower $E$) without making the other objective worse (e.g., increasing $T$).

* Any design point that falls **above and to the right** of the frontier is **dominated** by at least one point on the frontier, meaning there is an existing design that is both faster and more energy-efficient.
* The frontier itself represents the **best possible trade-off** between speed and energy for the given technology and design space.


### Relationship to Energy-Delay Product (EDP)

The **Energy-Delay Product (EDP)**, where $EDP = E \times T$, is directly related to the Pareto Frontier:

1.  **Isolines:** If you plot lines of constant EDP on your $E$-$T$ graph, they are **hyperbolas** ($E = EDP/T$).
2.  **EDP Minimization:** The point on the Pareto Frontier where the constant-EDP hyperbola **just touches** (is tangent to) the curve is the design that achieves the **Minimum EDP**.
    * This minimum EDP point is often the target for general-purpose circuits as it represents the most balanced trade-off between speed and energy.
3.  **MEP and MDP:** The curve also allows you to identify:
    * **Minimum Energy Point (MEP):** The point on the frontier with the absolute lowest $E$ (usually the slowest practical design).
    * **Minimum Delay Point (MDP):** The point on the frontier with the absolute lowest $T$ (usually the most power-hungry design).

By tracing the Pareto Frontier, all the crucial information needed to choose a final design based on their specific priorities is provided. For example, a mobile device prioritizes MEP, while a high-performance server prioritizes a point closer to MDP or Minimum EDP.

## Comparing Complete Engines (CPUs vs. GPUs vs TPUs vs KPUs)

While the EDP is excellent for evaluating different design points *within the same architectural family* (e.g., comparing two versions of a cache or two different ASIC designs), **it has significant limitations when comparing complete, heterogenous compute engines like CPUs and GPUs.**

This is because $E$ and $T$ must be measured for the **same unit of work**.

### Key Limitations:

1.  **Defining the "Workload" (The Unit of Computation):**
    * The core problem is defining a single, comparable operation. Is it **Energy per Instruction (E/I)**? **Energy per Floating Point Operation (E/FLOP)**? **Energy per Frame**?
        * **CPUs** excel at low-latency, sequential, control-flow-heavy tasks (e.g., operating system, database transactions).
        * **GPUs** excel at highly parallel, throughput-oriented tasks (e.g., matrix multiplication, image processing).
        * **TPUs** specialize on one specific task: batched MLP, and are super energy efficient on that task.
        * **KPUs** try to minimize data movement across all parallel linear algebra operations by orchestrating system level data flows.
     
    A GPU will have a terrible EDP for running a single-threaded task with many branches, while a CPU's E/FLOP on a massive matrix multiply will look bad compared to a GPU. The comparison is not at the correct level of abstraction.

2.  **Architectural Differences:**
    * A CPU is designed for **low latency ($T$)** on single threads, featuring large caches and complex out-of-order execution logic, which consumes significant static power.
    * A GPU is designed for **high throughput (work per second)**, using thousands of smaller, simpler cores optimized for dense, simultaneous work, which is very energy-efficient when fully utilized.
    * A KPU is designed for **low latency ($T$), energy-efficient ($E$)** concurrency.

### Better Metrics for System-Level Comparison

To compare heterogeneous engines, engineers usually abandon EDP and use **Throughput-centric Metrics** measured against a relevant workload:

| Metric | Formula | Use Case & Context |
| :--- | :--- | :--- |
| **Energy/Operation** | $E_{\text{total}} / \text{Ops}_{\text{total}}$ | **Most common for domain-specific accelerators.** Measures the energy to complete a defined unit of work (e.g., Joules per inference, or **pico-Joules/MAC** for AI chips). |
| **Operations/Watt** | $\text{Ops}_{\text{total}} / P_{\text{avg}}$ | **The most direct measure of efficiency (Performance/Power).** This is the inverse of the Joules/Operation, usually expressed as **Tera-Operations Per Second per Watt (TOPS/W)**. Higher is better. |
| **Throughput/Area** | $\text{Ops}_{\text{total}} / (T \times \text{Area})$ | Used in ASIC design to evaluate efficiency of silicon usage (how many operations can be packed onto a small chip area). |

For a true comparison between distinctly different micro-architectural approaches, such as, CPU/GPU/TPU/KPU, you would need to measure the **TOPS/W** of both devices while running a standardized, common neural network model operators, such as MLPs, ResNets, or Attention-heads, to ensure they are performing a comparable amount of work.