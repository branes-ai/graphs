# Capability Tier CLI Tools - Design Specification

**Date**: 2026-01-18
**Status**: Phase 3 Comparison Implemented
**Purpose**: Tools for product designers to understand compute power impact on battery life and mission parameters

## Capability Tier Framework

| Tier | Power Envelope | Applications | Key Constraints |
|------|----------------|--------------|-----------------|
| **Wearable AI** | 0.1-1W | Smartwatches, AR glasses, health monitors | Skin-safe temp, all-day battery |
| **Micro-Autonomy** | 1-10W | Drones, handheld scanners, smart IoT | Weight, battery size, thermal |
| **Industrial Edge** | 10-30W | Factory AMRs, cobots, automated sorting | Continuous operation, reliability |
| **Embodied AI** | 30-100W | Quadrupeds, humanoids, world-model sim | Dynamic movement, complex perception |
| **Automotive AI** | 100-500W | L3/L3+/L4 autonomous driving | Safety margins, redundancy |

## Energy Allocation Model

Each tier allocates power across three primary subsystems:

```
Total Power Budget = Perception + Control + Movement + System Overhead

Where:
- Perception: Sensors, preprocessing, DNN inference (vision, lidar, audio)
- Control: Planning, decision making, state estimation
- Movement: Actuators, motors, servos (platform-dependent)
- System Overhead: Comms, logging, thermal management
```

Typical allocation ratios by tier:

| Tier | Perception | Control | Movement | Overhead |
|------|------------|---------|----------|----------|
| Wearable AI | 45-55% | 15-25% | 10-20% | 10-20% |
| Micro-Autonomy | 40-50% | 10-20% | 25-35% | 5-15% |
| Industrial Edge | 35-45% | 15-25% | 25-35% | 5-15% |
| Embodied AI | 25-35% | 20-30% | 30-40% | 5-15% |
| Automotive AI | 50-60% | 15-25% | 5-15%* | 10-20% |

*Automotive "movement" is steering/braking assist, not propulsion

---

## Proposed CLI Tools

### 1. Discovery Tools

#### `discover_capability_tiers.py`
**Purpose**: List all capability tiers with their characteristics and supported platforms.

```bash
# List all tiers
./cli/discover_capability_tiers.py

# Show details for specific tier
./cli/discover_capability_tiers.py --tier embodied-ai

# List platforms in a tier
./cli/discover_capability_tiers.py --tier industrial-edge --list-platforms
```

**Output**: Tier definitions, power ranges, typical applications, supported hardware.

---

#### `discover_platforms_by_tier.py`
**Purpose**: Find hardware platforms that fit within a capability tier's power envelope.

```bash
# Find all platforms for micro-autonomy (1-10W)
./cli/discover_platforms_by_tier.py --tier micro-autonomy

# Find platforms with specific constraints
./cli/discover_platforms_by_tier.py --tier embodied-ai --min-tops 20 --max-tdp 75

# Filter by vendor
./cli/discover_platforms_by_tier.py --tier industrial-edge --vendor nvidia
```

**Output**: Platform list with TDP, TOPS, TOPS/W, form factor.

---

#### `discover_models_for_budget.py`
**Purpose**: Find DNN models that can run within a power/latency budget.

```bash
# Find models for 5W perception budget at 30fps
./cli/discover_models_for_budget.py --power-budget 5 --target-fps 30 --task detection

# Find models for specific platform
./cli/discover_models_for_budget.py --platform Jetson-Orin-Nano --power-budget 8 --task segmentation

# Multi-task discovery
./cli/discover_models_for_budget.py --power-budget 15 --tasks detection segmentation depth
```

**Output**: Model recommendations ranked by efficiency (accuracy per watt).

---

#### `discover_battery_configurations.py`
**Purpose**: List battery options for different mission durations and platforms.

```bash
# Find batteries for 2-hour drone mission
./cli/discover_battery_configurations.py --tier micro-autonomy --mission-hours 2

# Find batteries for 8-hour warehouse shift
./cli/discover_battery_configurations.py --tier industrial-edge --mission-hours 8

# With weight constraint
./cli/discover_battery_configurations.py --tier embodied-ai --mission-hours 4 --max-weight-kg 5
```

**Output**: Battery options with Wh, weight, volume, estimated runtime.

---

### 2. Exploration Tools

#### `explore_power_allocation.py`
**Purpose**: Understand how power is allocated across subsystems for different configurations.

```bash
# Show typical allocation for tier
./cli/explore_power_allocation.py --tier embodied-ai

# Explore allocation for specific platform + model
./cli/explore_power_allocation.py --platform Jetson-Orin-AGX --model yolov8n --task detection

# Interactive allocation exploration
./cli/explore_power_allocation.py --total-budget 50 --interactive
```

**Output**: Power breakdown visualization, allocation recommendations.

---

#### `explore_mission_profiles.py`
**Purpose**: Understand mission characteristics and their power implications.

```bash
# List standard mission profiles
./cli/explore_mission_profiles.py --tier micro-autonomy

# Show detailed profile
./cli/explore_mission_profiles.py --profile drone-inspection

# Custom mission definition
./cli/explore_mission_profiles.py --define \
    --perception-duty-cycle 0.8 \
    --movement-duty-cycle 0.6 \
    --peak-perception-load 10 \
    --cruise-perception-load 5
```

**Output**: Mission profile details, power curves, duty cycles.

---

#### `explore_battery_life.py`
**Purpose**: Understand factors affecting battery life and tradeoffs.

```bash
# Explore battery life for configuration
./cli/explore_battery_life.py \
    --platform Jetson-Orin-Nano \
    --battery-wh 100 \
    --model resnet18

# Show sensitivity analysis
./cli/explore_battery_life.py \
    --platform Jetson-Orin-Nano \
    --battery-wh 100 \
    --sensitivity

# Compare duty cycle impacts
./cli/explore_battery_life.py \
    --platform Jetson-Orin-Nano \
    --battery-wh 100 \
    --duty-cycles 0.5 0.75 1.0
```

**Output**: Battery life breakdown, sensitivity charts, optimization suggestions.

---

#### `explore_perception_control_tradeoff.py`
**Purpose**: Understand tradeoffs between perception capability and control responsiveness.

```bash
# Explore tradeoff space
./cli/explore_perception_control_tradeoff.py --tier embodied-ai --total-budget 60

# With specific constraints
./cli/explore_perception_control_tradeoff.py \
    --tier embodied-ai \
    --min-perception-fps 15 \
    --min-control-hz 100

# For specific application
./cli/explore_perception_control_tradeoff.py --application quadruped-locomotion
```

**Output**: Pareto frontier, allocation recommendations, constraint violations.

---

#### `explore_thermal_envelope.py`
**Purpose**: Understand thermal constraints and their impact on sustained performance.

```bash
# Explore thermal limits
./cli/explore_thermal_envelope.py --platform Jetson-Orin-Nano

# With ambient temperature
./cli/explore_thermal_envelope.py --platform Jetson-Orin-AGX --ambient-c 40

# Sustained vs burst performance
./cli/explore_thermal_envelope.py --platform Jetson-Orin-AGX --mission-hours 2
```

**Output**: Thermal headroom, throttling thresholds, sustained power limits.

---

### 3. Estimation Tools

#### `estimate_mission_duration.py`
**Purpose**: Estimate how long a mission can run given hardware, model, and battery.

```bash
# Basic estimation
./cli/estimate_mission_duration.py \
    --platform Jetson-Orin-Nano \
    --model yolov8n \
    --battery-wh 150

# With mission profile
./cli/estimate_mission_duration.py \
    --platform Jetson-Orin-Nano \
    --model yolov8n \
    --battery-wh 150 \
    --profile warehouse-amr

# With movement power
./cli/estimate_mission_duration.py \
    --platform Jetson-Orin-Nano \
    --model yolov8n \
    --battery-wh 150 \
    --movement-power 20 \
    --movement-duty-cycle 0.6
```

**Output**: Estimated runtime, power breakdown, confidence interval.

---

#### `estimate_power_consumption.py`
**Purpose**: Estimate total system power for a given configuration.

```bash
# Estimate for model + platform
./cli/estimate_power_consumption.py \
    --platform Jetson-Orin-AGX \
    --model resnet50 \
    --batch-size 1

# Multi-model pipeline
./cli/estimate_power_consumption.py \
    --platform Jetson-Orin-AGX \
    --models yolov8n:detection resnet18:classification \
    --pipeline

# Include movement subsystem
./cli/estimate_power_consumption.py \
    --platform Jetson-Orin-AGX \
    --model yolov8n \
    --movement-subsystem quadruped-12dof
```

**Output**: Per-subsystem power, total power, utilization metrics.

---

#### `estimate_battery_requirements.py`
**Purpose**: Estimate battery size needed for mission requirements.

```bash
# Basic battery sizing
./cli/estimate_battery_requirements.py \
    --platform Jetson-Orin-Nano \
    --model yolov8n \
    --mission-hours 4

# With safety margin
./cli/estimate_battery_requirements.py \
    --platform Jetson-Orin-Nano \
    --model yolov8n \
    --mission-hours 4 \
    --safety-margin 1.2

# With weight constraint (find feasible configurations)
./cli/estimate_battery_requirements.py \
    --tier micro-autonomy \
    --mission-hours 2 \
    --max-battery-weight-kg 0.5
```

**Output**: Required Wh, recommended battery specs, weight/volume estimates.

---

#### `estimate_perception_budget.py`
**Purpose**: Estimate power budget needed for perception requirements.

```bash
# For single model
./cli/estimate_perception_budget.py \
    --model yolov8n \
    --target-fps 30 \
    --platform Jetson-Orin-Nano

# For perception pipeline
./cli/estimate_perception_budget.py \
    --pipeline "yolov8n@30fps + depth_anything@15fps + audio_classifier@10fps" \
    --platform Jetson-Orin-AGX

# Find model to fit budget
./cli/estimate_perception_budget.py \
    --task detection \
    --target-fps 30 \
    --max-power 5 \
    --platform Jetson-Orin-Nano
```

**Output**: Power requirements, model recommendations, tradeoff options.

---

#### `estimate_operational_range.py`
**Purpose**: Estimate operational range/capability for mobile platforms.

```bash
# Drone range estimation
./cli/estimate_operational_range.py \
    --platform-type drone \
    --compute-platform Jetson-Orin-Nano \
    --battery-wh 100 \
    --model yolov8n

# AMR operational hours
./cli/estimate_operational_range.py \
    --platform-type amr \
    --compute-platform Jetson-Orin-AGX \
    --battery-wh 500 \
    --model resnet50

# Quadruped range
./cli/estimate_operational_range.py \
    --platform-type quadruped \
    --compute-platform Jetson-Orin-NX \
    --battery-wh 200 \
    --model yolov8s
```

**Output**: Range/duration estimates, limiting factors, optimization suggestions.

---

### 4. Comparison Tools

#### `compare_tier_platforms.py`
**Purpose**: Compare all platforms within a capability tier.

```bash
# Compare micro-autonomy platforms
./cli/compare_tier_platforms.py --tier micro-autonomy

# Compare with specific workload
./cli/compare_tier_platforms.py --tier industrial-edge --model yolov8n

# Rank by efficiency
./cli/compare_tier_platforms.py --tier embodied-ai --rank-by tops-per-watt
```

**Output**: Platform comparison table, efficiency rankings, recommendations.

---

#### `compare_power_allocations.py`
**Purpose**: Compare different power allocation strategies.

```bash
# Compare allocation strategies
./cli/compare_power_allocations.py \
    --total-budget 50 \
    --strategies perception-heavy balanced control-heavy

# For specific application
./cli/compare_power_allocations.py \
    --application warehouse-navigation \
    --total-budget 30

# With mission profile
./cli/compare_power_allocations.py \
    --profile drone-inspection \
    --total-budget 15
```

**Output**: Strategy comparison, mission impact analysis, recommendations.

---

#### `compare_mission_configurations.py`
**Purpose**: Compare different hardware/model configurations for same mission.

```bash
# Compare configurations for mission
./cli/compare_mission_configurations.py \
    --mission warehouse-amr-8hr \
    --configurations config1.yaml config2.yaml config3.yaml

# Auto-generate configurations
./cli/compare_mission_configurations.py \
    --mission drone-inspection-2hr \
    --tier micro-autonomy \
    --auto-configure
```

**Output**: Configuration comparison, cost/performance tradeoffs, recommendation.

---

#### `compare_battery_strategies.py`
**Purpose**: Compare battery configurations for mission requirements.

```bash
# Compare battery options
./cli/compare_battery_strategies.py \
    --platform Jetson-Orin-Nano \
    --model yolov8n \
    --mission-hours 4

# With hot-swap consideration
./cli/compare_battery_strategies.py \
    --platform Jetson-Orin-AGX \
    --model resnet50 \
    --mission-hours 8 \
    --allow-hot-swap

# Weight-constrained comparison
./cli/compare_battery_strategies.py \
    --tier micro-autonomy \
    --mission-hours 2 \
    --max-weight-kg 1.0
```

**Output**: Battery option comparison, runtime estimates, weight/cost tradeoffs.

---

#### `compare_perception_pipelines.py`
**Purpose**: Compare different perception pipeline configurations.

```bash
# Compare detection models
./cli/compare_perception_pipelines.py \
    --task detection \
    --platform Jetson-Orin-Nano \
    --target-fps 30

# Compare multi-task pipelines
./cli/compare_perception_pipelines.py \
    --pipelines \
        "yolov8n + depth_anything_small" \
        "yolov8s + midas_small" \
        "rt-detr-s + depth_anything_base" \
    --platform Jetson-Orin-AGX

# Power-constrained comparison
./cli/compare_perception_pipelines.py \
    --task detection+segmentation \
    --platform Jetson-Orin-Nano \
    --max-power 8
```

**Output**: Pipeline comparison, accuracy/power tradeoffs, recommendations.

---

### 5. Benchmark Tools

#### `benchmark_platform_power.py`
**Purpose**: Measure actual power consumption on target hardware.

```bash
# Benchmark platform with model
./cli/benchmark_platform_power.py \
    --model yolov8n \
    --duration 60

# Benchmark at different utilization levels
./cli/benchmark_platform_power.py \
    --model resnet50 \
    --utilization-sweep 25 50 75 100

# Benchmark pipeline
./cli/benchmark_platform_power.py \
    --models yolov8n resnet18 \
    --pipeline
```

**Output**: Measured power data, utilization curves, thermal observations.

---

#### `benchmark_mission_simulation.py`
**Purpose**: Simulate and measure power for mission profile.

```bash
# Simulate mission
./cli/benchmark_mission_simulation.py \
    --profile warehouse-amr \
    --model yolov8n \
    --duration 3600

# With real movement data
./cli/benchmark_mission_simulation.py \
    --profile custom \
    --movement-trace movement_log.csv \
    --model yolov8n
```

**Output**: Power profile over time, energy consumption, thermal data.

---

#### `benchmark_battery_runtime.py`
**Purpose**: Measure actual battery runtime for configuration.

```bash
# Full battery test
./cli/benchmark_battery_runtime.py \
    --model yolov8n \
    --until-shutdown

# Partial test with extrapolation
./cli/benchmark_battery_runtime.py \
    --model yolov8n \
    --test-duration 1800 \
    --extrapolate
```

**Output**: Measured runtime, discharge curve, temperature data.

---

#### `benchmark_thermal_sustained.py`
**Purpose**: Measure sustained performance under thermal constraints.

```bash
# Thermal stress test
./cli/benchmark_thermal_sustained.py \
    --model resnet50 \
    --duration 3600

# At elevated ambient
./cli/benchmark_thermal_sustained.py \
    --model resnet50 \
    --ambient-temp 45 \
    --duration 1800
```

**Output**: Sustained throughput, thermal throttling events, power curves.

---

## Summary

| Category | Count | Purpose |
|----------|-------|---------|
| `discover_` | 4 | Find platforms, models, batteries, tiers |
| `explore_` | 6 | Understand tradeoffs, allocations, constraints |
| `estimate_` | 6 | Calculate mission duration, power, battery needs |
| `compare_` | 5 | Compare configurations, strategies, pipelines |
| `benchmark_` | 4 | Measure actual power, runtime, thermal |
| **Total** | **25** | |

---

## Data Models Required

To support these tools, we need new data structures:

### `CapabilityTier`
```python
@dataclass
class CapabilityTier:
    name: str  # micro-autonomy, industrial-edge, embodied-ai, automotive-ai
    power_min_w: float
    power_max_w: float
    typical_applications: List[str]
    typical_allocation: PowerAllocation
    constraints: TierConstraints
```

### `PowerAllocation`
```python
@dataclass
class PowerAllocation:
    perception_ratio: float  # 0.0-1.0
    control_ratio: float
    movement_ratio: float
    overhead_ratio: float

    def validate(self) -> bool:
        return abs(sum([self.perception_ratio, self.control_ratio,
                       self.movement_ratio, self.overhead_ratio]) - 1.0) < 0.01
```

### `MissionProfile`
```python
@dataclass
class MissionProfile:
    name: str
    tier: CapabilityTier
    duration_hours: float
    perception_duty_cycle: float  # 0.0-1.0
    control_duty_cycle: float
    movement_duty_cycle: float
    peak_perception_power_w: float
    cruise_perception_power_w: float
    peak_movement_power_w: float
    cruise_movement_power_w: float
```

### `BatteryConfiguration`
```python
@dataclass
class BatteryConfiguration:
    chemistry: str  # LiPo, Li-ion, LiFePO4
    capacity_wh: float
    voltage_nominal: float
    weight_kg: float
    volume_cm3: float
    max_discharge_rate_c: float
    cycle_life: int
    operating_temp_min_c: float
    operating_temp_max_c: float
```

---

## Implementation Status

### Phase 1: Foundation - COMPLETE
1. [x] Data models (CapabilityTier, PowerAllocation, MissionProfile, BatteryConfiguration)
   - `src/graphs/mission/capability_tiers.py` - 5 tiers defined
   - `src/graphs/mission/power_allocation.py` - Allocation models
   - `src/graphs/mission/mission_profiles.py` - 11 mission profiles
   - `src/graphs/mission/battery.py` - 10 battery configurations
2. [x] `discover_capability_tiers.py` - List/explore capability tiers
3. [x] `discover_platforms_by_tier.py` - Find platforms by tier using mapper registry
4. [x] `explore_power_allocation.py` - Analyze power allocation strategies

### Phase 2: Estimation - COMPLETE
5. [x] `estimate_power_consumption.py` - Estimate system power by subsystem
   - Supports tier-specific and custom power allocations
   - Integrates with UnifiedAnalyzer for model-specific estimates
   - Visual breakdown of perception/control/movement/overhead
6. [x] `estimate_mission_duration.py` - Estimate mission runtime
   - Uses mission profiles for duty cycle adjustment
   - Supports predefined and custom battery configurations
   - Confidence scoring based on estimation quality
7. [x] `estimate_battery_requirements.py` - Size batteries for missions
   - Recommends batteries from database that meet requirements
   - Estimates weight and volume for custom batteries
   - Supports weight constraints for mobile platforms
8. [x] `explore_battery_life.py` - Analyze battery life tradeoffs
   - Power sensitivity analysis
   - Model comparison for runtime impact
   - Duty cycle impact analysis
   - Optimization suggestions

### Phase 3: Comparison - COMPLETE
9. [x] `compare_tier_platforms.py` - Compare platforms in a tier
   - Efficiency rankings by TDP, memory, or performance
   - Vendor breakdown and summary statistics
   - Model-specific performance estimates
10. [x] `compare_power_allocations.py` - Compare allocation strategies
    - Visual power distribution comparison
    - Capability estimates per strategy
    - Mission suitability analysis
11. [x] `compare_mission_configurations.py` - Compare configs for missions
    - Auto-generate configurations from tier
    - Score by runtime, weight, perception capability
    - Ranked recommendations with warnings
12. [x] `discover_models_for_budget.py` - Find models for budget
    - 18 models across detection, classification, segmentation, depth, pose
    - Power and FPS budget constraints
    - Efficiency and accuracy rankings

### Phase 4: Advanced - PENDING
13-25. Remaining tools

---

## Integration Notes

### Hardware Mapper Registry
The `discover_platforms_by_tier.py` tool uses the hardware mapper registry at
`src/graphs/hardware/mappers/__init__.py` which provides:
- `list_all_mappers()` - List all 44 available hardware platforms
- `get_mapper_info(name)` - Get TDP, memory, vendor, category without instantiating
- `list_mappers_by_tdp_range(min_w, max_w)` - Filter by power envelope

### embodied-schemas Integration
The capability tier tools are designed to integrate with the `embodied-schemas`
hardware registry. The mapper registry provides a bridge between the analysis
framework and the hardware specifications in embodied-schemas.

---

## Questions Resolved

1. Are the capability tier definitions and power ranges correct? **YES**
2. Should we add more tiers (e.g., "Wearable AI" at 0.1-1W)? **YES - Added**
3. Are there other key tools needed for product designers? **TBD in Phase 2-4**
4. Should mission profiles be predefined or fully customizable? **Both - 11 predefined + custom support**
5. Integration with existing hardware registry in `embodied-schemas`? **Planned via mapper registry**
