# KPU SKU generation 

No example input-spec YAMLs ship in the tree — the canonical way to see one is --from-sku round-trip. 
Let create one and read through the nested types.

```bash
> python cli/generate_kpu_sku.py -h
   usage: generate_kpu_sku.py [-h] (--input INPUT | --from-sku FROM_SKU)
                                [--output OUTPUT] [--validate] [--strict]

python cli/generate_kpu_sku.py --from-sku kpu_t256_32x32_lp5x16_16nm_tsmc_ffp --output /tmp/t256_input_spec.yaml
```

There are no example input-spec YAMLs checked in — start from a round-trip template and edit it. Here's the workflow plus the field-by-field rules.

## Workflow

```bash
  # 1. Extract a template from an existing SKU you want to base yours on
  python -c "
  import yaml; from embodied_schemas import load_kpus
  from graphs.hardware.kpu_sku_generator import input_spec_from_kpu_entry
  spec = input_spec_from_kpu_entry(load_kpus()['kpu_t256_32x32_lp5x16_16nm_tsmc_ffp'])
  print(yaml.safe_dump(spec.model_dump(mode='json', exclude_none=True),
                       sort_keys=False, default_flow_style=False, indent=2))
  " > my_kpu_spec.yaml

  # 2. Edit it
  $EDITOR my_kpu_spec.yaml

  # 3. Generate the full KPUEntry (rolls up die, perf, power) and validate
  python cli/generate_kpu_sku.py --input my_kpu_spec.yaml \
      --output my_new_kpu.yaml --validate
```

The generator owns the derived fields (die.transistors_billion, die.die_size_mm2, performance.*, power.tdp_watts roll-up, power.idle_power_watts); you don't author those.

## Field-by-field
  
## Identity (top-level)

```yaml
  id: kpu_t512_32x32_lp5x32_12nm_gf_fdx        # globally unique; convention: kpu_t<count>_<rows>x<cols>_<mem><ch>_<value>nm_<foundry>_<library>
  name: Stillwater KPU-T512
  vendor: stillwater
  process_node_id: tsmc_n16      # MUST resolve in data/process-nodes/<foundry>/<node>.yaml
  last_updated: '2026-05-10'     # YYYY-MM-DD
  notes: '...'                   # optional, free text

  kpu_architecture

  kpu_architecture:
    total_tiles: 512                 # MUST equal sum(tiles[].num_tiles)
    multi_precision_alu: [int4, int8, bf16, fp32]   # chip-wide precisions
    tiles:                           # one entry per tile *class* (not per tile)
    - tile_type: INT8-primary        # label, free string
      num_tiles: 358                 # > 0
      pe_array_rows: 20              # PE grid per tile
      pe_array_cols: 20
      pe_circuit_class: balanced_logic   # density/energy/leakage lookup key
      ops_per_tile_per_clock:        # peak ops/clock by precision
        int8: 800.0                  # = rows*cols*ops_per_pe (e.g., 20*20*2)
        int4: 1600.0
        bf16: 400.0
        fp16: 400.0
      schedule_class: output_stationary    # default; rarely changed
      pipeline_fill_cycles: 20
      pipeline_drain_cycles: 20
    - tile_type: BF16-primary
      ...
    noc:
      topology: mesh_2d
      mesh_rows: 16                  # mesh_rows * mesh_cols MUST equal total_tiles
      mesh_cols: 16                  # (also drives noc.num_routers)
      flit_bytes: 16
      router_circuit_class: hp_logic # use HP for low-latency routing
      bisection_bandwidth_gbps: 256.0
    memory:
      memory_type: lpddr5            # MemoryType enum: lpddr5, hbm3, ddr5, gddr6, ...
      memory_size_gb: 32.0
      memory_bus_bits: 256
      memory_bandwidth_gbps: 256.0
      memory_controllers: 16
      l1_kib_per_pe: 4               # per-PE scratch (0 if none)
      l2_kib_per_tile: 32
      l3_kib_per_tile: 256           # required, > 0
```

pe_circuit_class and router_circuit_class must be values the referenced ProcessNode declares 
densities for (hp_logic, balanced_logic, lp_logic, sram_hd, sram_hc, analog, io).

## silicon_bin — the area decomposition

Each block declares how to compute its transistor count from the architecture, so when you change num_tiles or memory sizes the area scales automatically. Five
transistor_source.kind values, with strict field rules:


|      kind      |     Required fields     |                   count_ref form                   |
|----------------|-------------------------|----------------------------------------------------|
| fixed          | mtx only                | (omit)                                             |
| per_pe         | per_unit_mtx, count_ref | tile.<tile_type> (matches a tile_type you defined) |
| per_kib        | per_unit_mtx, count_ref | l1_total_kib / l2_total_kib / l3_total_kib         |
| per_router     | per_unit_mtx, count_ref | noc                                                |
| per_controller | per_unit_mtx, count_ref | memory                                             |

```yaml
  silicon_bin:
    blocks:
    - name: pe_int8                  # one per PE-datapath flavor
      circuit_class: balanced_logic
      transistor_source: {kind: per_pe, per_unit_mtx: 0.006, count_ref: tile.INT8-primary}
    - name: l3_sram
      circuit_class: sram_hd
      transistor_source: {kind: per_kib, per_unit_mtx: 0.052, count_ref: l3_total_kib}
    - name: noc_routers
      circuit_class: hp_logic
      transistor_source: {kind: per_router, per_unit_mtx: 0.8, count_ref: noc}
    - name: memory_phys
      circuit_class: analog
      transistor_source: {kind: per_controller, per_unit_mtx: 6.0, count_ref: memory}
    - name: io_pads
      circuit_class: io
      transistor_source: {kind: fixed, mtx: 8.0}
    - name: control_logic
      circuit_class: balanced_logic
      transistor_source: {kind: fixed, mtx: 18.0}
```

The Pydantic validator rejects mixed combinations at load time (e.g., kind: fixed with a count_ref set, or kind: per_pe missing per_unit_mtx).

## clocks

```yaml
  clocks:
    base_clock_mhz: 1000.0           # > 0
    boost_clock_mhz: 1750.0          # > 0; should be >= base
```

These are the chip-level clock floor/ceiling. Per-profile clocks live in thermal_profiles[].clock_mhz.

## thermal_profiles + default_thermal_profile

Each operating point must be manually provided - the generator does not invent TDPs.

```yaml
  thermal_profiles:
  - name: 15W                        # label; referenced by default_thermal_profile
    tdp_watts: 15.0
    clock_mhz: 1100.0                # must lie within [base, boost]
    cooling_solution_id: passive_heatsink_large   # MUST resolve in data/cooling-solutions/
    # Optional per-precision calibration. Both maps: values in [0, 1].
    # Omit entirely until you have measurements; loaders fall back to
    # ~0.70 efficiency / ~0.95 utilization placeholders.
    efficiency_factor_by_precision: {int8: 0.68, bf16: 0.63, int4: 0.73}
    tile_utilization_by_precision:   {int8: 0.93, bf16: 0.88, int4: 0.95}
  - name: 30W
    tdp_watts: 30.0
    clock_mhz: 1400.0
    cooling_solution_id: active_fan
  - name: 50W
    tdp_watts: 50.0
    clock_mhz: 1650.0
    cooling_solution_id: active_fan
  default_thermal_profile: 30W       # MUST match one of the names above
```

The generator validates that each profile's tdp_watts fits its cooling solution's envelope (max_total_w, max_power_density_w_per_mm2) — this is exactly what
show_compute_product.py's cross-check panel surfaces.

## market
  
```yaml
  market:
    launch_date: 2026-Q2             # optional, free string
    launch_msrp_usd: 1092.0          # optional
    target_market: embodied          # edge / embodied / datacenter
    product_family: Stillwater KPU
    model_tier: high                 # entry / mid / high / datacenter
    is_available: false
    is_discontinued: false
```

## Authoring tips for new SKUs

1. Start from the closest existing SKU's round-trip dump — the silicon_bin coefficients (per_unit_mtx) are the hardest numbers to invent and are usually portable across SKUs
on the same node.
2. Change architecture first, then run --validate — bad total_tiles/mesh mismatches and dangling count_refs fail fast at Pydantic load.
3. Run cli/show_compute_product.py <new_id> after generation — the thermal cross-check panel catches TDP-vs-cooling violations that the generator may flag as warnings.
4. The four shipped SKUs (t64/t128/t256/t768) are your reference points — they cover the entry/mid/high/datacenter tiers and span tsmc_n16 → tsmc_n7, so most new SKUs are an interpolation.

A worked round-trip dump for kpu_t256_32x32_lp5x16_16nm_tsmc_ffp is in /tmp/t256_input_spec.yaml if you want to inspect a complete file.
