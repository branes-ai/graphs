# Makefile for CUDA kernel validation
# Target: Jetson Orin Nano (SM 8.7)

NVCC = nvcc
NVCC_FLAGS = -arch=sm_87 -O3 -lineinfo

# For desktop GPUs (uncomment appropriate line):
# NVCC_FLAGS = -arch=sm_89 -O3 -lineinfo  # RTX 4090
# NVCC_FLAGS = -arch=sm_86 -O3 -lineinfo  # RTX 3090
# NVCC_FLAGS = -arch=sm_80 -O3 -lineinfo  # A100

# Occupancy reporting
NVCC_FLAGS += --ptxas-options=-v

TARGETS = matvec_kernel matmul_kernel

all: $(TARGETS)

matvec_kernel: matvec_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

matmul_kernel: matmul_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Generate PTX for analysis
ptx: matvec_kernel.ptx matmul_kernel.ptx

%.ptx: %.cu
	$(NVCC) $(NVCC_FLAGS) -ptx -o $@ $<

# Generate SASS for detailed analysis
sass: matvec_kernel.sass matmul_kernel.sass

%.sass: %
	cuobjdump -sass $< > $@

# Profile with nvprof (legacy) or ncu (Nsight Compute)
profile-matvec: matvec_kernel
	@echo "=== Profiling Matrix-Vector Kernel ==="
	@if command -v ncu &> /dev/null; then \
		ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.per_cycle_active \
		./matvec_kernel; \
	else \
		nvprof --metrics achieved_occupancy,sm_efficiency,dram_utilization ./matvec_kernel; \
	fi

profile-matmul: matmul_kernel
	@echo "=== Profiling Matrix-Matrix Kernel ==="
	@if command -v ncu &> /dev/null; then \
		ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.per_cycle_active,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
		./matmul_kernel; \
	else \
		nvprof --metrics achieved_occupancy,sm_efficiency,shared_efficiency ./matmul_kernel; \
	fi

# Detailed occupancy analysis
occupancy: $(TARGETS)
	@echo "=== Occupancy Analysis ==="
	@for kernel in $(TARGETS); do \
		echo "\n--- $$kernel ---"; \
		ncu --query-metrics-mode=all -k regex:. --metrics \
			sm__warps_active.avg.pct_of_peak_sustained_active,\
			sm__maximum_warps_avg_per_active_cycle \
			./$$kernel 2>/dev/null || echo "Run on device with ncu"; \
	done

clean:
	rm -f $(TARGETS) *.ptx *.sass *.o

.PHONY: all clean ptx sass profile-matvec profile-matmul occupancy
