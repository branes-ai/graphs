#!/bin/bash
#
# Jetson Power Mode Manager
#
# Sets nvpmodel power mode and optionally locks clocks with jetson_clocks.
#
# Usage:
#   ./cli/jetson_power_mode.sh              # Show current mode and available modes
#   ./cli/jetson_power_mode.sh status       # Show current mode with clock details
#   ./cli/jetson_power_mode.sh 15w          # Set 15W mode
#   ./cli/jetson_power_mode.sh 30w          # Set 30W mode
#   ./cli/jetson_power_mode.sh 50w          # Set 50W mode
#   ./cli/jetson_power_mode.sh maxn         # Set MAXN mode (unconstrained)
#   ./cli/jetson_power_mode.sh maxn --lock  # Set MAXN and lock clocks to max
#
# Supported Jetson platforms:
#   AGX Orin 64GB: modes 0(MAXN), 1(15W), 2(30W), 3(50W)
#   AGX Orin 32GB: modes 0(MAXN), 1(15W), 2(30W), 3(40W)
#   Orin NX 16GB:  modes 0(MAXN), 1(10W), 2(15W), 3(25W)
#   Orin Nano 8GB: modes 0(MAXN), 1(7W), 2(15W)

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ---- helpers ----

die() { echo -e "${RED}Error: $1${NC}" >&2; exit 1; }

need_root() {
    if [ "$EUID" -ne 0 ]; then
        die "This operation requires root. Run with sudo or as root."
    fi
}

check_jetson() {
    if ! command -v nvpmodel &>/dev/null; then
        die "nvpmodel not found. This script is for NVIDIA Jetson platforms only."
    fi
}

# ---- query functions ----

get_current_mode() {
    sudo nvpmodel -q 2>/dev/null | grep -oP 'NV Power Mode:\s*\K\S+' || echo "unknown"
}

get_current_mode_id() {
    # Try POWER_MODEL_ID= format first, then bare number on its own line
    local output
    output=$(sudo nvpmodel -q 2>/dev/null)
    echo "$output" | grep -oP 'POWER_MODEL_ID=\K\d+' && return
    echo "$output" | grep -oP '^\s*\K\d+\s*$' | head -1 && return
    echo "?"
}

show_status() {
    echo ""
    echo -e "${BOLD}======================================${NC}"
    echo -e "${BOLD}  Jetson Power Mode Status${NC}"
    echo -e "${BOLD}======================================${NC}"
    echo ""

    # Current mode
    local mode_name mode_id
    mode_name=$(get_current_mode)
    mode_id=$(get_current_mode_id)
    echo -e "  Power Mode:  ${CYAN}${mode_name}${NC} (ID: ${mode_id})"

    # CPU info
    local cpu_cur cpu_min cpu_max gov
    cpu_cur=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo "?")
    cpu_min=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq 2>/dev/null || echo "?")
    cpu_max=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo "?")
    gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "?")

    if [ "$cpu_cur" != "?" ]; then
        echo -e "  CPU Freq:    $((cpu_cur / 1000)) MHz (range: $((cpu_min / 1000)) - $((cpu_max / 1000)) MHz)"
    fi
    echo -e "  CPU Governor: ${gov}"

    # Online CPUs
    local online
    online=$(cat /sys/devices/system/cpu/online 2>/dev/null || echo "?")
    echo -e "  CPUs Online: ${online}"

    # GPU info
    local gpu_cur gpu_min gpu_max
    for devfreq in /sys/devices/17000000.ga10b/devfreq/17000000.ga10b \
                   /sys/devices/gpu.0/devfreq/57000000.gpu; do
        if [ -d "$devfreq" ]; then
            gpu_cur=$(cat "${devfreq}/cur_freq" 2>/dev/null || echo "?")
            gpu_min=$(cat "${devfreq}/min_freq" 2>/dev/null || echo "?")
            gpu_max=$(cat "${devfreq}/max_freq" 2>/dev/null || echo "?")
            if [ "$gpu_cur" != "?" ]; then
                echo -e "  GPU Freq:    $((gpu_cur / 1000000)) MHz (range: $((gpu_min / 1000000)) - $((gpu_max / 1000000)) MHz)"
            fi
            break
        fi
    done

    # EMC (memory) info
    local emc_cur emc_max
    for emc in /sys/kernel/actmon_avg_activity/mc_all \
               /sys/devices/platform/bus@0/31b0000.emc/devfreq/31b0000.emc; do
        if [ -d "$emc" ]; then
            emc_cur=$(cat "${emc}/cur_freq" 2>/dev/null || echo "?")
            emc_max=$(cat "${emc}/max_freq" 2>/dev/null || echo "?")
            if [ "$emc_cur" != "?" ]; then
                echo -e "  EMC Freq:    $((emc_cur / 1000000)) MHz (max: $((emc_max / 1000000)) MHz)"
            fi
            break
        fi
    done

    # Thermal
    for tz in /sys/class/thermal/thermal_zone*/temp; do
        if [ -f "$tz" ]; then
            local temp zone_type
            temp=$(cat "$tz" 2>/dev/null || continue)
            zone_type=$(cat "$(dirname "$tz")/type" 2>/dev/null || echo "unknown")
            if [ "$zone_type" = "GPU-therm" ] || [ "$zone_type" = "gpu-therm" ]; then
                echo -e "  GPU Temp:    $((temp / 1000))C"
            fi
            if [ "$zone_type" = "CPU-therm" ] || [ "$zone_type" = "cpu-therm" ]; then
                echo -e "  CPU Temp:    $((temp / 1000))C"
            fi
        fi
    done

    echo ""
}

show_modes() {
    echo ""
    echo -e "${BOLD}Available Power Modes${NC}"
    echo -e "  (run 'sudo nvpmodel -p --verbose' for full details)"
    echo ""
    echo -e "  ${CYAN}15w${NC}    - Low power (fewer CPUs, lower clocks)"
    echo -e "  ${CYAN}30w${NC}    - Default (balanced)"
    echo -e "  ${CYAN}50w${NC}    - High performance (AGX 64GB) / 40w (AGX 32GB)"
    echo -e "  ${CYAN}maxn${NC}   - Maximum (unconstrained, all CPUs, max clocks)"
    echo ""
    echo -e "  Options:"
    echo -e "    ${YELLOW}--lock${NC}   Lock clocks to maximum after setting mode"
    echo -e "             (runs jetson_clocks; requires reboot to change mode after)"
    echo ""
}

# ---- mode mapping ----
# Map friendly names to nvpmodel IDs
# These cover AGX Orin 64GB/32GB. The nvpmodel IDs are:
#   0 = MAXN, 1 = 15W, 2 = 30W, 3 = 50W(64GB) or 40W(32GB)
# For Orin NX/Nano the IDs differ but the names still work via nvpmodel.

resolve_mode_id() {
    local mode="${1,,}"  # lowercase
    case "$mode" in
        maxn|0)    echo 0 ;;
        15w|1)     echo 1 ;;
        30w|2)     echo 2 ;;
        50w|40w|3) echo 3 ;;
        10w)       echo 1 ;;  # Orin NX
        25w)       echo 3 ;;  # Orin NX / Nano
        7w)        echo 1 ;;  # Orin Nano
        *)         die "Unknown power mode: $1. Use: 15w, 30w, 50w, maxn" ;;
    esac
}

# ---- set mode ----

set_mode() {
    local mode_name="$1"
    local lock_clocks="${2:-false}"
    local mode_id

    mode_id=$(resolve_mode_id "$mode_name")

    echo ""
    echo -e "${BOLD}Setting power mode...${NC}"
    echo -e "  Requested: ${CYAN}${mode_name}${NC} (nvpmodel ID: ${mode_id})"

    sudo nvpmodel -m "$mode_id"

    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}Power mode set successfully${NC}"
    else
        die "Failed to set power mode"
    fi

    if [ "$lock_clocks" = "true" ]; then
        echo ""
        echo -e "  ${YELLOW}Locking clocks to maximum (jetson_clocks)...${NC}"
        echo -e "  ${YELLOW}NOTE: Reboot required to change mode after locking${NC}"
        sudo jetson_clocks
        if [ $? -eq 0 ]; then
            echo -e "  ${GREEN}Clocks locked${NC}"
        else
            echo -e "  ${RED}Warning: jetson_clocks failed${NC}"
        fi
    fi

    # Show new status
    show_status
}

# ---- main ----

check_jetson

if [ $# -eq 0 ]; then
    show_status
    show_modes
    exit 0
fi

case "${1,,}" in
    status|info|show)
        show_status
        ;;
    help|-h|--help)
        echo "Usage: $0 [status | 15w | 30w | 50w | maxn] [--lock]"
        show_modes
        ;;
    *)
        lock=false
        if [ "${2:-}" = "--lock" ]; then
            lock=true
        fi
        set_mode "$1" "$lock"
        ;;
esac
