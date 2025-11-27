"""
Hardware Registry

Central registry for hardware profiles (specifications + calibrations).
Provides a unified interface for discovering, loading, and managing hardware data.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .config import RegistryConfig, get_config
from .profile import HardwareProfile
from ..calibration.schema import HardwareCalibration


# Global registry instance
_registry: Optional['HardwareRegistry'] = None


@dataclass
class DetectionResult:
    """Result of hardware auto-detection."""
    profile_id: str
    """ID of the matched profile."""

    confidence: float
    """Match confidence (0.0 to 1.0)."""

    detected_name: str
    """Name as detected from hardware."""

    profile: Optional[HardwareProfile] = None
    """The matched profile, if found."""


class HardwareRegistry:
    """
    Unified registry for hardware profiles.

    Provides:
    - Loading hardware specs and calibrations from a unified directory structure
    - Auto-detection and matching of current hardware
    - Simplified calibration workflow
    - Profile caching for performance

    Directory structure:
        registry_path/
        ├── cpu/
        │   └── i7_12700k/
        │       ├── spec.json
        │       └── calibrations/
        │           ├── performance_4900MHz_numpy.json
        │           └── performance_4900MHz_pytorch.json
        ├── gpu/
        │   └── h100_sxm5/
        │       ├── spec.json
        │       └── calibrations/
        │           └── default_1980MHz_pytorch.json
        └── boards/
            └── jetson_orin_nano/
                ├── spec.json
                └── calibrations/
                    ├── 7W_306MHz_numpy.json
                    ├── 15W_625MHz_numpy.json
                    └── MAXN_625MHz_pytorch.json

    Calibration filenames follow the pattern:
        {power_mode}_{frequency_mhz}MHz_{framework}.json
    """

    def __init__(self, config: Optional[RegistryConfig] = None):
        """
        Initialize the registry.

        Args:
            config: Registry configuration. If None, loads from default locations.
        """
        self.config = config or get_config()
        self._cache: Dict[str, HardwareProfile] = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        """Get the registry root path."""
        return self.config.registry_path

    def _ensure_loaded(self):
        """Ensure profiles are loaded into cache."""
        if not self._loaded:
            self.load_all()

    def load_all(self) -> int:
        """
        Load all profiles from the registry.

        Returns:
            Number of profiles loaded
        """
        self._cache.clear()

        if not self.path.exists():
            return 0

        count = 0
        # Scan device type directories (cpu, gpu, boards, etc.)
        for device_dir in self.path.iterdir():
            if not device_dir.is_dir():
                continue

            # Skip hidden directories and special files
            if device_dir.name.startswith('.'):
                continue

            # Scan hardware directories within each device type
            for hw_dir in device_dir.iterdir():
                if not hw_dir.is_dir():
                    continue

                # Check for spec.json
                spec_path = hw_dir / 'spec.json'
                if not spec_path.exists():
                    continue

                try:
                    profile = HardwareProfile.load(hw_dir)
                    self._cache[profile.id] = profile
                    count += 1
                except Exception as e:
                    print(f"Warning: Failed to load {hw_dir}: {e}")

        self._loaded = True
        return count

    def get(
        self,
        hardware_id: str,
        calibration_filter: Optional[Dict[str, Any]] = None
    ) -> Optional[HardwareProfile]:
        """
        Get a hardware profile by ID.

        Args:
            hardware_id: Profile ID (e.g., 'i7_12700k', 'h100_sxm5')
            calibration_filter: Optional filter for calibration selection.
                               Keys: 'power_mode', 'freq_mhz', 'framework'
                               If None, loads the most recent calibration.

        Returns:
            HardwareProfile if found, None otherwise
        """
        self._ensure_loaded()

        # If no filter, return cached profile
        if calibration_filter is None:
            return self._cache.get(hardware_id)

        # With filter, reload from disk with specific calibration
        profile = self._cache.get(hardware_id)
        if not profile:
            return None

        # Find the profile directory
        device_dir = self.path / profile.device_type / hardware_id
        if device_dir.exists():
            return HardwareProfile.load(device_dir, calibration_filter)

        return profile

    def list_all(self) -> List[str]:
        """
        List all available profile IDs.

        Returns:
            List of profile IDs
        """
        self._ensure_loaded()
        return sorted(self._cache.keys())

    def list_calibrations(self, hardware_id: str) -> List[Dict[str, Any]]:
        """
        List all available calibrations for a hardware profile.

        Args:
            hardware_id: Profile ID

        Returns:
            List of dicts with calibration info (power_mode, freq_mhz, framework, path)
        """
        self._ensure_loaded()
        profile = self._cache.get(hardware_id)
        if not profile:
            return []

        device_dir = self.path / profile.device_type / hardware_id
        return HardwareProfile.list_calibrations(device_dir)

    def list_by_type(self, device_type: str) -> List[str]:
        """
        List profile IDs filtered by device type.

        Args:
            device_type: 'cpu', 'gpu', 'tpu', etc.

        Returns:
            List of matching profile IDs
        """
        self._ensure_loaded()
        return sorted([
            pid for pid, profile in self._cache.items()
            if profile.device_type == device_type
        ])

    def search(self, query: str) -> List[HardwareProfile]:
        """
        Search for profiles matching a query string.

        Searches in: id, vendor, model, architecture, tags

        Args:
            query: Search string (case-insensitive)

        Returns:
            List of matching profiles
        """
        self._ensure_loaded()
        query_lower = query.lower()
        results = []

        for profile in self._cache.values():
            # Search in various fields
            searchable = [
                profile.id,
                profile.vendor,
                profile.model,
                profile.architecture or '',
                ' '.join(profile.tags),
            ]
            if any(query_lower in s.lower() for s in searchable):
                results.append(profile)

        return results

    def detect_hardware(self) -> DetectionResult:
        """
        Auto-detect current hardware and match to registry.

        For ARM/embedded devices, uses board detection (device-tree) for more
        accurate matching since CPU model names are often generic (e.g., "ARMv8").

        Returns:
            DetectionResult with matched profile information
        """
        from ..database import HardwareDetector

        self._ensure_loaded()

        detector = HardwareDetector()
        detection = detector.detect_all()

        # For ARM devices, try board detection first (more reliable than CPU name)
        board = detection.get('board')
        if board and board.family:
            board_match = self._match_board_to_profile(board, detection.get('cpu'))
            if board_match:
                return board_match

        # Try to match CPU
        if detection.get('cpu'):
            cpu = detection['cpu']
            detected_name = cpu.model_name

            # Collect all matching profiles with confidence scores
            matches = []
            for profile in self._cache.values():
                if profile.device_type == 'cpu':
                    confidence = self._calculate_match_confidence(profile, cpu)
                    if confidence > 0:
                        matches.append((confidence, profile))

            # Return best match if any
            if matches:
                matches.sort(key=lambda x: x[0], reverse=True)
                best_confidence, best_profile = matches[0]
                return DetectionResult(
                    profile_id=best_profile.id,
                    confidence=best_confidence,
                    detected_name=detected_name,
                    profile=best_profile
                )

            # Return partial result with no match
            return DetectionResult(
                profile_id='',
                confidence=0.0,
                detected_name=detected_name,
                profile=None
            )

        # Try to match GPU
        if detection.get('gpus'):
            gpu = detection['gpus'][0]
            detected_name = gpu.model_name

            for profile in self._cache.values():
                if profile.device_type == 'gpu':
                    if self._matches_hardware(profile, gpu.model_name, gpu.vendor):
                        return DetectionResult(
                            profile_id=profile.id,
                            confidence=0.9,
                            detected_name=detected_name,
                            profile=profile
                        )

            return DetectionResult(
                profile_id='',
                confidence=0.0,
                detected_name=detected_name,
                profile=None
            )

        return DetectionResult(
            profile_id='',
            confidence=0.0,
            detected_name='Unknown',
            profile=None
        )

    def _match_board_to_profile(self, board, cpu) -> Optional[DetectionResult]:
        """
        Match detected board to a CPU profile in the registry.

        Uses device-tree model string to identify specific hardware like Jetson.

        Args:
            board: DetectedBoard from hardware detection
            cpu: DetectedCPU (optional, for additional context)

        Returns:
            DetectionResult if a match is found, None otherwise
        """
        device_tree_model = board.device_tree_model or board.model or ""
        dt_lower = device_tree_model.lower()

        # Build detected name for reporting
        detected_name = device_tree_model or f"{board.vendor} {board.family or 'Board'}"

        # Jetson matching based on device tree
        if board.vendor == "NVIDIA" and board.family == "Jetson":
            # Match specific Jetson models
            jetson_profiles = {
                # Orin family
                'orin nano': 'jetson_orin_nano_cpu',
                'orin nx': 'jetson_orin_nx_cpu',
                'agx orin': 'jetson_orin_agx_cpu',
                'orin agx': 'jetson_orin_agx_cpu',
                # Xavier family
                'xavier nx': 'jetson_xavier_nx_cpu',
                'agx xavier': 'jetson_xavier_agx_cpu',
                'xavier agx': 'jetson_xavier_agx_cpu',
                # Older models
                'nano': 'jetson_nano_cpu',
                'tx2': 'jetson_tx2_cpu',
                'tx1': 'jetson_tx1_cpu',
            }

            for pattern, profile_id in jetson_profiles.items():
                if pattern in dt_lower:
                    profile = self._cache.get(profile_id)
                    if profile:
                        return DetectionResult(
                            profile_id=profile_id,
                            confidence=0.95,  # High confidence from device-tree
                            detected_name=detected_name,
                            profile=profile
                        )

            # Fallback: try to match any Jetson CPU profile by SoC
            if board.soc:
                soc_lower = board.soc.lower()
                for profile in self._cache.values():
                    if profile.device_type == 'cpu' and 'jetson' in profile.id.lower():
                        # Check if profile notes mention the SoC
                        notes = getattr(profile, 'notes', '') or ''
                        if soc_lower in notes.lower():
                            return DetectionResult(
                                profile_id=profile.id,
                                confidence=0.85,
                                detected_name=detected_name,
                                profile=profile
                            )

        # Raspberry Pi matching
        elif board.vendor == "Raspberry Pi Foundation":
            pi_profiles = {
                'raspberry pi 5': 'raspberry_pi_5_cpu',
                'raspberry pi 4': 'raspberry_pi_4_cpu',
                'raspberry pi 3': 'raspberry_pi_3_cpu',
            }
            for pattern, profile_id in pi_profiles.items():
                if pattern in dt_lower:
                    profile = self._cache.get(profile_id)
                    if profile:
                        return DetectionResult(
                            profile_id=profile_id,
                            confidence=0.95,
                            detected_name=detected_name,
                            profile=profile
                        )

        return None

    def _calculate_match_confidence(self, profile: HardwareProfile, cpu) -> float:
        """
        Calculate match confidence between a profile and detected CPU.

        Returns a confidence score from 0.0 to 1.0, where:
        - 0.0 = no match
        - 0.5 = partial match (architecture only)
        - 0.7 = good match (architecture + core count)
        - 0.9 = excellent match (name/ID match)
        """
        detected_lower = cpu.model_name.lower()
        profile_id_lower = profile.id.lower()
        profile_model_lower = profile.model.lower()

        # Check if profile ID appears in detected name (best match)
        # e.g., 'i7_12700k' in '12th Gen Intel(R) Core(TM) i7-12700K'
        id_parts = profile_id_lower.replace('_', ' ').replace('-', ' ').split()
        if all(part in detected_lower.replace('-', ' ').replace('_', ' ') for part in id_parts if len(part) > 2):
            return 0.95

        # Check model name match
        model_parts = profile_model_lower.replace('_', ' ').replace('-', ' ').split()
        if all(part in detected_lower.replace('-', ' ').replace('_', ' ') for part in model_parts if len(part) > 2):
            return 0.9

        # Architecture-based matching (for ARM/embedded devices)
        if hasattr(profile, 'architecture') and profile.architecture:
            arch_lower = profile.architecture.lower()

            # ARM Cortex matching with core count
            is_arm_match = False
            if 'cortex-a78' in arch_lower and 'armv8' in detected_lower:
                is_arm_match = True
            elif 'cortex-a72' in arch_lower and 'armv8' in detected_lower:
                is_arm_match = True
            elif 'cortex-a57' in arch_lower and 'armv8' in detected_lower:
                is_arm_match = True

            if is_arm_match:
                # Check core count for disambiguation
                profile_cores = getattr(profile, 'compute_units', None)
                detected_cores = getattr(cpu, 'cores', None)

                if profile_cores and detected_cores and profile_cores == detected_cores:
                    # Architecture + core count match
                    return 0.8
                elif profile_cores and detected_cores:
                    # Architecture match but core count mismatch - lower confidence
                    return 0.3
                else:
                    # Architecture match only
                    return 0.5

        return 0.0

    def _matches_hardware(self, profile: HardwareProfile, detected_name: str, vendor: str) -> bool:
        """Check if a profile matches detected hardware (legacy method for GPU matching)."""
        detected_lower = detected_name.lower()
        profile_id_lower = profile.id.lower()
        profile_model_lower = profile.model.lower()

        # Check if profile ID appears in detected name
        # e.g., 'i7_12700k' in '12th Gen Intel(R) Core(TM) i7-12700K'
        id_parts = profile_id_lower.replace('_', ' ').replace('-', ' ').split()
        if all(part in detected_lower.replace('-', ' ').replace('_', ' ') for part in id_parts if len(part) > 2):
            return True

        # Check model name
        model_parts = profile_model_lower.replace('_', ' ').replace('-', ' ').split()
        if all(part in detected_lower.replace('-', ' ').replace('_', ' ') for part in model_parts if len(part) > 2):
            return True

        return False

    def save_profile(self, profile: HardwareProfile):
        """
        Save a profile to the registry.

        Args:
            profile: Profile to save
        """
        # Determine directory based on device type
        device_dir = self.path / profile.device_type / profile.id
        profile.save(device_dir)

        # Update cache
        self._cache[profile.id] = profile

    def calibrate(
        self,
        hardware_id: str,
        quick: bool = False,
        operations: Optional[List[str]] = None,
        framework: Optional[str] = None,
        force: bool = False,
    ) -> HardwareProfile:
        """
        Calibrate hardware and save results to registry.

        Args:
            hardware_id: Profile ID to calibrate
            quick: Run quick calibration with fewer trials
            operations: Operations to calibrate (default: blas, stream)
            framework: Framework override ('numpy' or 'pytorch')
            force: Force calibration even if pre-flight checks fail

        Returns:
            Updated HardwareProfile with calibration data

        Raises:
            ValueError: If hardware_id not found in registry
            RuntimeError: If pre-flight checks fail and force=False
        """
        from ..calibration.calibrator import calibrate_hardware

        profile = self.get(hardware_id)
        if not profile:
            raise ValueError(f"Hardware '{hardware_id}' not found in registry")

        # Determine device type
        device = 'cuda' if profile.device_type == 'gpu' else 'cpu'

        # Get theoretical peaks
        theoretical_peaks = profile.theoretical_peaks
        peak_gflops = theoretical_peaks.get('fp32', max(theoretical_peaks.values()) if theoretical_peaks else 100.0)

        # Run calibration
        calibration = calibrate_hardware(
            hardware_name=profile.model,
            theoretical_peak_gflops=peak_gflops,
            theoretical_bandwidth_gbps=profile.peak_bandwidth_gbps,
            theoretical_peaks=theoretical_peaks,
            device=device,
            framework=framework or self.config.default_framework,
            operations=operations,
            quick=quick,
            force=force,
        )

        # Update profile with calibration
        profile.calibration = calibration
        profile.calibration_date = calibration.metadata.calibration_date

        # Save to registry
        self.save_profile(profile)

        return profile

    def detect_and_calibrate(
        self,
        quick: bool = False,
        operations: Optional[List[str]] = None,
        framework: Optional[str] = None,
        create_if_missing: bool = True,
        force: bool = False,
    ) -> HardwareProfile:
        """
        Auto-detect current hardware, calibrate it, and save to registry.

        This is the simplified one-command workflow.

        Args:
            quick: Run quick calibration
            operations: Operations to calibrate
            framework: Framework override
            create_if_missing: If True, create new profile for unknown hardware
            force: Force calibration even if pre-flight checks fail

        Returns:
            Calibrated HardwareProfile

        Raises:
            RuntimeError: If hardware not detected or not in registry
            RuntimeError: If pre-flight checks fail and force=False
        """
        from ..calibration.calibrator import calibrate_hardware
        from ..database import HardwareDetector

        # Detect hardware
        detection = self.detect_hardware()

        if detection.profile:
            # Found in registry - calibrate existing profile
            return self.calibrate(
                detection.profile_id,
                quick=quick,
                operations=operations,
                framework=framework,
                force=force,
            )

        if not create_if_missing:
            raise RuntimeError(
                f"Hardware '{detection.detected_name}' not found in registry.\n"
                f"Add it first or use create_if_missing=True"
            )

        # Create new profile for unknown hardware
        print(f"Creating new profile for: {detection.detected_name}")
        profile = self._create_profile_from_detection(detection)

        # Determine device type
        device = 'cuda' if profile.device_type == 'gpu' else 'cpu'

        # Run calibration
        calibration = calibrate_hardware(
            hardware_name=profile.model,
            theoretical_peak_gflops=profile.theoretical_peaks.get('fp32', 100.0),
            theoretical_bandwidth_gbps=profile.peak_bandwidth_gbps,
            theoretical_peaks=profile.theoretical_peaks,
            device=device,
            framework=framework or self.config.default_framework,
            operations=operations,
            quick=quick,
            force=force,
        )

        # Update and save profile
        profile.calibration = calibration
        profile.calibration_date = calibration.metadata.calibration_date
        self.save_profile(profile)

        return profile

    def _create_profile_from_detection(self, detection: DetectionResult) -> HardwareProfile:
        """Create a new profile from hardware detection."""
        from ..database import HardwareDetector

        detector = HardwareDetector()
        hw_info = detector.detect_all()

        # Generate a clean ID from detected name
        clean_id = self._generate_id(detection.detected_name)

        if hw_info.get('cpu'):
            cpu = hw_info['cpu']
            return HardwareProfile(
                id=clean_id,
                vendor=cpu.vendor or 'Unknown',
                model=cpu.model_name,
                device_type='cpu',
                theoretical_peaks={'fp32': 100.0},  # Placeholder, will be updated by calibration
                peak_bandwidth_gbps=50.0,  # Placeholder
                compute_units=cpu.cores,
            )

        if hw_info.get('gpus'):
            gpu = hw_info['gpus'][0]
            return HardwareProfile(
                id=clean_id,
                vendor=gpu.vendor or 'Unknown',
                model=gpu.model_name,
                device_type='gpu',
                theoretical_peaks={'fp32': 1000.0},  # Placeholder
                peak_bandwidth_gbps=500.0,  # Placeholder
                memory_gb=gpu.memory_gb,
            )

        raise RuntimeError("Could not detect CPU or GPU")

    def _generate_id(self, name: str) -> str:
        """Generate a clean ID from a hardware name."""
        import re
        # Remove special characters, convert to lowercase, replace spaces with underscores
        clean = re.sub(r'[^\w\s-]', '', name.lower())
        clean = re.sub(r'[-\s]+', '_', clean)
        # Remove common filler words
        for word in ['gen', 'intel', 'amd', 'nvidia', 'core', 'processor']:
            clean = clean.replace(f'{word}_', '')
        # Collapse multiple underscores
        clean = re.sub(r'_+', '_', clean).strip('_')
        return clean[:50]  # Limit length


def get_registry(config: Optional[RegistryConfig] = None) -> HardwareRegistry:
    """
    Get the global registry instance.

    Args:
        config: Optional configuration override

    Returns:
        HardwareRegistry instance
    """
    global _registry

    if config is not None:
        # Create new registry with provided config
        _registry = HardwareRegistry(config)
    elif _registry is None:
        # Create default registry
        _registry = HardwareRegistry()

    return _registry


def reset_registry():
    """Reset the global registry instance (mainly for testing)."""
    global _registry
    _registry = None
