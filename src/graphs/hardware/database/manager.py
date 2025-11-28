"""
Hardware Database Manager

Provides CRUD operations and query capabilities for the hardware database.
The database is file-based (JSON files) organized by vendor/type.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from collections import defaultdict

from .schema import HardwareSpec


class HardwareDatabase:
    """
    Manage hardware database with CRUD operations and queries.

    The database is stored as JSON files in a directory tree:
        hardware_database/
        ├── cpu/
        │   ├── intel/
        │   │   └── i7_12700k.json
        │   └── amd/
        │       └── ryzen_9_7950x.json
        └── gpu/
            └── nvidia/
                └── h100_sxm5_80gb.json
    """

    def __init__(self, db_root: Path):
        """
        Initialize hardware database.

        Args:
            db_root: Path to hardware_database/ directory
        """
        self.db_root = Path(db_root)
        self._cache: Dict[str, HardwareSpec] = {}
        self._loaded = False

    @staticmethod
    def _get_system_field(spec: HardwareSpec, field: str) -> Optional[Any]:
        """
        Safely extract a field from system block with fallback to deprecated fields.

        Args:
            spec: HardwareSpec instance
            field: Field name (e.g., 'vendor', 'device_type', 'platform')

        Returns:
            Field value or None if not found
        """
        # Try new consolidated system block first
        if spec.system and field in spec.system:
            return spec.system[field]

        # Fall back to deprecated top-level field for backward compatibility
        return getattr(spec, field, None)

    def load_all(self, force_reload: bool = False) -> Dict[str, HardwareSpec]:
        """
        Load all hardware specs into cache.

        Args:
            force_reload: If True, reload even if already loaded

        Returns:
            Dictionary of {hardware_id: HardwareSpec}
        """
        if self._loaded and not force_reload:
            return self._cache

        self._cache.clear()

        # Walk through all JSON files in database
        if not self.db_root.exists():
            print(f"Warning: Database root does not exist: {self.db_root}")
            return self._cache

        for json_file in self.db_root.rglob("*.json"):
            # Skip schema.json (validation schema, not hardware spec)
            if json_file.name == "schema.json":
                continue

            # Skip boards/ directory (board specs are loaded separately)
            if "boards" in json_file.parts:
                continue

            try:
                spec = HardwareSpec.from_json(json_file)

                # Validate
                errors = spec.validate()
                if errors:
                    print(f"[!] Validation errors in {json_file}:")
                    for error in errors:
                        print(f"  - {error}")
                    continue

                # Add to cache
                self._cache[spec.id] = spec

            except Exception as e:
                print(f"[!] Error loading {json_file}: {e}")
                continue

        self._loaded = True
        print(f"Loaded {len(self._cache)} hardware specs from database")
        return self._cache

    def get(self, hardware_id: str) -> Optional[HardwareSpec]:
        """
        Get hardware spec by ID.

        Args:
            hardware_id: Hardware ID (e.g., 'intel_i7_12700k')

        Returns:
            HardwareSpec if found, None otherwise
        """
        if not self._loaded:
            self.load_all()

        return self._cache.get(hardware_id)

    def search(self, **filters) -> List[HardwareSpec]:
        """
        Search hardware specs with filters.

        Args:
            **filters: Field name -> value or callable
                Examples:
                - vendor="Intel"
                - device_type="cpu"
                - platform="x86_64"
                - cores=lambda c: c >= 8
                - has_feature="Tensor Cores"

        Returns:
            List of matching HardwareSpec objects
        """
        if not self._loaded:
            self.load_all()

        results = []

        for spec in self._cache.values():
            match = True

            for field, value in filters.items():
                # Special case: has_feature (check in special_features list)
                if field == "has_feature":
                    if value not in spec.special_features:
                        match = False
                        break
                # Special case: has_isa (check in isa_extensions list)
                elif field == "has_isa":
                    if value not in spec.isa_extensions:
                        match = False
                        break
                # Regular field comparison
                elif hasattr(spec, field):
                    spec_value = getattr(spec, field)

                    # Callable filter (e.g., lambda x: x >= 8)
                    if callable(value):
                        if spec_value is None or not value(spec_value):
                            match = False
                            break
                    # Direct equality
                    else:
                        if spec_value != value:
                            match = False
                            break
                else:
                    # Unknown field
                    match = False
                    break

            if match:
                results.append(spec)

        return results

    def list_by_category(self) -> Dict[str, Dict[str, List[str]]]:
        """
        List hardware grouped by device type and vendor.

        Returns:
            Nested dict: {device_type: {vendor: [hardware_ids]}}

        Example:
            {
                "cpu": {
                    "Intel": ["intel_i7_12700k", "intel_i9_13900k"],
                    "AMD": ["amd_ryzen_9_7950x"]
                },
                "gpu": {
                    "NVIDIA": ["nvidia_h100_sxm5_80gb", "nvidia_a100_sxm4_80gb"]
                }
            }
        """
        if not self._loaded:
            self.load_all()

        categories = defaultdict(lambda: defaultdict(list))

        for spec in self._cache.values():
            device_type = self._get_system_field(spec, 'device_type')
            vendor = self._get_system_field(spec, 'vendor')
            if device_type and vendor:
                categories[device_type][vendor].append(spec.id)

        # Convert to regular dict for serialization
        return {
            device_type: dict(vendors)
            for device_type, vendors in categories.items()
        }

    def add(self, spec: HardwareSpec, overwrite: bool = False) -> bool:
        """
        Add new hardware spec to database.

        Args:
            spec: HardwareSpec to add
            overwrite: If True, overwrite existing entry with same ID

        Returns:
            True if added successfully, False otherwise
        """
        # Validate first
        errors = spec.validate()
        if errors:
            print(f"Cannot add hardware '{spec.id}': validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        # Check if exists
        if not overwrite and spec.id in self._cache:
            print(f"Hardware '{spec.id}' already exists. Use overwrite=True to replace.")
            return False

        # Determine file path based on device_type and vendor
        # e.g., cpu/intel/i7_12700k.json
        device_type = self._get_system_field(spec, 'device_type')
        vendor = self._get_system_field(spec, 'vendor')

        if not device_type or not vendor:
            print(f"[X] Error: Hardware spec missing device_type or vendor")
            return False

        vendor_dir = self.db_root / device_type / vendor.lower().replace(' ', '_')
        vendor_dir.mkdir(parents=True, exist_ok=True)

        filepath = vendor_dir / f"{spec.id}.json"

        # Save to disk
        try:
            spec.to_json(filepath)
            # Update cache
            self._cache[spec.id] = spec
            print(f"[OK] Added hardware: {spec.id} -> {filepath}")
            return True
        except Exception as e:
            print(f"[X] Error saving hardware '{spec.id}': {e}")
            return False

    def update(self, hardware_id: str, **updates) -> bool:
        """
        Update existing hardware spec.

        Args:
            hardware_id: Hardware ID to update
            **updates: Fields to update

        Returns:
            True if updated successfully, False otherwise
        """
        if not self._loaded:
            self.load_all()

        spec = self._cache.get(hardware_id)
        if not spec:
            print(f"Hardware '{hardware_id}' not found")
            return False

        # Apply updates
        for field, value in updates.items():
            if hasattr(spec, field):
                setattr(spec, field, value)
            else:
                print(f"Warning: Unknown field '{field}', skipping")

        # Validate after update
        errors = spec.validate()
        if errors:
            print(f"Update created validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        # Find file path
        filepath = self._find_spec_file(hardware_id)
        if not filepath:
            print(f"Could not find file for hardware '{hardware_id}'")
            return False

        # Save
        try:
            spec.to_json(filepath)
            print(f"[OK] Updated hardware: {hardware_id}")
            return True
        except Exception as e:
            print(f"[X] Error updating hardware '{hardware_id}': {e}")
            return False

    def delete(self, hardware_id: str) -> bool:
        """
        Delete hardware spec from database.

        Args:
            hardware_id: Hardware ID to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        if not self._loaded:
            self.load_all()

        if hardware_id not in self._cache:
            print(f"Hardware '{hardware_id}' not found")
            return False

        # Find and delete file
        filepath = self._find_spec_file(hardware_id)
        if not filepath:
            print(f"Could not find file for hardware '{hardware_id}'")
            return False

        try:
            filepath.unlink()
            del self._cache[hardware_id]
            print(f"[OK] Deleted hardware: {hardware_id}")
            return True
        except Exception as e:
            print(f"[X] Error deleting hardware '{hardware_id}': {e}")
            return False

    def _find_spec_file(self, hardware_id: str) -> Optional[Path]:
        """
        Find JSON file path for given hardware ID.

        Args:
            hardware_id: Hardware ID

        Returns:
            Path to JSON file, or None if not found
        """
        for json_file in self.db_root.rglob(f"{hardware_id}.json"):
            return json_file
        return None

    def validate_all(self) -> Dict[str, List[str]]:
        """
        Validate all hardware specs in database.

        Returns:
            Dictionary of {hardware_id: [error_messages]} for specs with errors
        """
        if not self._loaded:
            self.load_all()

        invalid_specs = {}

        for hardware_id, spec in self._cache.items():
            errors = spec.validate()
            if errors:
                invalid_specs[hardware_id] = errors

        return invalid_specs

    def load_boards(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all board/SoC definitions from the boards/ directory.

        Board definitions are separate from hardware specs and define
        embedded systems that contain multiple components (CPU + GPU).

        Returns:
            Dictionary of {board_id: board_spec_dict}
        """
        boards = {}
        boards_dir = self.db_root / "boards"

        if not boards_dir.exists():
            return boards

        for json_file in boards_dir.rglob("*.json"):
            try:
                with open(json_file) as f:
                    spec = json.load(f)

                board_id = spec.get('id')
                if board_id:
                    boards[board_id] = spec
            except Exception as e:
                print(f"[!] Error loading board {json_file}: {e}")
                continue

        return boards

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with counts and breakdowns
        """
        if not self._loaded:
            self.load_all()

        stats = {
            'total_count': len(self._cache),
            'by_device_type': defaultdict(int),
            'by_vendor': defaultdict(int),
            'by_platform': defaultdict(int),
            'by_os': defaultdict(int),
        }

        for spec in self._cache.values():
            device_type = self._get_system_field(spec, 'device_type')
            vendor = self._get_system_field(spec, 'vendor')
            platform = self._get_system_field(spec, 'platform')
            os_compatibility = self._get_system_field(spec, 'os_compatibility')

            if device_type:
                stats['by_device_type'][device_type] += 1
            if vendor:
                stats['by_vendor'][vendor] += 1
            if platform:
                stats['by_platform'][platform] += 1
            if os_compatibility:
                for os_type in os_compatibility:
                    stats['by_os'][os_type] += 1

        # Convert defaultdicts to regular dicts
        stats['by_device_type'] = dict(stats['by_device_type'])
        stats['by_vendor'] = dict(stats['by_vendor'])
        stats['by_platform'] = dict(stats['by_platform'])
        stats['by_os'] = dict(stats['by_os'])

        return stats


# Convenience function for getting database instance
_db_instance = None


def get_database(db_root: Optional[Path] = None) -> HardwareDatabase:
    """
    Get singleton hardware database instance.

    Args:
        db_root: Path to database root (only used on first call)

    Returns:
        HardwareDatabase instance
    """
    global _db_instance

    if _db_instance is None:
        if db_root is None:
            # Default location relative to this file
            db_root = Path(__file__).parent.parent.parent.parent.parent / "hardware_database"
        _db_instance = HardwareDatabase(db_root)
        _db_instance.load_all()

    return _db_instance
