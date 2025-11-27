"""
Registry Configuration

Manages configuration for the hardware registry, including storage paths
and backend selection.

Configuration is loaded from (in order of precedence):
1. Environment variables (GRAPHS_REGISTRY_PATH)
2. User config file (~/.config/graphs/config.json)
3. Project config file (.graphs/config.json in current directory)
4. Default paths
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class RegistryConfig:
    """Configuration for the hardware registry."""

    # Path to the hardware registry directory
    registry_path: Path = field(default_factory=lambda: Path())
    """Root directory containing hardware profiles."""

    # Backend type (for future extensibility)
    backend: str = "local"
    """Storage backend: 'local' (filesystem) or future options."""

    # Whether to auto-calibrate on first access
    auto_calibrate: bool = False
    """If True, automatically calibrate hardware when accessed without calibration."""

    # Default framework preference
    default_framework: Optional[str] = None
    """Override framework selection: 'numpy', 'pytorch', or None (auto)."""

    def __post_init__(self):
        # Ensure registry_path is a Path object
        if isinstance(self.registry_path, str):
            self.registry_path = Path(self.registry_path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['registry_path'] = str(self.registry_path)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegistryConfig':
        """Create from dictionary."""
        if 'registry_path' in data:
            data['registry_path'] = Path(data['registry_path'])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _find_project_root() -> Optional[Path]:
    """Find the project root by looking for pyproject.toml or setup.py."""
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / 'pyproject.toml').exists() or (parent / 'setup.py').exists():
            return parent
    return None


def _get_default_registry_path() -> Path:
    """Get the default registry path."""
    # Check for project-local registry first
    project_root = _find_project_root()
    if project_root:
        local_registry = project_root / 'hardware_registry'
        if local_registry.exists():
            return local_registry

    # Fall back to package-bundled registry
    package_dir = Path(__file__).parent.parent.parent.parent  # src/graphs/hardware/registry -> src
    bundled_registry = package_dir.parent / 'hardware_registry'
    if bundled_registry.exists():
        return bundled_registry

    # Last resort: user data directory
    if os.name == 'nt':  # Windows
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    elif os.name == 'darwin':  # macOS
        base = Path.home() / 'Library' / 'Application Support'
    else:  # Linux/Unix
        base = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))

    return base / 'graphs' / 'hardware_registry'


def _load_config_file(path: Path) -> Optional[Dict[str, Any]]:
    """Load configuration from a JSON file."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def get_config() -> RegistryConfig:
    """
    Get the registry configuration.

    Loads configuration from environment variables and config files,
    with sensible defaults.

    Returns:
        RegistryConfig instance
    """
    config_data = {}

    # 1. Start with defaults
    default_path = _get_default_registry_path()
    config_data['registry_path'] = default_path

    # 2. Load project config (.graphs/config.json)
    project_root = _find_project_root()
    if project_root:
        project_config = _load_config_file(project_root / '.graphs' / 'config.json')
        if project_config:
            config_data.update(project_config)

    # 3. Load user config (~/.config/graphs/config.json)
    if os.name == 'nt':
        user_config_dir = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
    else:
        user_config_dir = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))

    user_config = _load_config_file(user_config_dir / 'graphs' / 'config.json')
    if user_config:
        config_data.update(user_config)

    # 4. Environment variables (highest precedence)
    env_registry_path = os.environ.get('GRAPHS_REGISTRY_PATH')
    if env_registry_path:
        config_data['registry_path'] = env_registry_path

    env_backend = os.environ.get('GRAPHS_REGISTRY_BACKEND')
    if env_backend:
        config_data['backend'] = env_backend

    return RegistryConfig.from_dict(config_data)


def save_config(config: RegistryConfig, path: Optional[Path] = None):
    """
    Save configuration to a file.

    Args:
        config: Configuration to save
        path: Path to save to (default: user config directory)
    """
    if path is None:
        if os.name == 'nt':
            config_dir = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        else:
            config_dir = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
        path = config_dir / 'graphs' / 'config.json'

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
