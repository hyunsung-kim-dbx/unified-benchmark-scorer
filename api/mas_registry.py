"""
MAS (Multi-Agent Supervisor) endpoint registry.

Manages available MAS endpoints for benchmark testing.
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path

import httpx


@dataclass
class MASEndpoint:
    """Information about a MAS endpoint."""
    name: str
    url: str
    description: Optional[str] = None
    supports_benchmark_mode: bool = True
    is_active: bool = True


class MASRegistry:
    """
    Registry for managing MAS endpoints.

    Provides functionality to add, list, validate, and health-check MAS endpoints.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MAS registry.

        Args:
            config_path: Optional path to JSON config file for persistence
        """
        self.config_path = Path(config_path) if config_path else None
        self._endpoints: Dict[str, MASEndpoint] = {}

        if self.config_path and self.config_path.exists():
            self._load_config()

    def _load_config(self):
        """Load endpoints from config file."""
        try:
            with open(self.config_path) as f:
                data = json.load(f)
                for name, info in data.get("endpoints", {}).items():
                    self._endpoints[name] = MASEndpoint(
                        name=name,
                        url=info.get("url", ""),
                        description=info.get("description"),
                        supports_benchmark_mode=info.get("supports_benchmark_mode", True),
                        is_active=info.get("is_active", True)
                    )
        except Exception:
            pass

    def _save_config(self):
        """Save endpoints to config file."""
        if not self.config_path:
            return

        try:
            data = {
                "endpoints": {
                    name: asdict(endpoint)
                    for name, endpoint in self._endpoints.items()
                }
            }
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def add_endpoint(self, endpoint: MASEndpoint) -> bool:
        """
        Add or update an endpoint.

        Args:
            endpoint: MASEndpoint to add

        Returns:
            True if successful
        """
        self._endpoints[endpoint.name] = endpoint
        self._save_config()
        return True

    def remove_endpoint(self, name: str) -> bool:
        """
        Remove an endpoint by name.

        Args:
            name: Endpoint name

        Returns:
            True if removed, False if not found
        """
        if name in self._endpoints:
            del self._endpoints[name]
            self._save_config()
            return True
        return False

    def list_endpoints(self, active_only: bool = False) -> List[MASEndpoint]:
        """
        List all registered endpoints.

        Args:
            active_only: If True, only return active endpoints

        Returns:
            List of MASEndpoint objects
        """
        endpoints = list(self._endpoints.values())
        if active_only:
            endpoints = [e for e in endpoints if e.is_active]
        return endpoints

    def get_endpoint(self, name: str) -> Optional[MASEndpoint]:
        """
        Get endpoint by name.

        Args:
            name: Endpoint name

        Returns:
            MASEndpoint or None
        """
        return self._endpoints.get(name)

    def health_check(self, endpoint: MASEndpoint, timeout: int = 10) -> Dict:
        """
        Check if an endpoint is healthy.

        Args:
            endpoint: MASEndpoint to check
            timeout: Request timeout in seconds

        Returns:
            {"healthy": bool, "latency_ms": float, "error": str | None}
        """
        import time

        try:
            start = time.time()

            # Try health endpoint first
            response = httpx.get(
                f"{endpoint.url.rstrip('/')}/health",
                timeout=timeout
            )

            latency_ms = (time.time() - start) * 1000

            if response.status_code == 200:
                return {
                    "healthy": True,
                    "latency_ms": latency_ms,
                    "error": None,
                    "supports_benchmark_mode": self._check_benchmark_mode(endpoint)
                }
            else:
                return {
                    "healthy": False,
                    "latency_ms": latency_ms,
                    "error": f"HTTP {response.status_code}"
                }

        except httpx.ConnectError:
            return {
                "healthy": False,
                "latency_ms": 0,
                "error": "Connection refused"
            }
        except httpx.TimeoutException:
            return {
                "healthy": False,
                "latency_ms": timeout * 1000,
                "error": "Timeout"
            }
        except Exception as e:
            return {
                "healthy": False,
                "latency_ms": 0,
                "error": str(e)
            }

    def _check_benchmark_mode(self, endpoint: MASEndpoint) -> bool:
        """Check if endpoint supports benchmark mode."""
        try:
            # Try to get API info
            response = httpx.get(
                f"{endpoint.url.rstrip('/')}/info",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("supports_benchmark_mode", True)
        except Exception:
            pass
        return endpoint.supports_benchmark_mode

    def health_check_all(self, timeout: int = 10) -> Dict[str, Dict]:
        """
        Health check all registered endpoints.

        Args:
            timeout: Request timeout per endpoint

        Returns:
            Dict mapping endpoint names to health status
        """
        results = {}
        for name, endpoint in self._endpoints.items():
            results[name] = self.health_check(endpoint, timeout)
        return results
