"""
Databricks Workspace client for listing resources.

Provides functionality to discover Genie Spaces and other workspace resources.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import httpx
from databricks.sdk import WorkspaceClient


@dataclass
class GenieSpaceInfo:
    """Information about a Genie Space."""
    space_id: str
    name: str
    description: Optional[str] = None
    created_at: Optional[str] = None
    warehouse_id: Optional[str] = None


class WorkspaceResourceClient:
    """
    Client for discovering Databricks workspace resources.

    Provides methods to list Genie Spaces, SQL Warehouses, and other resources.
    """

    def __init__(self, host: str, token: str):
        """
        Initialize workspace client.

        Args:
            host: Databricks workspace host
            token: Databricks access token
        """
        self.host = host.rstrip("/")
        self.token = token

        self._http_client = httpx.Client(
            base_url=self.host,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )

        self._sdk_client = WorkspaceClient(
            host=host,
            token=token
        )

    def list_genie_spaces(self) -> List[GenieSpaceInfo]:
        """
        List all accessible Genie Spaces.

        Returns:
            List of GenieSpaceInfo objects
        """
        try:
            response = self._http_client.get("/api/2.0/genie/spaces")
            response.raise_for_status()

            data = response.json()
            spaces = []

            for space in data.get("spaces", []):
                spaces.append(GenieSpaceInfo(
                    space_id=space.get("space_id", ""),
                    name=space.get("title", space.get("name", "Unnamed")),
                    description=space.get("description"),
                    created_at=space.get("created_at"),
                    warehouse_id=space.get("warehouse_id")
                ))

            return spaces

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Genie API may not be available
                return []
            raise
        except Exception:
            return []

    def list_sql_warehouses(self) -> List[Dict[str, Any]]:
        """
        List all accessible SQL Warehouses.

        Returns:
            List of warehouse info dicts
        """
        try:
            warehouses = list(self._sdk_client.warehouses.list())
            return [
                {
                    "id": w.id,
                    "name": w.name,
                    "state": str(w.state) if w.state else "UNKNOWN",
                    "cluster_size": w.cluster_size,
                    "auto_stop_mins": w.auto_stop_mins,
                }
                for w in warehouses
            ]
        except Exception:
            return []

    def get_genie_space_details(self, space_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific Genie Space.

        Args:
            space_id: The Genie Space ID

        Returns:
            Space details dict or None if not found
        """
        try:
            response = self._http_client.get(f"/api/2.0/genie/spaces/{space_id}")
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def validate_genie_space(self, space_id: str) -> Dict[str, Any]:
        """
        Validate if a Genie Space ID is accessible.

        Args:
            space_id: The Genie Space ID to validate

        Returns:
            {"valid": bool, "name": str | None, "error": str | None}
        """
        try:
            details = self.get_genie_space_details(space_id)
            if details:
                return {
                    "valid": True,
                    "name": details.get("title", details.get("name", "Unknown")),
                    "error": None
                }
            return {
                "valid": False,
                "name": None,
                "error": "Space not found"
            }
        except Exception as e:
            return {
                "valid": False,
                "name": None,
                "error": str(e)
            }

    def close(self):
        """Close HTTP client."""
        self._http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
