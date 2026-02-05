"""
MAS (Multi-Agent Supervisor) benchmark client adapter.

Provides a Genie-compatible interface for MAS queries,
enabling unified benchmark comparison across systems.
"""

import time
from typing import Dict, Optional, Any

import httpx


class MASBenchmarkClient:
    """
    Adapter that makes MAS look like GenieClient.

    When benchmark_mode=True, MAS should return:
    {
        "status": "COMPLETED",
        "sql": "...",           # From underlying Genie call
        "result": {...},        # From underlying Genie call
        "response_text": "...", # Formatted MAS response
        "charts": [...]         # Any generated charts
    }
    """

    def __init__(
        self,
        mas_endpoint: str,
        genie_space_id: str,
        benchmark_mode: bool = True,
        timeout: int = 180,
    ):
        """
        Initialize MAS benchmark client.

        Args:
            mas_endpoint: MAS supervisor endpoint URL
            genie_space_id: Genie Space ID for MAS to use
            benchmark_mode: Enable benchmark mode to capture raw Genie results
            timeout: Request timeout in seconds (MAS may be slower)
        """
        self.mas_endpoint = mas_endpoint.rstrip("/")
        self.genie_space_id = genie_space_id
        self.benchmark_mode = benchmark_mode
        self.timeout = timeout

        self._client = httpx.Client(
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )

    def ask(self, question: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Ask MAS a question and return Genie-compatible response format.

        Args:
            question: Natural language question
            timeout: Optional timeout override

        Returns:
            {
                "status": "COMPLETED" | "FAILED" | "TIMEOUT",
                "sql": str | None,
                "result": {"columns": [...], "data": [...], "row_count": int} | None,
                "error": str | None,
                "response_time_ms": float,
                "raw_response": str  # Original MAS text response
            }
        """
        start_time = time.time()
        effective_timeout = timeout or self.timeout

        try:
            response = self._client.post(
                f"{self.mas_endpoint}/supervisor/query",
                json={
                    "question": question,
                    "genie_space_id": self.genie_space_id,
                    "benchmark_mode": self.benchmark_mode,
                },
                timeout=effective_timeout
            )
            response.raise_for_status()

            return self._parse_response(response.json(), start_time)

        except httpx.TimeoutException:
            return self._error_response("Request timed out", start_time, "TIMEOUT")
        except httpx.HTTPStatusError as e:
            return self._error_response(
                f"HTTP error: {e.response.status_code} - {e.response.text}",
                start_time
            )
        except httpx.ConnectError:
            return self._error_response(
                f"Failed to connect to MAS endpoint: {self.mas_endpoint}",
                start_time
            )
        except Exception as e:
            return self._error_response(str(e), start_time)

    def _parse_response(self, data: Dict, start_time: float) -> Dict[str, Any]:
        """Parse MAS response into Genie-compatible format."""
        response_time_ms = (time.time() - start_time) * 1000

        # Check if MAS returned structured benchmark data
        status = data.get("status", "COMPLETED")

        if status == "FAILED":
            return {
                "status": "FAILED",
                "sql": None,
                "result": None,
                "error": data.get("error", "MAS query failed"),
                "response_time_ms": response_time_ms,
                "raw_response": data.get("response_text") or data.get("response")
            }

        # Extract SQL and result from benchmark mode response
        sql = data.get("sql")
        result = data.get("result")

        # Normalize result format if present
        if result and isinstance(result, dict):
            result = {
                "columns": result.get("columns", []),
                "data": result.get("data", []),
                "row_count": result.get("row_count", len(result.get("data", [])))
            }

        return {
            "status": "COMPLETED",
            "sql": sql,
            "result": result,
            "error": None,
            "response_time_ms": response_time_ms,
            "raw_response": data.get("response_text") or data.get("response"),
            "charts": data.get("charts", [])
        }

    def _error_response(
        self,
        error: str,
        start_time: float,
        status: str = "FAILED"
    ) -> Dict[str, Any]:
        """Create error response."""
        return {
            "status": status,
            "sql": None,
            "result": None,
            "error": error,
            "response_time_ms": (time.time() - start_time) * 1000,
            "raw_response": None
        }

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
