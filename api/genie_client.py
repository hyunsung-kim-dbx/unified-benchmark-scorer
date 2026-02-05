"""
Genie Space client for benchmark scoring.

Provides a simplified interface for querying Databricks Genie Spaces
and extracting SQL and results for comparison.
"""

import time
from typing import Dict, Optional, Any
from dataclasses import dataclass

import httpx


@dataclass
class GenieResponse:
    """Response from a Genie Space query."""
    status: str  # COMPLETED, FAILED, TIMEOUT
    sql: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    response_time_ms: float = 0.0
    raw_response: Optional[Dict] = None


class GenieClient:
    """
    Client for querying Databricks Genie Spaces.

    Implements the conversational API to ask questions and retrieve
    SQL and results for benchmark comparison.
    """

    def __init__(
        self,
        host: str,
        token: str,
        space_id: str,
        timeout: int = 120,
    ):
        """
        Initialize Genie client.

        Args:
            host: Databricks workspace host (e.g., https://workspace.cloud.databricks.com)
            token: Databricks access token
            space_id: Genie Space ID
            timeout: Request timeout in seconds
        """
        self.host = host.rstrip("/")
        self.token = token
        self.space_id = space_id
        self.timeout = timeout

        self._client = httpx.Client(
            base_url=self.host,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def ask(self, question: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Ask a question to the Genie Space.

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
                "raw_response": dict
            }
        """
        start_time = time.time()
        effective_timeout = timeout or self.timeout

        try:
            # Start conversation
            conversation = self._start_conversation(question)
            conversation_id = conversation.get("conversation_id")
            message_id = conversation.get("message_id")

            if not conversation_id or not message_id:
                return self._error_response(
                    "Failed to start conversation",
                    start_time
                )

            # Poll for completion
            result = self._poll_for_result(
                conversation_id,
                message_id,
                effective_timeout,
                start_time
            )

            return result

        except httpx.TimeoutException:
            return self._error_response("Request timed out", start_time, "TIMEOUT")
        except httpx.HTTPStatusError as e:
            return self._error_response(f"HTTP error: {e.response.status_code}", start_time)
        except Exception as e:
            return self._error_response(str(e), start_time)

    def _start_conversation(self, question: str) -> Dict:
        """Start a new conversation with the Genie Space."""
        response = self._client.post(
            f"/api/2.0/genie/spaces/{self.space_id}/start-conversation",
            json={"content": question}
        )
        response.raise_for_status()
        return response.json()

    def _poll_for_result(
        self,
        conversation_id: str,
        message_id: str,
        timeout: int,
        start_time: float
    ) -> Dict[str, Any]:
        """Poll for message completion and extract results."""
        poll_interval = 2.0

        while (time.time() - start_time) < timeout:
            message = self._get_message(conversation_id, message_id)
            status = message.get("status", "")

            if status == "COMPLETED":
                return self._extract_result(message, start_time)
            elif status in ("FAILED", "CANCELLED"):
                error = message.get("error", {}).get("message", "Unknown error")
                return self._error_response(error, start_time)

            time.sleep(poll_interval)

        return self._error_response("Polling timed out", start_time, "TIMEOUT")

    def _get_message(self, conversation_id: str, message_id: str) -> Dict:
        """Get message status and content."""
        response = self._client.get(
            f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}"
        )
        response.raise_for_status()
        return response.json()

    def _extract_result(self, message: Dict, start_time: float) -> Dict[str, Any]:
        """Extract SQL and results from completed message."""
        response_time_ms = (time.time() - start_time) * 1000

        # Extract SQL from attachments
        sql = None
        result = None

        attachments = message.get("attachments", [])
        for attachment in attachments:
            if attachment.get("type") == "query":
                query_info = attachment.get("query", {})
                sql = query_info.get("query")

                # Extract result if available
                result_data = query_info.get("result", {})
                if result_data:
                    columns = [
                        col.get("name")
                        for col in result_data.get("schema", {}).get("columns", [])
                    ]
                    data = result_data.get("data_array", [])
                    result = {
                        "columns": columns,
                        "data": [list(row) for row in data],
                        "row_count": len(data)
                    }
                break

        return {
            "status": "COMPLETED",
            "sql": sql,
            "result": result,
            "error": None,
            "response_time_ms": response_time_ms,
            "raw_response": message
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
