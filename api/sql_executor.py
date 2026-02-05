"""
SQL execution utility for Databricks warehouses.

Executes SQL statements and returns results in a normalized format
for benchmark comparison.
"""

import time
from typing import Dict, Optional, Any, List

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState


class SQLExecutor:
    """
    Execute SQL statements on Databricks SQL Warehouse.

    Used to execute expected_sql from benchmarks to establish
    ground truth for comparison.
    """

    def __init__(
        self,
        host: str,
        token: str,
        warehouse_id: str,
        timeout: int = 60,
    ):
        """
        Initialize SQL executor.

        Args:
            host: Databricks workspace host
            token: Databricks access token
            warehouse_id: SQL Warehouse ID
            timeout: Statement execution timeout in seconds
        """
        self.host = host
        self.warehouse_id = warehouse_id
        self.timeout = timeout

        self._client = WorkspaceClient(
            host=host,
            token=token
        )

    def execute(self, sql: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute SQL and return results.

        Args:
            sql: SQL statement to execute
            timeout: Optional timeout override

        Returns:
            {
                "status": "SUCCEEDED" | "FAILED",
                "columns": [...],
                "data": [[...], ...],
                "row_count": int,
                "error": str | None,
                "execution_time_ms": float
            }
        """
        start_time = time.time()
        effective_timeout = f"{timeout or self.timeout}s"

        try:
            response = self._client.statement_execution.execute_statement(
                warehouse_id=self.warehouse_id,
                statement=sql,
                wait_timeout=effective_timeout
            )

            execution_time_ms = (time.time() - start_time) * 1000

            if response.status.state == StatementState.SUCCEEDED:
                # Extract columns
                columns = []
                if response.manifest and response.manifest.schema:
                    columns = [col.name for col in response.manifest.schema.columns]

                # Extract data
                data = []
                if response.result and response.result.data_array:
                    data = [list(row) for row in response.result.data_array]

                row_count = response.manifest.total_row_count if response.manifest else len(data)

                return {
                    "status": "SUCCEEDED",
                    "columns": columns,
                    "data": data,
                    "row_count": row_count,
                    "error": None,
                    "execution_time_ms": execution_time_ms
                }
            else:
                error_msg = "Unknown error"
                if response.status.error:
                    error_msg = response.status.error.message

                return {
                    "status": "FAILED",
                    "columns": [],
                    "data": [],
                    "row_count": 0,
                    "error": error_msg,
                    "execution_time_ms": execution_time_ms
                }

        except Exception as e:
            return {
                "status": "FAILED",
                "columns": [],
                "data": [],
                "row_count": 0,
                "error": str(e),
                "execution_time_ms": (time.time() - start_time) * 1000
            }

    def validate_sql(self, sql: str) -> Dict[str, Any]:
        """
        Validate SQL without executing (dry run).

        Args:
            sql: SQL statement to validate

        Returns:
            {"valid": bool, "error": str | None}
        """
        try:
            # Use EXPLAIN to validate without executing
            explain_sql = f"EXPLAIN {sql}"
            result = self.execute(explain_sql, timeout=30)

            return {
                "valid": result["status"] == "SUCCEEDED",
                "error": result.get("error")
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
