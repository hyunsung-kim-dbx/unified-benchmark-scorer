"""
Unified Benchmark Runner.

Orchestrates benchmark execution across multiple systems
(Genie tuned/untuned, MAS tuned/untuned) and generates comparison reports.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

from ..api.sql_executor import SQLExecutor
from .llm_judge import LLMJudge


class SystemType(Enum):
    """Types of systems under test."""
    GENIE_UNTUNED = "genie_untuned"
    GENIE_TUNED = "genie_tuned"
    MAS_UNTUNED = "mas_untuned"
    MAS_TUNED = "mas_tuned"


@dataclass
class SystemConfig:
    """Configuration for a single system under test."""
    system_type: SystemType
    client: Any  # GenieClient or MASBenchmarkClient
    name: str
    description: str = ""


@dataclass
class BenchmarkResult:
    """Result of running one benchmark against one system."""
    benchmark_id: str
    system: SystemType
    question: str
    expected_sql: str
    system_sql: Optional[str]
    expected_result: Optional[Dict]
    system_result: Optional[Dict]
    passed: bool
    failure_reason: Optional[str]
    failure_category: Optional[str]
    response_time_ms: float
    comparison_details: Dict = field(default_factory=dict)


class UnifiedBenchmarkRunner:
    """
    Run benchmarks across multiple systems and compare results.

    Usage:
        runner = UnifiedBenchmarkRunner(
            systems=[
                SystemConfig(SystemType.GENIE_UNTUNED, genie_untuned_client, "Genie Base"),
                SystemConfig(SystemType.GENIE_TUNED, genie_tuned_client, "Genie Tuned"),
                SystemConfig(SystemType.MAS_UNTUNED, mas_untuned_client, "MAS + Base"),
                SystemConfig(SystemType.MAS_TUNED, mas_tuned_client, "MAS + Tuned"),
            ],
            sql_executor=sql_executor,
            llm_judge=llm_judge
        )

        report = runner.run(benchmarks)
    """

    def __init__(
        self,
        systems: List[SystemConfig],
        sql_executor: SQLExecutor,
        llm_client: LLMJudge,
        config: Optional[Dict] = None,
        progress_callback: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Initialize the benchmark runner.

        Args:
            systems: List of systems to test
            sql_executor: SQL executor for running expected queries
            llm_client: LLM judge for comparing results
            config: Optional configuration dict
            progress_callback: Optional callback for progress updates
        """
        self.systems = systems
        self.sql_executor = sql_executor
        self.llm_judge = llm_client
        self.config = config or {}
        self.progress_callback = progress_callback

        self._parallel_workers = self.config.get("parallel_workers", 2)
        self._timeout = self.config.get("timeout", 120)

    def run(self, benchmarks: List[Dict]) -> Dict:
        """
        Run all benchmarks against all systems.

        Args:
            benchmarks: List of benchmark dicts with:
                - id: str
                - question: str
                - expected_sql: str

        Returns:
            Comprehensive comparison report
        """
        # Run async in sync context
        return asyncio.run(self._run_async(benchmarks))

    async def _run_async(self, benchmarks: List[Dict]) -> Dict:
        """Async implementation of run()."""
        results: Dict[SystemType, List[BenchmarkResult]] = {
            system.system_type: [] for system in self.systems
        }

        total = len(benchmarks)

        for idx, benchmark in enumerate(benchmarks):
            self._emit_progress({
                "phase": "benchmark",
                "completed": idx,
                "total": total,
                "message": f"Running benchmark {benchmark.get('id', idx+1)}"
            })

            # Execute expected SQL once (ground truth for all systems)
            expected_result = await self._execute_expected_sql(benchmark["expected_sql"])

            # Test each system
            for system in self.systems:
                try:
                    result = await self._score_single(benchmark, system, expected_result)
                    results[system.system_type].append(result)
                except Exception as e:
                    error_result = self._create_error_result(benchmark, system, e)
                    results[system.system_type].append(error_result)

        self._emit_progress({
            "phase": "complete",
            "completed": total,
            "total": total,
            "message": "Benchmark complete"
        })

        return self._compile_report(results)

    async def _execute_expected_sql(self, sql: str) -> Optional[Dict]:
        """Execute the expected SQL to get ground truth result."""
        try:
            result = self.sql_executor.execute(sql)
            if result["status"] == "SUCCEEDED":
                return {
                    "columns": result["columns"],
                    "data": result["data"],
                    "row_count": result["row_count"]
                }
            return None
        except Exception:
            return None

    async def _score_single(
        self,
        benchmark: Dict,
        system: SystemConfig,
        expected_result: Optional[Dict]
    ) -> BenchmarkResult:
        """Score a single benchmark against a single system."""
        benchmark_id = benchmark.get("id", "unknown")
        question = benchmark["question"]
        expected_sql = benchmark["expected_sql"]

        # Query the system
        start_time = time.time()
        try:
            response = system.client.ask(question, timeout=self._timeout)
        except Exception as e:
            return BenchmarkResult(
                benchmark_id=benchmark_id,
                system=system.system_type,
                question=question,
                expected_sql=expected_sql,
                system_sql=None,
                expected_result=expected_result,
                system_result=None,
                passed=False,
                failure_reason=str(e),
                failure_category="execution_error",
                response_time_ms=(time.time() - start_time) * 1000
            )

        response_time_ms = response.get("response_time_ms", 0)

        # Check for system failure
        if response.get("status") != "COMPLETED":
            failure_category = "timeout" if response.get("status") == "TIMEOUT" else "execution_error"
            return BenchmarkResult(
                benchmark_id=benchmark_id,
                system=system.system_type,
                question=question,
                expected_sql=expected_sql,
                system_sql=response.get("sql"),
                expected_result=expected_result,
                system_result=response.get("result"),
                passed=False,
                failure_reason=response.get("error", "System query failed"),
                failure_category=failure_category,
                response_time_ms=response_time_ms
            )

        system_result = response.get("result")

        # Compare results using LLM judge
        if expected_result is None:
            # Can't compare without ground truth
            return BenchmarkResult(
                benchmark_id=benchmark_id,
                system=system.system_type,
                question=question,
                expected_sql=expected_sql,
                system_sql=response.get("sql"),
                expected_result=None,
                system_result=system_result,
                passed=False,
                failure_reason="Failed to execute expected SQL for ground truth",
                failure_category="execution_error",
                response_time_ms=response_time_ms
            )

        comparison = self.llm_judge.compare(
            question=question,
            expected_sql=expected_sql,
            expected_result=expected_result,
            actual_sql=response.get("sql"),
            actual_result=system_result
        )

        return BenchmarkResult(
            benchmark_id=benchmark_id,
            system=system.system_type,
            question=question,
            expected_sql=expected_sql,
            system_sql=response.get("sql"),
            expected_result=expected_result,
            system_result=system_result,
            passed=comparison["passed"],
            failure_reason=comparison.get("failure_reason"),
            failure_category=comparison.get("failure_category"),
            response_time_ms=response_time_ms,
            comparison_details=comparison
        )

    def _create_error_result(
        self,
        benchmark: Dict,
        system: SystemConfig,
        error: Exception
    ) -> BenchmarkResult:
        """Create an error result for exception cases."""
        return BenchmarkResult(
            benchmark_id=benchmark.get("id", "unknown"),
            system=system.system_type,
            question=benchmark["question"],
            expected_sql=benchmark["expected_sql"],
            system_sql=None,
            expected_result=None,
            system_result=None,
            passed=False,
            failure_reason=str(error),
            failure_category="execution_error",
            response_time_ms=0
        )

    def _compile_report(self, results: Dict[SystemType, List[BenchmarkResult]]) -> Dict:
        """Generate comprehensive comparison report."""
        # Calculate per-system stats
        per_system = {}
        for system_type, system_results in results.items():
            passed = sum(1 for r in system_results if r.passed)
            failed = len(system_results) - passed
            total = len(system_results)

            # Calculate avg latency
            latencies = [r.response_time_ms for r in system_results if r.response_time_ms > 0]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            # Failure category breakdown
            failure_categories = {}
            for r in system_results:
                if not r.passed and r.failure_category:
                    failure_categories[r.failure_category] = \
                        failure_categories.get(r.failure_category, 0) + 1

            per_system[system_type.value] = {
                "score": passed / total if total > 0 else 0,
                "passed": passed,
                "failed": failed,
                "total": total,
                "avg_latency_ms": avg_latency,
                "failure_categories": failure_categories
            }

        # Find best system
        best_system = max(per_system.items(), key=lambda x: x[1]["score"])

        # Head-to-head analysis
        head_to_head = self._compute_head_to_head(results)

        # Question breakdown
        question_breakdown = self._compute_question_breakdown(results)

        return {
            "summary": {
                "best_system": best_system[0],
                "best_score": best_system[1]["score"],
                "total_benchmarks": len(next(iter(results.values()), []))
            },
            "per_system": per_system,
            "head_to_head": head_to_head,
            "question_breakdown": question_breakdown,
            "results": {
                system_type.value: [self._result_to_dict(r) for r in system_results]
                for system_type, system_results in results.items()
            }
        }

    def _compute_head_to_head(
        self,
        results: Dict[SystemType, List[BenchmarkResult]]
    ) -> Dict:
        """Compute head-to-head comparisons."""
        head_to_head = {}

        # Tuning impact for Genie
        if SystemType.GENIE_UNTUNED in results and SystemType.GENIE_TUNED in results:
            head_to_head["tuning_impact_genie"] = self._compare_systems(
                results[SystemType.GENIE_UNTUNED],
                results[SystemType.GENIE_TUNED]
            )

        # Tuning impact for MAS
        if SystemType.MAS_UNTUNED in results and SystemType.MAS_TUNED in results:
            head_to_head["tuning_impact_mas"] = self._compare_systems(
                results[SystemType.MAS_UNTUNED],
                results[SystemType.MAS_TUNED]
            )

        # MAS vs Genie (untuned)
        if SystemType.GENIE_UNTUNED in results and SystemType.MAS_UNTUNED in results:
            head_to_head["mas_vs_genie_untuned"] = self._compare_systems(
                results[SystemType.GENIE_UNTUNED],
                results[SystemType.MAS_UNTUNED]
            )

        # MAS vs Genie (tuned)
        if SystemType.GENIE_TUNED in results and SystemType.MAS_TUNED in results:
            head_to_head["mas_vs_genie_tuned"] = self._compare_systems(
                results[SystemType.GENIE_TUNED],
                results[SystemType.MAS_TUNED]
            )

        return head_to_head

    def _compare_systems(
        self,
        baseline: List[BenchmarkResult],
        comparison: List[BenchmarkResult]
    ) -> Dict:
        """Compare two systems and find improvements/regressions."""
        baseline_by_id = {r.benchmark_id: r for r in baseline}
        comparison_by_id = {r.benchmark_id: r for r in comparison}

        improved = []  # Comparison passed, baseline failed
        regressed = []  # Baseline passed, comparison failed

        for bid in baseline_by_id:
            if bid in comparison_by_id:
                b_passed = baseline_by_id[bid].passed
                c_passed = comparison_by_id[bid].passed

                if c_passed and not b_passed:
                    improved.append(bid)
                elif b_passed and not c_passed:
                    regressed.append(bid)

        return {
            "improved": improved,
            "regressed": regressed,
            "net_improvement": len(improved) - len(regressed)
        }

    def _compute_question_breakdown(
        self,
        results: Dict[SystemType, List[BenchmarkResult]]
    ) -> List[Dict]:
        """Compute per-question breakdown across systems."""
        # Group by benchmark_id
        by_question: Dict[str, Dict] = {}

        for system_type, system_results in results.items():
            for result in system_results:
                if result.benchmark_id not in by_question:
                    by_question[result.benchmark_id] = {
                        "benchmark_id": result.benchmark_id,
                        "question": result.question
                    }
                by_question[result.benchmark_id][system_type.value] = {
                    "passed": result.passed,
                    "latency_ms": result.response_time_ms,
                    "failure_reason": result.failure_reason
                }

        return list(by_question.values())

    def _result_to_dict(self, result: BenchmarkResult) -> Dict:
        """Convert BenchmarkResult to dict."""
        return {
            "benchmark_id": result.benchmark_id,
            "system": result.system.value,
            "question": result.question,
            "expected_sql": result.expected_sql,
            "system_sql": result.system_sql,
            "passed": result.passed,
            "failure_reason": result.failure_reason,
            "failure_category": result.failure_category,
            "response_time_ms": result.response_time_ms,
            "comparison_details": result.comparison_details
        }

    def _emit_progress(self, event: Dict):
        """Emit progress event if callback is set."""
        if self.progress_callback:
            self.progress_callback(event)
