#!/usr/bin/env python3
"""
Run unified benchmarks across all systems.

Usage:
    python scripts/run_benchmark.py \
        --benchmarks benchmarks/benchmarks.json \
        --genie-untuned-space-id abc123 \
        --genie-tuned-space-id def456 \
        --mas-endpoint http://localhost:8000 \
        --output results/comparison_$(date +%Y%m%d).json \
        --report results/comparison_$(date +%Y%m%d).md
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from api.genie_client import GenieClient
from api.mas_client import MASBenchmarkClient
from api.sql_executor import SQLExecutor
from scoring.unified_runner import UnifiedBenchmarkRunner, SystemConfig, SystemType
from scoring.report_generator import ComparisonReportGenerator
from scoring.llm_judge import AnthropicJudge, OpenAIJudge


def get_llm_judge():
    """Get configured LLM judge."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        return AnthropicJudge(api_key=anthropic_key)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return OpenAIJudge(api_key=openai_key)

    raise ValueError(
        "No LLM API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run unified benchmarks across multiple systems"
    )
    parser.add_argument(
        "--benchmarks",
        required=True,
        help="Path to benchmarks JSON file"
    )
    parser.add_argument(
        "--genie-untuned-space-id",
        help="Genie Space ID for untuned system"
    )
    parser.add_argument(
        "--genie-tuned-space-id",
        help="Genie Space ID for tuned system"
    )
    parser.add_argument(
        "--mas-endpoint",
        help="MAS supervisor endpoint URL"
    )
    parser.add_argument(
        "--output",
        help="Output JSON path for full results"
    )
    parser.add_argument(
        "--report",
        help="Output Markdown report path"
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=["genie_untuned", "genie_tuned", "mas_untuned", "mas_tuned"],
        default=["genie_untuned", "genie_tuned", "mas_untuned", "mas_tuned"],
        help="Systems to test (default: all)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=2,
        help="Parallel workers per system"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds for each query"
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file"
    )

    args = parser.parse_args()

    # Load environment variables
    env_path = Path(args.env_file)
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")

    # Get credentials from environment
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    warehouse_id = os.getenv("SQL_WAREHOUSE_ID")

    if not all([host, token, warehouse_id]):
        print("Error: Missing required environment variables:")
        print("  DATABRICKS_HOST, DATABRICKS_TOKEN, SQL_WAREHOUSE_ID")
        sys.exit(1)

    # Load benchmarks
    print(f"Loading benchmarks from {args.benchmarks}...")
    with open(args.benchmarks) as f:
        benchmarks = json.load(f)
    print(f"Loaded {len(benchmarks)} benchmarks")

    # Initialize systems based on --systems flag
    systems = []

    genie_untuned_id = args.genie_untuned_space_id or os.getenv("GENIE_UNTUNED_SPACE_ID")
    genie_tuned_id = args.genie_tuned_space_id or os.getenv("GENIE_TUNED_SPACE_ID")
    mas_endpoint = args.mas_endpoint or os.getenv("MAS_ENDPOINT")

    if "genie_untuned" in args.systems and genie_untuned_id:
        systems.append(SystemConfig(
            SystemType.GENIE_UNTUNED,
            GenieClient(host, token, genie_untuned_id, timeout=args.timeout),
            "Genie (Untuned)"
        ))
        print(f"  Added: Genie (Untuned) - Space ID: {genie_untuned_id[:8]}...")

    if "genie_tuned" in args.systems and genie_tuned_id:
        systems.append(SystemConfig(
            SystemType.GENIE_TUNED,
            GenieClient(host, token, genie_tuned_id, timeout=args.timeout),
            "Genie (Tuned)"
        ))
        print(f"  Added: Genie (Tuned) - Space ID: {genie_tuned_id[:8]}...")

    if "mas_untuned" in args.systems and mas_endpoint and genie_untuned_id:
        systems.append(SystemConfig(
            SystemType.MAS_UNTUNED,
            MASBenchmarkClient(mas_endpoint, genie_untuned_id, timeout=args.timeout + 60),
            "MAS + Untuned"
        ))
        print(f"  Added: MAS + Untuned - Endpoint: {mas_endpoint}")

    if "mas_tuned" in args.systems and mas_endpoint and genie_tuned_id:
        systems.append(SystemConfig(
            SystemType.MAS_TUNED,
            MASBenchmarkClient(mas_endpoint, genie_tuned_id, timeout=args.timeout + 60),
            "MAS + Tuned"
        ))
        print(f"  Added: MAS + Tuned - Endpoint: {mas_endpoint}")

    if not systems:
        print("Error: No systems configured. Check Space IDs and endpoint settings.")
        sys.exit(1)

    print(f"\nTesting {len(systems)} system(s)")

    # Progress callback
    def on_progress(event):
        phase = event.get("phase", "")
        message = event.get("message", "")
        completed = event.get("completed", 0)
        total = event.get("total", 0)
        print(f"  [{phase.upper()}] {message} ({completed}/{total})")

    # Create runner
    print("\nInitializing benchmark runner...")
    runner = UnifiedBenchmarkRunner(
        systems=systems,
        sql_executor=SQLExecutor(host, token, warehouse_id),
        llm_client=get_llm_judge(),
        config={
            "parallel_workers": args.parallel,
            "timeout": args.timeout
        },
        progress_callback=on_progress
    )

    # Run benchmarks
    print("\nRunning benchmarks...")
    start_time = datetime.now()

    results = runner.run(benchmarks)

    elapsed = datetime.now() - start_time
    print(f"\nBenchmark completed in {elapsed.total_seconds():.1f}s")

    # Generate report
    report = ComparisonReportGenerator(results)

    # Save outputs
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_json(str(output_path))
        print(f"\nResults saved to: {args.output}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report.to_markdown())
        print(f"Report saved to: {args.report}")

    # Print summary
    print("\n")
    report.print_summary()


if __name__ == "__main__":
    main()
