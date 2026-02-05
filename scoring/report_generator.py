"""
Report generator for unified benchmark results.

Generates human-readable reports in Markdown, HTML, and JSON formats.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional


class ComparisonReportGenerator:
    """Generate human-readable reports from unified benchmark results."""

    def __init__(self, results: Dict):
        """
        Initialize report generator.

        Args:
            results: Results dict from UnifiedBenchmarkRunner
        """
        self.results = results
        self.summary = results.get("summary", {})
        self.per_system = results.get("per_system", {})
        self.head_to_head = results.get("head_to_head", {})
        self.question_breakdown = results.get("question_breakdown", [])

    def to_markdown(self) -> str:
        """Generate Markdown report."""
        lines = []

        # Header
        lines.append("# Unified Benchmark Results")
        lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        # Executive Summary
        lines.append("## Executive Summary\n")
        lines.append(f"- **Best System**: {self.summary.get('best_system', 'N/A')}")
        lines.append(f"- **Best Score**: {self.summary.get('best_score', 0):.1%}")
        lines.append(f"- **Total Benchmarks**: {self.summary.get('total_benchmarks', 0)}")
        lines.append("")

        # System Comparison Table
        lines.append("## System Comparison\n")
        lines.append("| System | Score | Passed | Failed | Avg Latency |")
        lines.append("|--------|-------|--------|--------|-------------|")

        for system, stats in self.per_system.items():
            score = stats.get("score", 0)
            passed = stats.get("passed", 0)
            failed = stats.get("failed", 0)
            latency = stats.get("avg_latency_ms", 0)
            marker = " ✓" if system == self.summary.get("best_system") else ""

            lines.append(
                f"| {system}{marker} | {score:.1%} | {passed} | {failed} | {latency:.0f}ms |"
            )
        lines.append("")

        # Head-to-Head Analysis
        lines.append("## Head-to-Head Analysis\n")

        if "tuning_impact_genie" in self.head_to_head:
            h2h = self.head_to_head["tuning_impact_genie"]
            lines.append("### Tuning Impact (Genie)\n")
            lines.append(f"- Improved: {len(h2h.get('improved', []))} questions")
            lines.append(f"- Regressed: {len(h2h.get('regressed', []))} questions")
            lines.append(f"- **Net Improvement: {h2h.get('net_improvement', 0):+d}**")
            lines.append("")

        if "tuning_impact_mas" in self.head_to_head:
            h2h = self.head_to_head["tuning_impact_mas"]
            lines.append("### Tuning Impact (MAS)\n")
            lines.append(f"- Improved: {len(h2h.get('improved', []))} questions")
            lines.append(f"- Regressed: {len(h2h.get('regressed', []))} questions")
            lines.append(f"- **Net Improvement: {h2h.get('net_improvement', 0):+d}**")
            lines.append("")

        if "mas_vs_genie_untuned" in self.head_to_head:
            h2h = self.head_to_head["mas_vs_genie_untuned"]
            lines.append("### MAS vs Genie (Untuned)\n")
            lines.append(f"- MAS better: {len(h2h.get('improved', []))} questions")
            lines.append(f"- Genie better: {len(h2h.get('regressed', []))} questions")
            lines.append(f"- **Net: {h2h.get('net_improvement', 0):+d}**")
            lines.append("")

        if "mas_vs_genie_tuned" in self.head_to_head:
            h2h = self.head_to_head["mas_vs_genie_tuned"]
            lines.append("### MAS vs Genie (Tuned)\n")
            lines.append(f"- MAS better: {len(h2h.get('improved', []))} questions")
            lines.append(f"- Genie better: {len(h2h.get('regressed', []))} questions")
            lines.append(f"- **Net: {h2h.get('net_improvement', 0):+d}**")
            lines.append("")

        # Failure Category Breakdown
        lines.append("## Failure Categories\n")

        for system, stats in self.per_system.items():
            categories = stats.get("failure_categories", {})
            if categories:
                lines.append(f"### {system}\n")
                for category, count in sorted(
                    categories.items(), key=lambda x: -x[1]
                ):
                    lines.append(f"- {category}: {count}")
                lines.append("")

        # Divergent Questions (where systems disagree)
        divergent = self._find_divergent_questions()
        if divergent:
            lines.append("## Divergent Questions\n")
            lines.append("Questions where systems produced different results:\n")

            for q in divergent[:10]:  # Show top 10
                lines.append(f"### {q['benchmark_id']}\n")
                lines.append(f"**Question**: {q['question']}\n")
                lines.append("| System | Result |")
                lines.append("|--------|--------|")
                for system in self.per_system.keys():
                    if system in q:
                        result = "✓ Pass" if q[system].get("passed") else "✗ Fail"
                        lines.append(f"| {system} | {result} |")
                lines.append("")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Generate HTML report with basic styling."""
        md_content = self.to_markdown()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Unified Benchmark Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{ color: #1a1a1a; border-bottom: 2px solid #0066cc; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 30px; }}
        h3 {{ color: #555; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #0066cc;
            color: white;
        }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .pass {{ color: #28a745; font-weight: bold; }}
        .fail {{ color: #dc3545; font-weight: bold; }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <pre style="white-space: pre-wrap;">{md_content}</pre>
</body>
</html>"""
        return html

    def to_json(self, path: Optional[str] = None) -> str:
        """
        Export full results to JSON.

        Args:
            path: Optional file path to save to

        Returns:
            JSON string
        """
        json_str = json.dumps(self.results, indent=2, default=str)

        if path:
            with open(path, 'w') as f:
                f.write(json_str)

        return json_str

    def print_summary(self) -> None:
        """Print summary to console."""
        width = 80
        print("=" * width)
        print("UNIFIED BENCHMARK RESULTS".center(width))
        print("=" * width)
        print()

        # System comparison
        print(f"{'System':<25} {'Score':<10} {'Passed':<8} {'Failed':<8} {'Latency':<10}")
        print("-" * width)

        for system, stats in self.per_system.items():
            score = f"{stats.get('score', 0):.1%}"
            passed = str(stats.get("passed", 0))
            failed = str(stats.get("failed", 0))
            latency = f"{stats.get('avg_latency_ms', 0):.0f}ms"
            marker = " ← BEST" if system == self.summary.get("best_system") else ""

            print(f"{system:<25} {score:<10} {passed:<8} {failed:<8} {latency:<10}{marker}")

        print()
        print("-" * width)
        print("HEAD-TO-HEAD ANALYSIS")
        print("-" * width)

        for comparison_name, h2h in self.head_to_head.items():
            improved = len(h2h.get("improved", []))
            regressed = len(h2h.get("regressed", []))
            net = h2h.get("net_improvement", 0)
            print(f"{comparison_name}: +{improved} improved, -{regressed} regressed (net: {net:+d})")

        print()
        print("=" * width)

    def _find_divergent_questions(self) -> List[Dict]:
        """Find questions where systems disagree."""
        divergent = []

        for q in self.question_breakdown:
            results = []
            for system in self.per_system.keys():
                if system in q:
                    results.append(q[system].get("passed"))

            # Check if there's disagreement
            if len(set(results)) > 1:
                divergent.append(q)

        return divergent
