"""
Unified Benchmark Scorer - Streamlit Frontend

A Databricks App for comparing benchmark performance across
Genie Space (tuned/untuned) and MAS (tuned/untuned) systems.
"""

import json
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from api.genie_client import GenieClient
from api.mas_client import MASBenchmarkClient
from api.sql_executor import SQLExecutor
from scoring.unified_runner import UnifiedBenchmarkRunner, SystemConfig, SystemType
from scoring.report_generator import ComparisonReportGenerator
from scoring.llm_judge import AnthropicJudge, OpenAIJudge
from utils.auth import get_user_context, get_fallback_token

# Load .env for local development
load_dotenv()

# Page config
st.set_page_config(
    page_title="Unified Benchmark Scorer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None
if "running" not in st.session_state:
    st.session_state.running = False


def get_config_value(key: str, default: str = "") -> str:
    """Get config value from secrets or environment."""
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


def get_llm_client():
    """Get configured LLM client for judging."""
    anthropic_key = get_config_value("ANTHROPIC_API_KEY")
    if anthropic_key:
        return AnthropicJudge(api_key=anthropic_key)

    openai_key = get_config_value("OPENAI_API_KEY")
    if openai_key:
        return OpenAIJudge(api_key=openai_key)

    raise ValueError("No LLM API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")


def main():
    st.title("📊 Unified Benchmark Scorer")
    st.markdown("Compare Genie Space and Multi-Agent Supervisor performance")

    # Show user context
    user_ctx = get_user_context()
    if user_ctx["email"]:
        st.sidebar.success(f"👤 Logged in as: {user_ctx['email']}")
    else:
        st.sidebar.warning("⚠️ Running without user context (local mode)")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Genie Space IDs
        st.subheader("Genie Spaces")
        genie_untuned_id = st.text_input(
            "Untuned Space ID",
            value=get_config_value("GENIE_UNTUNED_SPACE_ID"),
            type="password"
        )
        genie_tuned_id = st.text_input(
            "Tuned Space ID",
            value=get_config_value("GENIE_TUNED_SPACE_ID"),
            type="password"
        )

        # MAS endpoint
        st.subheader("Multi-Agent Supervisor")
        mas_endpoint = st.text_input(
            "MAS Endpoint",
            value=get_config_value("MAS_ENDPOINT", "http://localhost:8000")
        )

        # Systems to test
        st.subheader("Systems to Test")
        test_genie_untuned = st.checkbox("Genie (Untuned)", value=True)
        test_genie_tuned = st.checkbox("Genie (Tuned)", value=True)
        test_mas_untuned = st.checkbox("MAS + Untuned", value=True)
        test_mas_tuned = st.checkbox("MAS + Tuned", value=True)

        # Advanced settings
        with st.expander("Advanced Settings"):
            parallel_workers = st.slider("Parallel Workers", 1, 5, 2)
            timeout = st.slider("Timeout (seconds)", 30, 300, 120)

    # Main content area
    tab1, tab2, tab3 = st.tabs(["🚀 Run Benchmark", "📈 Results", "📋 History"])

    with tab1:
        render_run_tab(
            genie_untuned_id=genie_untuned_id,
            genie_tuned_id=genie_tuned_id,
            mas_endpoint=mas_endpoint,
            systems_config={
                "genie_untuned": test_genie_untuned,
                "genie_tuned": test_genie_tuned,
                "mas_untuned": test_mas_untuned,
                "mas_tuned": test_mas_tuned,
            },
            parallel_workers=parallel_workers,
            timeout=timeout
        )

    with tab2:
        render_results_tab()

    with tab3:
        render_history_tab()


def render_run_tab(
    genie_untuned_id: str,
    genie_tuned_id: str,
    mas_endpoint: str,
    systems_config: dict,
    parallel_workers: int,
    timeout: int
):
    """Render the benchmark execution tab."""

    st.header("Upload Benchmarks")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload benchmark JSON file",
        type=["json"],
        help="JSON array with objects containing: id, question, expected_sql"
    )

    # Or paste JSON
    with st.expander("Or paste JSON directly"):
        json_input = st.text_area(
            "Benchmark JSON",
            height=200,
            placeholder='[{"id": "q1", "question": "...", "expected_sql": "..."}]'
        )

    # Load benchmarks
    benchmarks = None
    if uploaded_file:
        try:
            benchmarks = json.load(uploaded_file)
            st.success(f"✅ Loaded {len(benchmarks)} benchmarks from file")
        except Exception as e:
            st.error(f"Error parsing file: {e}")
    elif json_input.strip():
        try:
            benchmarks = json.loads(json_input)
            st.success(f"✅ Loaded {len(benchmarks)} benchmarks from input")
        except Exception as e:
            st.error(f"Error parsing JSON: {e}")

    # Preview benchmarks
    if benchmarks:
        with st.expander(f"Preview Benchmarks ({len(benchmarks)} questions)"):
            df = pd.DataFrame(benchmarks)
            display_cols = [c for c in ["id", "question"] if c in df.columns]
            if display_cols:
                st.dataframe(df[display_cols], use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)

    # Run button
    st.divider()

    col1, col2 = st.columns([1, 3])
    with col1:
        run_button = st.button(
            "🚀 Run Benchmark",
            type="primary",
            disabled=not benchmarks or st.session_state.running,
            use_container_width=True
        )

    with col2:
        if st.session_state.running:
            st.info("⏳ Benchmark in progress...")

    # Execute benchmark
    if run_button and benchmarks:
        run_benchmark(
            benchmarks=benchmarks,
            genie_untuned_id=genie_untuned_id,
            genie_tuned_id=genie_tuned_id,
            mas_endpoint=mas_endpoint,
            systems_config=systems_config,
            parallel_workers=parallel_workers,
            timeout=timeout
        )


def run_benchmark(
    benchmarks: list,
    genie_untuned_id: str,
    genie_tuned_id: str,
    mas_endpoint: str,
    systems_config: dict,
    parallel_workers: int,
    timeout: int
):
    """Execute the benchmark run."""

    st.session_state.running = True

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Get user's token for queries
        user_token = get_fallback_token()

        if not user_token:
            st.error("No authentication token available. Set DATABRICKS_TOKEN.")
            st.session_state.running = False
            return

        host = get_config_value("DATABRICKS_HOST")
        warehouse_id = get_config_value("SQL_WAREHOUSE_ID")

        if not host or not warehouse_id:
            st.error("Missing DATABRICKS_HOST or SQL_WAREHOUSE_ID configuration.")
            st.session_state.running = False
            return

        # Initialize clients
        status_text.text("Initializing clients...")

        systems = []

        if systems_config["genie_untuned"] and genie_untuned_id:
            systems.append(SystemConfig(
                system_type=SystemType.GENIE_UNTUNED,
                client=GenieClient(host, user_token, genie_untuned_id, timeout=timeout),
                name="Genie (Untuned)"
            ))

        if systems_config["genie_tuned"] and genie_tuned_id:
            systems.append(SystemConfig(
                system_type=SystemType.GENIE_TUNED,
                client=GenieClient(host, user_token, genie_tuned_id, timeout=timeout),
                name="Genie (Tuned)"
            ))

        if systems_config["mas_untuned"] and genie_untuned_id and mas_endpoint:
            systems.append(SystemConfig(
                system_type=SystemType.MAS_UNTUNED,
                client=MASBenchmarkClient(mas_endpoint, genie_untuned_id, timeout=timeout + 60),
                name="MAS + Untuned"
            ))

        if systems_config["mas_tuned"] and genie_tuned_id and mas_endpoint:
            systems.append(SystemConfig(
                system_type=SystemType.MAS_TUNED,
                client=MASBenchmarkClient(mas_endpoint, genie_tuned_id, timeout=timeout + 60),
                name="MAS + Tuned"
            ))

        if not systems:
            st.error("No systems selected or configured!")
            st.session_state.running = False
            return

        # Progress callback
        def on_progress(event: dict):
            phase = event.get("phase", "")
            completed = event.get("completed", 0)
            total = event.get("total", len(benchmarks))
            message = event.get("message", "")

            progress = completed / total if total > 0 else 0
            progress_bar.progress(progress)
            status_text.text(f"[{phase.upper()}] {message} ({completed}/{total})")

        # Create runner
        runner = UnifiedBenchmarkRunner(
            systems=systems,
            sql_executor=SQLExecutor(host, user_token, warehouse_id),
            llm_client=get_llm_client(),
            config={
                "parallel_workers": parallel_workers,
                "timeout": timeout
            },
            progress_callback=on_progress
        )

        # Run!
        status_text.text("Running benchmarks...")
        results = runner.run(benchmarks)

        # Store results
        st.session_state.benchmark_results = results

        progress_bar.progress(1.0)
        status_text.text("✅ Benchmark complete!")

        # Show quick summary
        best_system = results.get("summary", {}).get("best_system", "N/A")
        best_score = results.get("summary", {}).get("best_score", 0)
        st.success(f"Benchmark complete! Best system: **{best_system}** with {best_score:.1%} accuracy")

    except Exception as e:
        st.error(f"Error running benchmark: {e}")
        import traceback
        st.code(traceback.format_exc())

    finally:
        st.session_state.running = False


def render_results_tab():
    """Render the results visualization tab."""

    results = st.session_state.benchmark_results

    if not results:
        st.info("No results yet. Run a benchmark first!")
        return

    st.header("📈 Benchmark Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    summary = results.get("summary", {})
    per_system = results.get("per_system", {})

    with col1:
        st.metric("Best System", summary.get("best_system", "N/A"))
    with col2:
        st.metric("Best Score", f"{summary.get('best_score', 0):.1%}")
    with col3:
        st.metric("Total Questions", summary.get("total_benchmarks", 0))
    with col4:
        if per_system:
            avg_latency = sum(s.get("avg_latency_ms", 0) for s in per_system.values()) / len(per_system)
            st.metric("Avg Latency", f"{avg_latency:.0f}ms")
        else:
            st.metric("Avg Latency", "N/A")

    st.divider()

    # Per-system comparison table
    st.subheader("System Comparison")

    comparison_data = []
    for system_name, stats in per_system.items():
        comparison_data.append({
            "System": system_name,
            "Score": f"{stats.get('score', 0):.1%}",
            "Passed": stats.get("passed", 0),
            "Failed": stats.get("failed", 0),
            "Avg Latency (ms)": f"{stats.get('avg_latency_ms', 0):.0f}"
        })

    if comparison_data:
        st.dataframe(
            pd.DataFrame(comparison_data),
            use_container_width=True,
            hide_index=True
        )

    # Head-to-head analysis
    st.subheader("Head-to-Head Analysis")

    head_to_head = results.get("head_to_head", {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Tuning Impact**")

        genie_tuning = head_to_head.get("tuning_impact_genie", {})
        mas_tuning = head_to_head.get("tuning_impact_mas", {})

        st.markdown(f"""
- **Genie**: +{len(genie_tuning.get('improved', []))} improved, -{len(genie_tuning.get('regressed', []))} regressed (net: {genie_tuning.get('net_improvement', 0):+d})
- **MAS**: +{len(mas_tuning.get('improved', []))} improved, -{len(mas_tuning.get('regressed', []))} regressed (net: {mas_tuning.get('net_improvement', 0):+d})
        """)

    with col2:
        st.markdown("**MAS vs Direct Genie**")

        mas_vs_untuned = head_to_head.get("mas_vs_genie_untuned", {})
        mas_vs_tuned = head_to_head.get("mas_vs_genie_tuned", {})

        st.markdown(f"""
- **vs Untuned**: {mas_vs_untuned.get('net_improvement', 0):+d} questions
- **vs Tuned**: {mas_vs_tuned.get('net_improvement', 0):+d} questions
        """)

    # Question breakdown
    st.subheader("Question-Level Results")

    question_breakdown = results.get("question_breakdown", [])
    if question_breakdown:
        # Create a DataFrame for display
        breakdown_df = pd.DataFrame(question_breakdown)

        # Add filter
        filter_option = st.selectbox(
            "Filter by",
            ["All", "All Passed", "All Failed", "Divergent (systems disagree)"]
        )

        if filter_option == "All Passed":
            breakdown_df = breakdown_df[breakdown_df.apply(
                lambda row: all(
                    row.get(sys, {}).get("passed", False)
                    for sys in per_system.keys() if sys in row
                ),
                axis=1
            )]
        elif filter_option == "All Failed":
            breakdown_df = breakdown_df[breakdown_df.apply(
                lambda row: all(
                    not row.get(sys, {}).get("passed", True)
                    for sys in per_system.keys() if sys in row
                ),
                axis=1
            )]
        elif filter_option == "Divergent (systems disagree)":
            breakdown_df = breakdown_df[breakdown_df.apply(
                lambda row: len(set(
                    row.get(sys, {}).get("passed", None)
                    for sys in per_system.keys() if sys in row
                )) > 1,
                axis=1
            )]

        st.dataframe(breakdown_df, use_container_width=True)

    # Download results
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "📥 Download JSON Results",
            data=json.dumps(results, indent=2, default=str),
            file_name=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col2:
        report_gen = ComparisonReportGenerator(results)
        st.download_button(
            "📥 Download Markdown Report",
            data=report_gen.to_markdown(),
            file_name=f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )


def render_history_tab():
    """Render the benchmark history tab."""
    st.info("History feature coming soon - will store results in Delta table")


if __name__ == "__main__":
    main()
