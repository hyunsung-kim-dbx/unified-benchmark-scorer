"""
Unified Benchmark Scorer - Streamlit Frontend

A Databricks App for comparing benchmark performance across
Genie Space (tuned/untuned) and MAS (tuned/untuned) systems.
"""

import json
import os
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from api.genie_client import GenieClient
from api.mas_client import MASBenchmarkClient
from api.sql_executor import SQLExecutor
from api.workspace_client import WorkspaceResourceClient, GenieSpaceInfo
from api.mas_registry import MASRegistry, MASEndpoint
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
if "benchmarks" not in st.session_state:
    st.session_state.benchmarks = None
if "genie_spaces" not in st.session_state:
    st.session_state.genie_spaces = None
if "mas_registry" not in st.session_state:
    st.session_state.mas_registry = MASRegistry()


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


def load_genie_spaces() -> List[GenieSpaceInfo]:
    """Load available Genie Spaces."""
    token = get_fallback_token()
    host = get_config_value("DATABRICKS_HOST")

    if not token or not host:
        return []

    try:
        client = WorkspaceResourceClient(host, token)
        spaces = client.list_genie_spaces()
        client.close()
        return spaces
    except Exception as e:
        st.error(f"Failed to load Genie Spaces: {e}")
        return []


def main():
    st.title("📊 Unified Benchmark Scorer")
    st.markdown("Compare Genie Space and Multi-Agent Supervisor performance")

    # Show user context
    user_ctx = get_user_context()
    if user_ctx["email"]:
        st.sidebar.success(f"👤 Logged in as: {user_ctx['email']}")
    else:
        st.sidebar.warning("⚠️ Running without user context (local mode)")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Run Benchmark",
        "📈 Results",
        "🔧 Genie Spaces",
        "🤖 MAS Endpoints",
        "📋 History"
    ])

    with tab1:
        render_run_tab()

    with tab2:
        render_results_tab()

    with tab3:
        render_genie_spaces_tab()

    with tab4:
        render_mas_endpoints_tab()

    with tab5:
        render_history_tab()


def render_genie_spaces_tab():
    """Render the Genie Spaces management tab."""
    st.header("🔧 Genie Spaces")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🔄 Refresh Genie Spaces", use_container_width=True):
            with st.spinner("Loading Genie Spaces..."):
                st.session_state.genie_spaces = load_genie_spaces()

    # Load spaces if not cached
    if st.session_state.genie_spaces is None:
        with st.spinner("Loading Genie Spaces..."):
            st.session_state.genie_spaces = load_genie_spaces()

    spaces = st.session_state.genie_spaces

    if not spaces:
        st.info("No Genie Spaces found or unable to load. You can manually enter Space IDs in the Run Benchmark tab.")
    else:
        st.success(f"Found {len(spaces)} Genie Space(s)")

        # Display as table
        space_data = []
        for space in spaces:
            space_data.append({
                "Space ID": space.space_id,
                "Name": space.name,
                "Description": space.description or "-",
                "Warehouse ID": space.warehouse_id or "-",
            })

        df = pd.DataFrame(space_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Quick copy buttons
        st.subheader("Quick Copy")
        for space in spaces:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"{space.name}: {space.space_id}")
            with col2:
                if st.button("📋 Copy", key=f"copy_{space.space_id}"):
                    st.write(f"Space ID: `{space.space_id}`")

    # Manual validation
    st.divider()
    st.subheader("Validate Space ID")

    validate_id = st.text_input("Enter Space ID to validate")
    if st.button("Validate") and validate_id:
        token = get_fallback_token()
        host = get_config_value("DATABRICKS_HOST")

        if token and host:
            client = WorkspaceResourceClient(host, token)
            result = client.validate_genie_space(validate_id)
            client.close()

            if result["valid"]:
                st.success(f"✅ Valid Space: {result['name']}")
            else:
                st.error(f"❌ Invalid: {result['error']}")


def render_mas_endpoints_tab():
    """Render the MAS Endpoints management tab."""
    st.header("🤖 MAS Endpoints")

    registry = st.session_state.mas_registry

    # Add new endpoint
    st.subheader("Add New Endpoint")

    col1, col2 = st.columns(2)
    with col1:
        new_name = st.text_input("Endpoint Name", placeholder="production-mas")
    with col2:
        new_url = st.text_input("Endpoint URL", placeholder="http://localhost:8000")

    new_description = st.text_input("Description (optional)", placeholder="Production MAS instance")

    if st.button("➕ Add Endpoint", type="primary"):
        if new_name and new_url:
            endpoint = MASEndpoint(
                name=new_name,
                url=new_url,
                description=new_description or None
            )
            registry.add_endpoint(endpoint)
            st.success(f"Added endpoint: {new_name}")
            st.rerun()
        else:
            st.error("Name and URL are required")

    # List existing endpoints
    st.divider()
    st.subheader("Registered Endpoints")

    endpoints = registry.list_endpoints()

    if not endpoints:
        st.info("No MAS endpoints registered. Add one above or enter URL directly in Run Benchmark tab.")
    else:
        for endpoint in endpoints:
            with st.expander(f"📡 {endpoint.name}", expanded=True):
                st.text(f"URL: {endpoint.url}")
                if endpoint.description:
                    st.text(f"Description: {endpoint.description}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("🔍 Health Check", key=f"health_{endpoint.name}"):
                        with st.spinner("Checking..."):
                            result = registry.health_check(endpoint)
                            if result["healthy"]:
                                st.success(f"✅ Healthy ({result['latency_ms']:.0f}ms)")
                            else:
                                st.error(f"❌ {result['error']}")

                with col2:
                    active_status = "Active" if endpoint.is_active else "Inactive"
                    st.text(f"Status: {active_status}")

                with col3:
                    if st.button("🗑️ Remove", key=f"remove_{endpoint.name}"):
                        registry.remove_endpoint(endpoint.name)
                        st.rerun()

    # Health check all
    if endpoints:
        st.divider()
        if st.button("🔍 Health Check All"):
            with st.spinner("Checking all endpoints..."):
                results = registry.health_check_all()
                for name, status in results.items():
                    if status["healthy"]:
                        st.success(f"✅ {name}: Healthy ({status['latency_ms']:.0f}ms)")
                    else:
                        st.error(f"❌ {name}: {status['error']}")


def render_run_tab():
    """Render the benchmark execution tab."""

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Genie Space selection
        st.subheader("Genie Spaces")

        spaces = st.session_state.genie_spaces or []
        space_options = {s.name: s.space_id for s in spaces} if spaces else {}

        # Untuned Space
        if space_options:
            untuned_selection = st.selectbox(
                "Untuned Space",
                options=["Manual Entry"] + list(space_options.keys()),
                key="untuned_select"
            )
            if untuned_selection == "Manual Entry":
                genie_untuned_id = st.text_input(
                    "Untuned Space ID",
                    value=get_config_value("GENIE_UNTUNED_SPACE_ID"),
                    type="password",
                    key="untuned_manual"
                )
            else:
                genie_untuned_id = space_options[untuned_selection]
                st.text(f"ID: {genie_untuned_id[:12]}...")
        else:
            genie_untuned_id = st.text_input(
                "Untuned Space ID",
                value=get_config_value("GENIE_UNTUNED_SPACE_ID"),
                type="password"
            )

        # Tuned Space
        if space_options:
            tuned_selection = st.selectbox(
                "Tuned Space",
                options=["Manual Entry"] + list(space_options.keys()),
                key="tuned_select"
            )
            if tuned_selection == "Manual Entry":
                genie_tuned_id = st.text_input(
                    "Tuned Space ID",
                    value=get_config_value("GENIE_TUNED_SPACE_ID"),
                    type="password",
                    key="tuned_manual"
                )
            else:
                genie_tuned_id = space_options[tuned_selection]
                st.text(f"ID: {genie_tuned_id[:12]}...")
        else:
            genie_tuned_id = st.text_input(
                "Tuned Space ID",
                value=get_config_value("GENIE_TUNED_SPACE_ID"),
                type="password"
            )

        # MAS endpoint selection
        st.subheader("Multi-Agent Supervisor")

        registry = st.session_state.mas_registry
        mas_endpoints = registry.list_endpoints(active_only=True)
        mas_options = {e.name: e.url for e in mas_endpoints} if mas_endpoints else {}

        if mas_options:
            mas_selection = st.selectbox(
                "MAS Endpoint",
                options=["Manual Entry"] + list(mas_options.keys()),
                key="mas_select"
            )
            if mas_selection == "Manual Entry":
                mas_endpoint = st.text_input(
                    "MAS URL",
                    value=get_config_value("MAS_ENDPOINT", "http://localhost:8000"),
                    key="mas_manual"
                )
            else:
                mas_endpoint = mas_options[mas_selection]
                st.text(f"URL: {mas_endpoint}")
        else:
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

    # Main content
    st.header("📤 Upload Benchmarks")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload benchmark JSON file",
        type=["json"],
        help="JSON array with objects containing: id, question, expected_sql (+ optional: korean_question, category, difficulty)"
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
            st.session_state.benchmarks = benchmarks
            st.success(f"✅ Loaded {len(benchmarks)} benchmarks from file")
        except Exception as e:
            st.error(f"Error parsing file: {e}")
    elif json_input.strip():
        try:
            benchmarks = json.loads(json_input)
            st.session_state.benchmarks = benchmarks
            st.success(f"✅ Loaded {len(benchmarks)} benchmarks from input")
        except Exception as e:
            st.error(f"Error parsing JSON: {e}")
    elif st.session_state.benchmarks:
        benchmarks = st.session_state.benchmarks
        st.info(f"Using {len(benchmarks)} previously loaded benchmarks")

    # Preview and filter benchmarks
    if benchmarks:
        st.subheader("📋 Benchmark Preview")

        # Get unique categories and difficulties
        categories = sorted(set(b.get("category", "unknown") for b in benchmarks))
        difficulties = sorted(set(b.get("difficulty", "unknown") for b in benchmarks))

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_categories = st.multiselect(
                "Filter by Category",
                options=categories,
                default=categories
            )

        with col2:
            selected_difficulties = st.multiselect(
                "Filter by Difficulty",
                options=difficulties,
                default=difficulties
            )

        with col3:
            st.metric("Total Benchmarks", len(benchmarks))

        # Filter benchmarks
        filtered_benchmarks = [
            b for b in benchmarks
            if b.get("category", "unknown") in selected_categories
            and b.get("difficulty", "unknown") in selected_difficulties
        ]

        st.metric("Selected Benchmarks", len(filtered_benchmarks))

        # Display preview
        with st.expander(f"Preview ({len(filtered_benchmarks)} questions)", expanded=False):
            preview_data = []
            for b in filtered_benchmarks:
                preview_data.append({
                    "ID": b.get("id", ""),
                    "Question": b.get("korean_question", b.get("question", ""))[:100] + "...",
                    "Category": b.get("category", "-"),
                    "Difficulty": b.get("difficulty", "-"),
                })

            if preview_data:
                st.dataframe(
                    pd.DataFrame(preview_data),
                    use_container_width=True,
                    hide_index=True
                )

        # Use filtered benchmarks
        benchmarks = filtered_benchmarks

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
            systems_config={
                "genie_untuned": test_genie_untuned,
                "genie_tuned": test_genie_tuned,
                "mas_untuned": test_mas_untuned,
                "mas_tuned": test_mas_tuned,
            },
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
