"""
MCP Tab for Streamlit App
=========================
Provides UI to test MCP tools and view server configuration.
"""

import streamlit as st
import json
import subprocess
import sys
import os

# Import MCP server functions for direct testing
try:
    from mcp_server import (
        TOOLS,
        execute_tool,
        get_strategy_rules,
        get_earnings_this_week,
        get_stock_details,
        get_strategy_performance,
        run_backtest,
        get_beat_miss_analysis,
        compare_stop_losses,
        MCP_AVAILABLE
    )
    MCP_IMPORT_SUCCESS = True
except ImportError as e:
    MCP_IMPORT_SUCCESS = False
    MCP_IMPORT_ERROR = str(e)


def render_mcp_tab():
    """Render the MCP (Model Context Protocol) tab."""
    
    st.subheader("🤖 MCP Server - AI Integration")
    
    st.markdown("""
    This tab allows you to interact with the **Model Context Protocol (MCP) server** for the 
    Earnings Momentum Strategy. MCP enables AI assistants like Claude to query your strategy 
    data and run analyses in real-time.
    """)
    
    st.markdown("---")
    
    # Status Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if MCP_IMPORT_SUCCESS:
            st.success("✅ MCP Module Loaded")
        else:
            st.error(f"❌ Import Error: {MCP_IMPORT_ERROR}")
    
    with col2:
        if MCP_IMPORT_SUCCESS and MCP_AVAILABLE:
            st.success("✅ MCP SDK Installed")
        else:
            st.warning("⚠️ MCP SDK Not Installed")
            st.caption("Install with: `pip install mcp`")
    
    with col3:
        st.info(f"📊 {len(TOOLS) if MCP_IMPORT_SUCCESS else 0} Tools Available")
    
    st.markdown("---")
    
    # Tabs for different sections
    mcp_tab1, mcp_tab2, mcp_tab3, mcp_tab4 = st.tabs([
        "🧪 Test Tools", 
        "📋 Available Tools", 
        "⚙️ Server Config",
        "💬 Example Queries"
    ])
    
    # ==========================================
    # TAB 1: Test Tools
    # ==========================================
    with mcp_tab1:
        st.markdown("### Test MCP Tools")
        st.markdown("Run tools directly to see what data AI assistants can access.")
        
        if not MCP_IMPORT_SUCCESS:
            st.error("Cannot test tools - MCP module not loaded.")
            return
        
        # Tool selector
        tool_names = [t["name"] for t in TOOLS]
        selected_tool = st.selectbox(
            "Select Tool:",
            tool_names,
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        # Get tool info
        tool_info = next((t for t in TOOLS if t["name"] == selected_tool), None)
        
        if tool_info:
            st.caption(tool_info["description"])
            
            # Build arguments UI based on tool schema
            arguments = {}
            props = tool_info["input_schema"].get("properties", {})
            
            if props:
                st.markdown("**Parameters:**")
                
                for prop_name, prop_info in props.items():
                    prop_type = prop_info.get("type", "string")
                    prop_desc = prop_info.get("description", "")
                    
                    if prop_type == "number":
                        if prop_name == "stop_loss_pct":
                            arguments[prop_name] = st.slider(
                                f"{prop_name}",
                                min_value=-50,
                                max_value=-1,
                                value=-10,
                                help=prop_desc
                            )
                        else:
                            arguments[prop_name] = st.number_input(
                                f"{prop_name}",
                                value=0.0,
                                help=prop_desc
                            )
                    elif prop_type == "integer":
                        if prop_name == "holding_days":
                            arguments[prop_name] = st.slider(
                                f"{prop_name}",
                                min_value=1,
                                max_value=10,
                                value=5,
                                help=prop_desc
                            )
                        else:
                            arguments[prop_name] = st.number_input(
                                f"{prop_name}",
                                value=0,
                                step=1,
                                help=prop_desc
                            )
                    else:  # string
                        if prop_name == "ticker":
                            arguments[prop_name] = st.text_input(
                                f"{prop_name}",
                                value="AAPL",
                                help=prop_desc
                            ).upper()
                        else:
                            arguments[prop_name] = st.text_input(
                                f"{prop_name}",
                                help=prop_desc
                            )
            
            # Execute button
            if st.button("🚀 Run Tool", type="primary", width="stretch"):
                with st.spinner("Executing..."):
                    result = execute_tool(selected_tool, arguments)
                    
                    try:
                        result_json = json.loads(result)
                        
                        # Pretty display based on tool type
                        if "error" in result_json:
                            st.error(f"Error: {result_json['error']}")
                        else:
                            _display_tool_result(selected_tool, result_json)
                        
                        # Raw JSON expander
                        with st.expander("📄 View Raw JSON Response"):
                            st.json(result_json)
                            
                    except json.JSONDecodeError:
                        st.code(result, language="json")
    
    # ==========================================
    # TAB 2: Available Tools
    # ==========================================
    with mcp_tab2:
        st.markdown("### Available MCP Tools")
        st.markdown("These tools can be called by AI assistants connected to the MCP server.")
        
        if MCP_IMPORT_SUCCESS:
            for i, tool in enumerate(TOOLS):
                with st.expander(f"**{tool['name']}**", expanded=(i == 0)):
                    st.markdown(f"**Description:** {tool['description']}")
                    
                    props = tool["input_schema"].get("properties", {})
                    required = tool["input_schema"].get("required", [])
                    
                    if props:
                        st.markdown("**Parameters:**")
                        for prop_name, prop_info in props.items():
                            req_badge = "🔴 required" if prop_name in required else "⚪ optional"
                            st.markdown(f"- `{prop_name}` ({prop_info.get('type', 'any')}) - {prop_info.get('description', '')} {req_badge}")
                    else:
                        st.caption("No parameters required")
                    
                    # Example usage
                    st.markdown("**Example Call:**")
                    example_args = {}
                    if "ticker" in props:
                        example_args["ticker"] = "AAPL"
                    if "stop_loss_pct" in props:
                        example_args["stop_loss_pct"] = -10
                    if "holding_days" in props:
                        example_args["holding_days"] = 5
                    
                    st.code(f'{{"tool": "{tool["name"]}", "arguments": {json.dumps(example_args)}}}', language="json")
        else:
            st.warning("Tools not available - MCP module not loaded.")
    
    # ==========================================
    # TAB 3: Server Configuration
    # ==========================================
    with mcp_tab3:
        st.markdown("### MCP Server Configuration")
        
        st.markdown("#### Claude Desktop Integration")
        st.markdown("""
        To use this MCP server with Claude Desktop, add the following to your 
        `claude_desktop_config.json` file:
        """)
        
        # Get the path to the mcp_server.py
        server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "mcp_server.py"))
        
        config = {
            "mcpServers": {
                "earnings-momentum": {
                    "command": "python",
                    "args": [server_path],
                    "env": {}
                }
            }
        }
        
        st.code(json.dumps(config, indent=2), language="json")
        
        st.markdown("#### Config File Locations")
        st.markdown("""
        | Platform | Location |
        |----------|----------|
        | macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
        | Windows | `%APPDATA%\\Claude\\claude_desktop_config.json` |
        | Linux | `~/.config/Claude/claude_desktop_config.json` |
        """)
        
        st.markdown("---")
        
        st.markdown("#### Install MCP SDK")
        st.code("pip install mcp", language="bash")
        
        st.markdown("#### Test Server Locally")
        st.code("python mcp_server.py --test", language="bash")
        
        st.markdown("#### Run Server (stdio mode)")
        st.code("python mcp_server.py", language="bash")
    
    # ==========================================
    # TAB 4: Example Queries
    # ==========================================
    with mcp_tab4:
        st.markdown("### Example AI Queries")
        st.markdown("""
        Once connected, you can ask Claude questions like these and it will use 
        the MCP tools to get real-time data from your strategy:
        """)
        
        examples = [
            {
                "query": "What stocks are signaling for earnings this week?",
                "tool": "get_earnings_this_week",
                "description": "Lists all stocks in the tracker with earnings this week"
            },
            {
                "query": "Run a backtest with a 15% stop loss and 3-day holding period",
                "tool": "run_backtest",
                "description": "Executes backtest with custom parameters"
            },
            {
                "query": "How does the strategy perform when stocks beat earnings vs miss?",
                "tool": "get_beat_miss_analysis", 
                "description": "Compares returns for earnings beats vs misses"
            },
            {
                "query": "What's the optimal stop loss level for this strategy?",
                "tool": "compare_stop_losses",
                "description": "Compares all stop loss levels from -2% to -20%"
            },
            {
                "query": "Give me details on NVDA from the tracker",
                "tool": "get_stock_details",
                "description": "Gets full details for a specific ticker"
            },
            {
                "query": "What are the overall strategy performance metrics?",
                "tool": "get_strategy_performance",
                "description": "Returns total returns, win rate, sector breakdown"
            },
            {
                "query": "Explain the entry and exit rules of the strategy",
                "tool": "get_strategy_rules",
                "description": "Returns strategy documentation"
            }
        ]
        
        for ex in examples:
            with st.container():
                st.markdown(f"**💬 \"{ex['query']}\"**")
                st.caption(f"→ Uses `{ex['tool']}` - {ex['description']}")
                st.markdown("")


def _display_tool_result(tool_name: str, result: dict):
    """Display tool results in a user-friendly format."""
    
    if tool_name == "get_strategy_rules":
        st.markdown("### Strategy Rules")
        
        st.markdown("**Entry Criteria:**")
        for key, val in result.get("entry_criteria", {}).items():
            st.markdown(f"- {key.replace('_', ' ').title()}: {val}")
        
        st.markdown("**Entry Timing:**")
        for key, val in result.get("entry_timing", {}).items():
            st.markdown(f"- **{key}**: {val}")
        
        st.markdown("**Exit Rules:**")
        for key, val in result.get("exit_rules", {}).items():
            st.markdown(f"- {key.replace('_', ' ').title()}: {val}")
    
    elif tool_name == "get_earnings_this_week":
        st.markdown(f"### Earnings This Week")
        st.caption(f"Week: {result.get('week_range', 'N/A')}")
        st.metric("Total Stocks", result.get("count", 0))
        
        stocks = result.get("stocks", [])
        if stocks:
            import pandas as pd
            df = pd.DataFrame(stocks)
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info("No earnings this week in tracker.")
    
    elif tool_name == "get_stock_details":
        st.markdown(f"### {result.get('ticker', 'N/A')} - {result.get('company_name', 'N/A')}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sector", result.get("sector", "N/A"))
        with col2:
            st.metric("Market Cap", result.get("market_cap", "N/A"))
        with col3:
            returns = result.get("returns", {})
            st.metric("5D Return", f"{returns.get('5d', 0)*100:.1f}%" if returns.get('5d') else "N/A")
        
        st.markdown("**Earnings Info:**")
        earnings = result.get("earnings", {})
        st.write(f"- Date: {earnings.get('date', 'N/A')}")
        st.write(f"- Timing: {earnings.get('timing', 'N/A')}")
        st.write(f"- EPS Estimate: {earnings.get('eps_estimate', 'N/A')}")
        st.write(f"- Reported EPS: {earnings.get('reported_eps', 'N/A')}")
        st.write(f"- EPS Surprise: {earnings.get('eps_surprise_pct', 'N/A')}%")
    
    elif tool_name == "get_strategy_performance":
        st.markdown("### Strategy Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", result.get("total_trades", 0))
        with col2:
            returns = result.get("returns_5d", {})
            st.metric("Total Return", f"{returns.get('total', 0):.1f}%")
        with col3:
            st.metric("Avg Return", f"{returns.get('average', 0):.2f}%")
        with col4:
            st.metric("Win Rate", f"{result.get('win_rate_pct', 0):.1f}%")
        
        # Sector breakdown
        sector_stats = result.get("sector_breakdown", {})
        if sector_stats:
            st.markdown("**Sector Breakdown:**")
            import pandas as pd
            sector_df = pd.DataFrame([
                {"Sector": k, **v} for k, v in sector_stats.items()
            ]).sort_values("avg_return", ascending=False)
            st.dataframe(sector_df, width="stretch", hide_index=True)
    
    elif tool_name == "run_backtest":
        st.markdown("### Backtest Results")
        
        params = result.get("parameters", {})
        st.caption(f"Stop Loss: {params.get('stop_loss_pct')}% | Holding Days: {params.get('holding_days')}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", result.get("total_trades", 0))
        with col2:
            normal = result.get("normal_model", {})
            st.metric("Normal 5D Return", f"{normal.get('total_return_pct', 0):.1f}%")
        with col3:
            strategy = result.get("strategy_model", {})
            st.metric("Strategy Return", f"{strategy.get('total_return_pct', 0):.1f}%")
        with col4:
            st.metric("Alpha", f"{strategy.get('alpha_pct', 0):+.1f}%")
        
        # Exit breakdown
        exit_breakdown = result.get("exit_breakdown", {})
        if exit_breakdown:
            st.markdown("**Exit Breakdown:**")
            for reason, count in exit_breakdown.items():
                st.write(f"- {reason.replace('_', ' ').title()}: {count}")
    
    elif tool_name == "get_beat_miss_analysis":
        st.markdown("### Beat vs Miss Analysis")
        
        st.metric("Trades with EPS Data", result.get("total_with_eps_data", 0))
        
        col1, col2, col3 = st.columns(3)
        
        beat = result.get("beat", {})
        miss = result.get("miss", {})
        
        with col1:
            if beat:
                st.markdown("**Beats:**")
                st.metric("Count", beat.get("count", 0))
                st.metric("Avg Return", f"{beat.get('avg_return_pct', 0):.2f}%")
                st.metric("Win Rate", f"{beat.get('win_rate_pct', 0):.1f}%")
        
        with col2:
            if miss:
                st.markdown("**Misses:**")
                st.metric("Count", miss.get("count", 0))
                st.metric("Avg Return", f"{miss.get('avg_return_pct', 0):.2f}%")
                st.metric("Win Rate", f"{miss.get('win_rate_pct', 0):.1f}%")
        
        with col3:
            if beat and miss:
                spread = beat.get("avg_return_pct", 0) - miss.get("avg_return_pct", 0)
                st.markdown("**Spread:**")
                st.metric("Beat - Miss", f"{spread:+.2f}%")
    
    elif tool_name == "compare_stop_losses":
        st.markdown("### Stop Loss Comparison")
        
        st.success(f"**Best Stop Loss:** {result.get('best_stop_loss_pct')}% → {result.get('best_total_return_pct'):.1f}% total return")
        
        comparison = result.get("comparison", [])
        if comparison:
            import pandas as pd
            df = pd.DataFrame(comparison)
            df.columns = ["Stop Loss %", "Total Return %", "Alpha %", "Win Rate %"]
            st.dataframe(df, width="stretch", hide_index=True)
    
    else:
        # Default: just show the result
        st.json(result)