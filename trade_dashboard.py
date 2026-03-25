import streamlit as st


TRADE_DASHBOARD_URL = "https://trade-dashboard-c7dm.onrender.com/"


def render_trade_dashboard_tab():
    """Render the trade dashboard embed."""
    st.markdown(
        """
        <style>
            .trade-dashboard-container {
                position: relative;
                width: 100%;
                height: 90vh;
                overflow: hidden;
                background: #0f172a;
                border-radius: 8px;
                border: 1px solid #334155;
                margin-bottom: 0.5rem;
            }
            .trade-dashboard-container iframe {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                border: none;
                border-radius: 8px;
            }
            .fullscreen-link {
                display: inline-block;
                margin-top: 0.5rem;
                color: #60a5fa;
                text-decoration: none;
            }
            .fullscreen-link:hover {
                text-decoration: underline;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="trade-dashboard-container">
            <iframe
                title="Trade Dashboard"
                src="{TRADE_DASHBOARD_URL}"
                allowFullScreen="true"
            ></iframe>
        </div>
        <a class="fullscreen-link" href="{TRADE_DASHBOARD_URL}" target="_blank" rel="noopener noreferrer">
            Open Trade Dashboard in a new tab ↗
        </a>
        """,
        unsafe_allow_html=True,
    )

