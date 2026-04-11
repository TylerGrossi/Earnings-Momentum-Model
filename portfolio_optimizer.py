import streamlit as st


PORTFOLIO_OPTIMIZER_URL = "https://tylergrossi.shinyapps.io/Portfolio-Optimizer/"


def render_portfolio_optimizer_tab():
    """Render the portfolio optimizer embed."""
    st.markdown(
        """
        <style>
            .portfolio-optimizer-container {
                position: relative;
                width: 100%;
                height: 90vh;
                overflow: hidden;
                background: #0f172a;
                border-radius: 8px;
                border: 1px solid #334155;
                margin-bottom: 0.5rem;
            }
            .portfolio-optimizer-container iframe {
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
        <div class="portfolio-optimizer-container">
            <iframe
                title="Portfolio Optimizer"
                src="{PORTFOLIO_OPTIMIZER_URL}"
                allowFullScreen="true"
            ></iframe>
        </div>
        <a class="fullscreen-link" href="{PORTFOLIO_OPTIMIZER_URL}" target="_blank" rel="noopener noreferrer">
            Open Portfolio Optimizer in a new tab ↗
        </a>
        """,
        unsafe_allow_html=True,
    )
