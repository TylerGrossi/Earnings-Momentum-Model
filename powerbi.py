import streamlit as st


def render_powerbi_tab():
    """Render the PowerBI tab."""
    
    st.markdown("""
    <style>
        .powerbi-container {
            position: relative;
            width: 100%;
            height: 85vh;
            overflow: hidden;
            background: #0f172a;
            border-radius: 8px;
            border: 1px solid #334155;
            margin-bottom: 1rem;
        }
        .powerbi-container iframe {
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
    <div class="powerbi-container">
        <iframe 
            title="Finance Models"
            src="https://app.powerbi.com/view?r=eyJrIjoiZWRlNGNjYTgtODNhYy00MjBjLThhMjctMzgyNmYzNzIwZGRiIiwidCI6IjhkMWE2OWVjLTAzYjUtNDM0NS1hZTIxLWRhZDExMmY1ZmI0ZiIsImMiOjN9" 
            allowFullScreen="true">
        </iframe>
    </div>
    <a class="fullscreen-link" href="https://app.powerbi.com/view?r=eyJrIjoiZWRlNGNjYTgtODNhYy00MjBjLThhMjctMzgyNmYzNzIwZGRiIiwidCI6IjhkMWE2OWVjLTAzYjUtNDM0NS1hZTIxLWRhZDExMmY1ZmI0ZiIsImMiOjN9" target="_blank">Open in Full Screen ↗</a>
    """, unsafe_allow_html=True)