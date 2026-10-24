"""Tab: map of schools, colored by the user's current score."""

import plotly.express as px
import streamlit as st

import plots

# Light-to-dark blue: darker = higher score
BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def render(table):
    st.subheader("Map")
    st.write("Darker dots score higher with your weights. Bigger dots have more undergrads.")
    # scatter_geo draws the map itself: no internet map tiles or WebGL needed
    d = table.dropna(subset=["lat", "lon", "undergrads"])
    fig = px.scatter_geo(
        d, lat="lat", lon="lon", color="score", size="undergrads", size_max=20,
        color_continuous_scale=BLUES, range_color=(0, 100), hover_name="name",
        hover_data={"lat": False, "lon": False, "undergrads": ":,", "score": ":.0f", "your_rank": True},
        scope="usa", height=600,
    )
    fig.update_traces(marker=dict(line=dict(color=plots.SURFACE, width=0.5)))
    fig.update_geos(bgcolor=plots.SURFACE, landcolor="#f0efec", subunitcolor="#c3c2b7")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title="Score"),
                      paper_bgcolor=plots.SURFACE)
    st.plotly_chart(fig, width="stretch", theme=None)
    st.caption("Schools outside the 50 states + DC (e.g. Puerto Rico) are not shown on this map.")
