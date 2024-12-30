import streamlit as st

# Welcome page
st.set_page_config(layout="wide", page_title="Football Analysis Dashboard", page_icon="⚽")
st.title("Welcome to the SDSA Football Analysis Dashboard")

# Display the logo
logo_path = "assets/logo.png"
st.image(logo_path, use_container_width=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Welcome", "Match Report", "Team Report", "Player in Match", "Player Report", "Stat Finder"]
)

if page == "Welcome":
    st.write("Select a page from the sidebar.")

elif page == "Match Report":
    with st.expander("Match Report Details", expanded=True):
        # Dynamically load the match report module
        from pages.match_report import app as match_report_app
        match_report_app()

elif page == "Team Report":
    with st.expander("Team Report Details", expanded=True):
        from pages.team_report import app as team_report_app
        team_report_app()

elif page == "Player in Match":
    with st.expander("Player in Match Details", expanded=True):
        from pages.player_in_match import app as player_in_match_app
        player_in_match_app()

elif page == "Player Report":
    with st.expander("Player Report Details", expanded=True):
        from pages.player_report import app as player_report_app
        player_report_app()

elif page == "Stat Finder":
    with st.expander("Stat Finder Details", expanded=True):
        from pages.stat_finder import app as stat_finder_app
        stat_finder_app()
