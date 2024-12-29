import streamlit as st

# Welcome page
st.set_page_config(layout="wide", page_title="Football Analysis Dashboard", page_icon="⚽")
st.title("Welcome to the SDSA Football Analysis Dashboard")
logo_path = "assets/logo.png"
st.image(logo_path, use_container_width =True)
# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Welcome", "Match Report", "Team Report", "Player in Match", "Player Report", "Stat Finder"]
)

if page == "Welcome":
    st.write("Select a page from the sidebar.")
elif page == "Match Report":
    # Dynamically load the match report module
    from pages.match_report import app as match_report_app
    match_report_app()
elif page == "Team Report":
    from pages.team_report import app as team_report_app
    team_report_app()
elif page == "Player in Match":
    from pages.player_in_match import app as player_in_match_app
    player_in_match_app()
elif page == "Player Report":
    from pages.player_report import app as player_report_app
    player_report_app()
elif page == "Stat Finder":
    from pages.stat_finder import app as stat_finder_app
    stat_finder_app()
