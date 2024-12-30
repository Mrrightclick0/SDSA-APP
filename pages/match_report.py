import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
import matplotlib.patches as patches
from mplsoccer import Pitch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from highlight_text import ax_text
import os
from unidecode import unidecode
from scipy.spatial import ConvexHull
import streamlit as st
def app():
    st.title("Match Report")
    # Print the modified DataFrame
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    green = '#69f900'
    red = '#ff4b44'
    blue = '#00a0de'
    violet = '#a369ff'
    bg_color= '#ffffff'
    line_color= '#000000'
    #bg_color= '#000000'
    #line_color= '#ffffff'
    col1 = '#ff4b44'
    col2 = '#00a0de'
    hcol=col1
    acol=col2

    # Paths to data directories
    event_data_dir = os.path.join("Stat")
    shot_data_dir = os.path.join("Stat")

    # Function to list files
    def list_files(directory, keyword):
        return [f for f in os.listdir(directory) if keyword in f]

    # Extract unique match names from file names
    def extract_match_names(event_files, shot_files):
        matches = set(f.split("_EventsData.csv")[0] for f in event_files) & set(f.split("_ShotsData.csv")[0] for f in shot_files)
        return sorted(matches)

    # List available CSV files
    event_files = list_files(event_data_dir, "_EventsData.csv")
    shot_files = list_files(shot_data_dir, "_ShotsData.csv")

    # Extract match names
    match_names = extract_match_names(event_files, shot_files)

    # Extract unique teams from event files
    def extract_teams_from_files(event_files, event_data_dir):
        teams = set()
        for file in event_files:
            df_temp = pd.read_csv(os.path.join(event_data_dir, file))
            if 'teamName' in df_temp.columns:
                teams.update(df_temp['teamName'].unique())
        return sorted(teams)

    teams = extract_teams_from_files(event_files, event_data_dir)
    # Streamlit Layout

    # Sidebar for Team and Match Selection
    with st.sidebar:
        st.title("Football Match Report Dashboard")
        
        # Team Selection
        selected_team = st.selectbox("Select a Team", [""] + teams, index=0)

        # Filter matches for the selected team
        filtered_matches = []
        if selected_team:
            for match in match_names:
                if selected_team in match:
                    filtered_matches.append(match)

        # Match Selection
        selected_match = st.selectbox("Select a Match", [""] + filtered_matches, index=0)
        
        # Proceed Button
        proceed = st.button("Go")

    # Initialize variables to avoid NameError
    df = None
    shots_df = None

    if proceed and selected_match:
        # Automatically select corresponding event and shot data files
        selected_event_file = f"{selected_match}_EventsData.csv"
        selected_shot_file = f"{selected_match}_ShotsData.csv"

        try:
            # Load the selected files
            df = pd.read_csv(os.path.join(event_data_dir, selected_event_file))
            shots_df = pd.read_csv(os.path.join(shot_data_dir, selected_shot_file))
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Event Data as CSV",
                data=csv,
                file_name=f"{selected_match}_EventsData.csv",
                mime="text/csv",
            )

            # Remove rows with missing names or team names
            df = df.dropna(subset=['name', 'teamName'])

            # Ensure required columns are present
            required_columns = ['teamName', 'oppositionTeamName']
            if not all(col in df.columns for col in required_columns):
                st.error(f"The event data file for {selected_match} does not contain the required columns: {required_columns}")
            else:
                # Extract home and away team names
                hteamName = df.iloc[0]['teamName']
                ateamName = df.iloc[0]['oppositionTeamName']

                # Extract player information
                columns_to_extract = ['playerId', 'shirtNo', 'name', 'position', 'isFirstEleven', 'teamName', 'oppositionTeamName']
                players_df = df[columns_to_extract]
                players_df['shirtNo'] = players_df['shirtNo'].apply(lambda x: str(int(x)) if pd.notna(x) else 'nan')
                players_df = players_df.drop_duplicates(subset='playerId')

                # Team-specific data
                homedf = df[df['teamName'] == hteamName]
                awaydf = df[df['teamName'] == ateamName]

                # Metrics
                hxT = homedf['xT'].sum().round(2)
                axT = awaydf['xT'].sum().round(2)

                hgoal_count = len(homedf[(homedf['teamName'] == hteamName) & (homedf['type'] == 'Goal') & (~homedf['qualifiers'].str.contains('OwnGoal'))])
                agoal_count = len(awaydf[(awaydf['teamName'] == ateamName) & (awaydf['type'] == 'Goal') & (~awaydf['qualifiers'].str.contains('OwnGoal'))])
                hgoal_count += len(awaydf[(awaydf['teamName'] == ateamName) & (awaydf['type'] == 'Goal') & (awaydf['qualifiers'].str.contains('OwnGoal'))])
                agoal_count += len(homedf[(homedf['teamName'] == hteamName) & (homedf['type'] == 'Goal') & (homedf['qualifiers'].str.contains('OwnGoal'))])

                hshots_xgdf = shots_df[shots_df['teamName'] == hteamName]
                ashots_xgdf = shots_df[shots_df['teamName'] == ateamName]
                hxg = round(hshots_xgdf['expectedGoals'].sum(), 2)
                axg = round(ashots_xgdf['expectedGoals'].sum(), 2)
                hxgot = round(hshots_xgdf['expectedGoalsOnTarget'].sum(), 2)
                axgot = round(ashots_xgdf['expectedGoalsOnTarget'].sum(), 2)

                # Display Metrics
                st.header("Match Summary")
                st.write(f"**{hteamName}** xT: {hxT}, Goals: {hgoal_count}, xG: {hxg}, xGOT: {hxgot}")
                st.write(f"**{ateamName}** xT: {axT}, Goals: {agoal_count}, xG: {axg}, xGOT: {axgot}")
                # Display Player DataFrame
                st.header("Player Information")
                st.dataframe(players_df)
                def get_passes_df(df):
                    df1 = df[~df['type'].str.contains('SubstitutionOn|FormationChange|FormationSet|Card')]
                    df = df1
                    df.loc[:, "receiver"] = df["playerId"].shift(-1)
                    passes_ids = df.index[df['type'] == 'Pass']
                    df_passes = df.loc[passes_ids, ["index", "x", "y", "endX", "endY", "teamName", "playerId", "receiver", "type", "outcomeType", "pass_or_carry_angle"]]

                    return df_passes
                events_dict=df
                passes_df = get_passes_df(df)
                path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]

                def get_passes_between_df(teamName, passes_df, players_df):
                    passes_df = passes_df[(passes_df["teamName"] == teamName)]
                    # df = pd.DataFrame(events_dict)
                    dfteam = df[(df['teamName'] == teamName) & (~df['type'].str.contains('SubstitutionOn|FormationChange|FormationSet|Card'))]
                    passes_df = passes_df.merge(players_df[["playerId", "isFirstEleven"]], on='playerId', how='left')
                    # calculate median positions for player's passes
                    average_locs_and_count_df = (dfteam.groupby('playerId').agg({'x': ['median'], 'y': ['median', 'count']}))
                    average_locs_and_count_df.columns = ['pass_avg_x', 'pass_avg_y', 'count']
                    average_locs_and_count_df = average_locs_and_count_df.merge(players_df[['playerId', 'name', 'shirtNo', 'position', 'isFirstEleven']], on='playerId', how='left')
                    average_locs_and_count_df = average_locs_and_count_df.set_index('playerId')
                    average_locs_and_count_df['name'] = average_locs_and_count_df['name'].apply(unidecode)
                    # calculate the number of passes between each position (using min/ max so we get passes both ways)
                    passes_player_ids_df = passes_df.loc[:, ['index', 'playerId', 'receiver', 'teamName']]
                    passes_player_ids_df['pos_max'] = (passes_player_ids_df[['playerId', 'receiver']].max(axis='columns'))
                    passes_player_ids_df['pos_min'] = (passes_player_ids_df[['playerId', 'receiver']].min(axis='columns'))
                    # get passes between each player
                    passes_between_df = passes_player_ids_df.groupby(['pos_min', 'pos_max']).index.count().reset_index()
                    passes_between_df.rename({'index': 'pass_count'}, axis='columns', inplace=True)
                    # add on the location of each player so we have the start and end positions of the lines
                    passes_between_df = passes_between_df.merge(average_locs_and_count_df, left_on='pos_min', right_index=True)
                    passes_between_df = passes_between_df.merge(average_locs_and_count_df, left_on='pos_max', right_index=True, suffixes=['', '_end'])

                    return passes_between_df, average_locs_and_count_df

                # home_team_id = list(teams_dict.keys())[0]
                home_passes_between_df, home_average_locs_and_count_df = get_passes_between_df(hteamName, passes_df, players_df)
                # away_team_id = list(teams_dict.keys())[1]
                away_passes_between_df, away_average_locs_and_count_df = get_passes_between_df(ateamName, passes_df, players_df)

                def pass_network_visualization(ax, passes_between_df, average_locs_and_count_df, col, teamName, flipped=False):
                    MAX_LINE_WIDTH = 15
                    MAX_MARKER_SIZE = 3000
                    passes_between_df['width'] = (passes_between_df.pass_count / passes_between_df.pass_count.max() *MAX_LINE_WIDTH)
                    # average_locs_and_count_df['marker_size'] = (average_locs_and_count_df['count']/ average_locs_and_count_df['count'].max() * MAX_MARKER_SIZE) #You can plot variable size of each player's node according to their passing volume, in the plot using this
                    MIN_TRANSPARENCY = 0.05
                    MAX_TRANSPARENCY = 0.85
                    color = np.array(to_rgba(col))
                    color = np.tile(color, (len(passes_between_df), 1))
                    c_transparency = passes_between_df.pass_count / passes_between_df.pass_count.max()
                    c_transparency = (c_transparency * (MAX_TRANSPARENCY - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
                    color[:, 3] = c_transparency

                    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    # ax.set_ylim(-0.5, 68.5)

                    # Plotting those lines between players
                    pass_lines = pitch.lines(passes_between_df.pass_avg_x, passes_between_df.pass_avg_y, passes_between_df.pass_avg_x_end, passes_between_df.pass_avg_y_end,
                                            lw=passes_between_df.width, color=color, zorder=1, ax=ax)

                    # Plotting the player nodes
                    for index, row in average_locs_and_count_df.iterrows():
                        if row['isFirstEleven'] == True:
                            pass_nodes = pitch.scatter(row['pass_avg_x'], row['pass_avg_y'], s=1000, marker='o', color=bg_color, edgecolor=line_color, linewidth=2, alpha=1, ax=ax)
                        else:
                            pass_nodes = pitch.scatter(row['pass_avg_x'], row['pass_avg_y'], s=1000, marker='s', color=bg_color, edgecolor=line_color, linewidth=2, alpha=0.75, ax=ax)

                    # Plotting the shirt no. of each player
                    for index, row in average_locs_and_count_df.iterrows():
                        player_initials = row["shirtNo"]
                        pitch.annotate(player_initials, xy=(row.pass_avg_x, row.pass_avg_y), c=col, ha='center', va='center', size=18, ax=ax)

                    # Plotting a vertical line to show the median vertical position of all passes
                    avgph = round(average_locs_and_count_df['pass_avg_x'].median(), 2)
                    # avgph_show = round((avgph*1.05),2)
                    avgph_show = avgph
                    ax.axvline(x=avgph, color='gray', linestyle='--', alpha=0.75, linewidth=2)

                    # Defense line Height
                    center_backs_height = average_locs_and_count_df[average_locs_and_count_df['position']=='DC']
                    def_line_h = round(center_backs_height['pass_avg_x'].median(), 2)
                    ax.axvline(x=def_line_h, color='gray', linestyle='dotted', alpha=0.5, linewidth=2)
                    # Forward line Height
                    Forwards_height = average_locs_and_count_df[average_locs_and_count_df['isFirstEleven']==1]
                    Forwards_height = Forwards_height.sort_values(by='pass_avg_x', ascending=False)
                    Forwards_height = Forwards_height.head(2)
                    fwd_line_h = round(Forwards_height['pass_avg_x'].mean(), 2)
                    ax.axvline(x=fwd_line_h, color='gray', linestyle='dotted', alpha=0.5, linewidth=2)
                    # coloring the middle zone in the pitch
                    ymid = [0, 0, 68, 68]
                    xmid = [def_line_h, fwd_line_h, fwd_line_h, def_line_h]
                    ax.fill(xmid, ymid, col, alpha=0.1)

                    team_passes_df = passes_df[(passes_df["teamName"] == teamName)]
                    team_passes_df['pass_or_carry_angle'] = team_passes_df['pass_or_carry_angle'].abs()
                    team_passes_df = team_passes_df[(team_passes_df['pass_or_carry_angle']>=0) & (team_passes_df['pass_or_carry_angle']<=90)]
                    
                    med_ang = team_passes_df['pass_or_carry_angle'].median()
                    verticality = round((1 - med_ang/90)*100, 2)

                    passes_between_df = passes_between_df.sort_values(by='pass_count', ascending=False).head(1).reset_index(drop=True)
                    most_pass_from = passes_between_df['name'][0]
                    most_pass_to = passes_between_df['name_end'][0]
                    most_pass_count = passes_between_df['pass_count'][0]
                    

                    if teamName == ateamName:
                    # inverting the pitch for away team
                        ax.invert_xaxis()
                        ax.invert_yaxis()
                        ax.text(avgph-1, 73, f"{avgph_show}m", fontsize=15, color=line_color, ha='left')
                        ax.text(105, 73,f"verticality: {verticality}%", fontsize=15, color=line_color, ha='left')
                    else:
                        ax.text(avgph-1, -5, f"{avgph_show}m", fontsize=15, color=line_color, ha='right')
                        ax.text(105, -5, f"verticality: {verticality}%", fontsize=15, color=line_color, ha='right')

                    # Headlines and other texts
                    if teamName == hteamName:
                        ax.text(2,66, "circle = starter\nbox = sub", color=hcol, size=12, ha='left', va='top')
                        ax.set_title(f"{hteamName}\nPassing Network", color=line_color, size=25, fontweight='bold')
                    # ax.text(52.5, 75, f"{hteamName}\nPassing Network", color=line_color, size=35, ha='center', va='center', fontweight='bold', fontproperties=custom_font)

                    else:
                        ax.text(2,2, "circle = starter\nbox = sub", color=acol, size=12, ha='right', va='top')
                        ax.set_title(f"{ateamName}\nPassing Network", color=line_color, size=25, fontweight='bold')

                    return {
                        'Team_Name': teamName,
                        'Defense_Line_Height': def_line_h,
                        'Vericality_%': verticality,
                        'Most_pass_combination_from': most_pass_from,
                        'Most_pass_combination_to': most_pass_to,
                        'Most_passes_in_combination': most_pass_count,
                    } 

                fig,axs=plt.subplots(1,2, figsize=(20,10), facecolor=bg_color)
                pass_network_stats_home = pass_network_visualization(axs[0], home_passes_between_df, home_average_locs_and_count_df, hcol, hteamName)
                pass_network_stats_away = pass_network_visualization(axs[1], away_passes_between_df, away_average_locs_and_count_df, acol, ateamName)
                pass_network_stats_list = [pass_network_stats_home, pass_network_stats_away]
                pass_network_stats_df = pd.DataFrame(pass_network_stats_list)

                
                    # Display the visualization
                st.header("Passing Network Statistics")
                st.pyplot(fig)

                    # Display the passing network stats
                st.dataframe(pass_network_stats_df)
                def get_defensive_action_df(events_dict):
                    defensive_actions_ids = df.index[(df['type'] == 'Aerial') & (df['qualifiers'].str.contains('Defensive')) |
                                                    (df['type'] == 'BallRecovery') |
                                                    (df['type'] == 'BlockedPass') |
                                                    (df['type'] == 'Challenge') |
                                                    (df['type'] == 'Clearance') |
                                                    (df['type'] == 'Error') |
                                                    (df['type'] == 'Foul') |
                                                    (df['type'] == 'Interception') |
                                                    (df['type'] == 'Tackle')]
                    df_defensive_actions = df.loc[defensive_actions_ids, ["index", "x", "y", "teamName", "playerId", "type", "outcomeType"]]

                    return df_defensive_actions

                defensive_actions_df = get_defensive_action_df(events_dict)

                def get_da_count_df(team_name, defensive_actions_df, players_df):
                    defensive_actions_df = defensive_actions_df[defensive_actions_df["teamName"] == team_name]
                    # add column with first eleven players only
                    defensive_actions_df = defensive_actions_df.merge(players_df[["playerId", "isFirstEleven"]], on='playerId', how='left')
                    # calculate mean positions for players
                    average_locs_and_count_df = (defensive_actions_df.groupby('playerId').agg({'x': ['median'], 'y': ['median', 'count']}))
                    average_locs_and_count_df.columns = ['x', 'y', 'count']
                    average_locs_and_count_df = average_locs_and_count_df.merge(players_df[['playerId', 'name', 'shirtNo', 'position', 'isFirstEleven']], on='playerId', how='left')
                    average_locs_and_count_df = average_locs_and_count_df.set_index('playerId')

                    return  average_locs_and_count_df

                defensive_home_average_locs_and_count_df = get_da_count_df(hteamName, defensive_actions_df, players_df)
                defensive_away_average_locs_and_count_df = get_da_count_df(ateamName, defensive_actions_df, players_df)
                defensive_home_average_locs_and_count_df = defensive_home_average_locs_and_count_df[defensive_home_average_locs_and_count_df['position'] != 'GK']
                defensive_away_average_locs_and_count_df = defensive_away_average_locs_and_count_df[defensive_away_average_locs_and_count_df['position'] != 'GK']

                def defensive_block(ax, average_locs_and_count_df, team_name, col):
                    defensive_actions_team_df = defensive_actions_df[defensive_actions_df["teamName"] == team_name]
                    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, line_zorder=2, corner_arcs=True)
                    pitch.draw(ax=ax)
                    ax.set_facecolor(bg_color)
                    ax.set_xlim(-0.5, 105.5)
                    # ax.set_ylim(-0.5, 68.5)

                    # using variable marker size for each player according to their defensive engagements
                    MAX_MARKER_SIZE = 3500
                    average_locs_and_count_df['marker_size'] = (average_locs_and_count_df['count']/ average_locs_and_count_df['count'].max() * MAX_MARKER_SIZE)
                    # plotting the heatmap of the team defensive actions
                    color = np.array(to_rgba(col))
                    flamingo_cmap = LinearSegmentedColormap.from_list("Flamingo - 100 colors", [bg_color, col], N=500)
                    kde = pitch.kdeplot(defensive_actions_team_df.x, defensive_actions_team_df.y, ax=ax, fill=True, levels=5000, thresh=0.02, cut=4, cmap=flamingo_cmap)

                    # using different node marker for starting and substitute players
                    average_locs_and_count_df = average_locs_and_count_df.reset_index(drop=True)
                    for index, row in average_locs_and_count_df.iterrows():
                        if row['isFirstEleven'] == True:
                            da_nodes = pitch.scatter(row['x'], row['y'], s=row['marker_size']+100, marker='o', color=bg_color, edgecolor=line_color, linewidth=1, 
                                                alpha=1, zorder=3, ax=ax)
                        else:
                            da_nodes = pitch.scatter(row['x'], row['y'], s=row['marker_size']+100, marker='s', color=bg_color, edgecolor=line_color, linewidth=1, 
                                                    alpha=1, zorder=3, ax=ax)
                    # plotting very tiny scatterings for the defensive actions
                    da_scatter = pitch.scatter(defensive_actions_team_df.x, defensive_actions_team_df.y, s=10, marker='x', color='yellow', alpha=0.2, ax=ax)

                    # Plotting the shirt no. of each player
                    for index, row in average_locs_and_count_df.iterrows():
                        player_initials = row["shirtNo"]
                        pitch.annotate(player_initials, xy=(row.x, row.y), c=line_color, ha='center', va='center', size=(14), ax=ax)

                    # Plotting a vertical line to show the median vertical position of all defensive actions, which is called Defensive Actions Height
                    dah = round(average_locs_and_count_df['x'].mean(), 2)
                    dah_show = round((dah*1.05), 2)
                    ax.axvline(x=dah, color='gray', linestyle='--', alpha=0.75, linewidth=2)

                    # Defense line Height
                    center_backs_height = average_locs_and_count_df[average_locs_and_count_df['position']=='DC']
                    def_line_h = round(center_backs_height['x'].median(), 2)
                    ax.axvline(x=def_line_h, color='gray', linestyle='dotted', alpha=0.5, linewidth=2)
                    # Forward line Height
                    Forwards_height = average_locs_and_count_df[average_locs_and_count_df['isFirstEleven']==1]
                    Forwards_height = Forwards_height.sort_values(by='x', ascending=False)
                    Forwards_height = Forwards_height.head(2)
                    fwd_line_h = round(Forwards_height['x'].mean(), 2)
                    ax.axvline(x=fwd_line_h, color='gray', linestyle='dotted', alpha=0.5, linewidth=2)

                    compactness = round((1 - ((fwd_line_h - def_line_h) / 105)) * 100, 2)
                    
                    if team_name == ateamName:
                        # inverting the axis for away team
                        ax.invert_xaxis()
                        ax.invert_yaxis()
                        ax.text(dah-1, 73, f"{dah_show}m", fontsize=15, color=line_color, ha='left', va='center')
                    else:
                        ax.text(dah-1, -5, f"{dah_show}m", fontsize=15, color=line_color, ha='right', va='center')

                    # Headlines and other texts
                    if team_name == hteamName:
                        ax.text(105, -5, f'Compact:{compactness}%', fontsize=15, color=line_color, ha='right', va='center')
                        ax.text(2,66, "circle = starter\nbox = sub", color='gray', size=12, ha='left', va='top')
                        ax.set_title(f"{hteamName}\nDefensive Action Heatmap", color=line_color, fontsize=25, fontweight='bold')
                    else:
                        ax.text(105, 73, f'Compact:{compactness}%', fontsize=15, color=line_color, ha='left', va='center')
                        ax.text(2,2, "circle = starter\nbox = sub", color='gray', size=12, ha='right', va='top')
                        ax.set_title(f"{ateamName}\nDefensive Action Heatmap", color=line_color, fontsize=25, fontweight='bold')

                    return {
                        'Team_Name': team_name,
                        'Average_Defensive_Action_Height': dah,
                        'Forward_Line_Pressing_Height': fwd_line_h
                    }

                fig,axs=plt.subplots(1,2, figsize=(20,10), facecolor=bg_color)
                defensive_block_stats_home = defensive_block(axs[0], defensive_home_average_locs_and_count_df, hteamName, hcol)
                defensive_block_stats_away = defensive_block(axs[1], defensive_away_average_locs_and_count_df, ateamName, acol)
                defensive_block_stats_home_list = [defensive_block_stats_home, defensive_block_stats_away]
                defensive_block_stats_home_df = pd.DataFrame(defensive_block_stats_home_list)

                    # Display the visualization

                st.header("Defensive Action Statistics")
                st.pyplot(fig)

                #st.dataframe(defensive_block_stats_home_df)
                defensive_block_stats_list = []
                defensive_block_stats_list.append(defensive_block_stats_home)
                defensive_block_stats_list.append(defensive_block_stats_away)
                defensive_block_stats_df = pd.DataFrame(defensive_block_stats_list)
                def draw_progressive_pass_map(ax, team_name, col):
                    dfpro = df[(df['teamName']==team_name) & (df['prog_pass']>=9.11) & (~df['qualifiers'].str.contains('CornerTaken|Freekick')) & 
                    (df['x']>=35) & (df['outcomeType']=='Successful')]
                    # df_pen = df[(df['teamName']==team_name) & (df['type']=='Pass') & (df['endX']>=88.5) & (df['endY']>=13.6) & (df['endY']<=54.4) &
                    #             ~((df['x']>=88.5) & (df['y']>=13.6) & (df['y']<=54.6))]
                    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2,
                                        corner_arcs=True)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    # ax.set_ylim(-0.5, 68.5)

                    if team_name == ateamName:
                        ax.invert_xaxis()
                        ax.invert_yaxis()

                    pro_count = len(dfpro)

                    # calculating the counts
                    left_pro = len(dfpro[dfpro['y']>=45.33])
                    mid_pro = len(dfpro[(dfpro['y']>=22.67) & (dfpro['y']<45.33)])
                    right_pro = len(dfpro[(dfpro['y']>=0) & (dfpro['y']<22.67)])
                    left_percentage = round((left_pro/pro_count)*100)
                    mid_percentage = round((mid_pro/pro_count)*100)
                    right_percentage = round((right_pro/pro_count)*100)

                    ax.hlines(22.67, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)
                    ax.hlines(45.33, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)

                    # showing the texts in the pitch
                    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
                    if col == hcol:
                        ax.text(8, 11.335, f'{right_pro}\n({right_percentage}%)', color=hcol, fontsize=24, va='center', ha='center', bbox=bbox_props)
                        ax.text(8, 34, f'{mid_pro}\n({mid_percentage}%)', color=hcol, fontsize=24, va='center', ha='center', bbox=bbox_props)
                        ax.text(8, 56.675, f'{left_pro}\n({left_percentage}%)', color=hcol, fontsize=24, va='center', ha='center', bbox=bbox_props)
                    else:
                        ax.text(8, 11.335, f'{right_pro}\n({right_percentage}%)', color=acol, fontsize=24, va='center', ha='center', bbox=bbox_props)
                        ax.text(8, 34, f'{mid_pro}\n({mid_percentage}%)', color=acol, fontsize=24, va='center', ha='center', bbox=bbox_props)
                        ax.text(8, 56.675, f'{left_pro}\n({left_percentage}%)', color=acol, fontsize=24, va='center', ha='center', bbox=bbox_props)

                    # plotting the passes
                    pro_pass = pitch.lines(dfpro.x, dfpro.y, dfpro.endX, dfpro.endY, lw=3.5, comet=True, color=col, ax=ax, alpha=0.5)
                    # plotting some scatters at the end of each pass
                    pro_pass_end = pitch.scatter(dfpro.endX, dfpro.endY, s=35, edgecolor=col, linewidth=1, color=bg_color, zorder=2, ax=ax)

                    counttext = f"{pro_count} Progressive Passes"

                    # Heading and other texts
                    if col == hcol:
                        ax.set_title(f"{hteamName}\n{counttext}", color=line_color, fontsize=25, fontweight='bold')
                    else:
                        ax.set_title(f"{ateamName}\n{counttext}", color=line_color, fontsize=25, fontweight='bold')

                    return {
                        'Team_Name': team_name,
                        'Total_Progressive_Passes': pro_count,
                        'Progressive_Passes_From_Left': left_pro,
                        'Progressive_Passes_From_Center': mid_pro,
                        'Progressive_Passes_From_Right': right_pro
                    }

                fig, axs = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)

                # Draw progressive pass maps for home and away teams
                Progressvie_Passes_Stats_home = draw_progressive_pass_map(axs[0], hteamName, hcol)
                Progressvie_Passes_Stats_away = draw_progressive_pass_map(axs[1], ateamName, acol)

                # Collect statistics into a list
                Progressvie_Passes_Stats_list = [
                    Progressvie_Passes_Stats_home,
                    Progressvie_Passes_Stats_away
                ]

                # Convert to DataFrame
                Progressvie_Passes_Stats_df = pd.DataFrame(Progressvie_Passes_Stats_list)

                # Display the visualization
                st.header("Progressive Passes Statistics")
                st.pyplot(fig)

                # Display the progressive pass statistics
                #st.dataframe(Progressvie_Passes_Stats_df)
                def draw_progressive_carry_map(ax, team_name, col):
                    dfpro = df[(df['teamName']==team_name) & (df['prog_carry']>=9.11) & (df['endX']>=35)]
                    # df_pen = df[(df['teamName']==team_name) & (df['type']=='Pass') & (df['endX']>=88.5) & (df['endY']>=13.6) & (df['endY']<=54.4) &
                    #             ~((df['x']>=88.5) & (df['y']>=13.6) & (df['y']<=54.6))]
                    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2,
                                        corner_arcs=True)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    # ax.set_ylim(-5, 68.5)

                    if team_name == ateamName:
                        ax.invert_xaxis()
                        ax.invert_yaxis()

                    pro_count = len(dfpro)

                    # calculating the counts
                    left_pro = len(dfpro[dfpro['y']>=45.33])
                    mid_pro = len(dfpro[(dfpro['y']>=22.67) & (dfpro['y']<45.33)])
                    right_pro = len(dfpro[(dfpro['y']>=0) & (dfpro['y']<22.67)])
                    left_percentage = round((left_pro/pro_count)*100)
                    mid_percentage = round((mid_pro/pro_count)*100)
                    right_percentage = round((right_pro/pro_count)*100)

                    ax.hlines(22.67, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)
                    ax.hlines(45.33, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)

                    # showing the texts in the pitch
                    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
                    if col == hcol:
                        ax.text(8, 11.335, f'{right_pro}\n({right_percentage}%)', color=hcol, fontsize=24, va='center', ha='center', bbox=bbox_props)
                        ax.text(8, 34, f'{mid_pro}\n({mid_percentage}%)', color=hcol, fontsize=24, va='center', ha='center', bbox=bbox_props)
                        ax.text(8, 56.675, f'{left_pro}\n({left_percentage}%)', color=hcol, fontsize=24, va='center', ha='center', bbox=bbox_props)
                    else:
                        ax.text(8, 11.335, f'{right_pro}\n({right_percentage}%)', color=acol, fontsize=24, va='center', ha='center', bbox=bbox_props)
                        ax.text(8, 34, f'{mid_pro}\n({mid_percentage}%)', color=acol, fontsize=24, va='center', ha='center', bbox=bbox_props)
                        ax.text(8, 56.675, f'{left_pro}\n({left_percentage}%)', color=acol, fontsize=24, va='center', ha='center', bbox=bbox_props)

                    # plotting the carries
                    for index, row in dfpro.iterrows():
                        arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', color=col, zorder=4, mutation_scale=20, 
                                                        alpha=0.9, linewidth=2, linestyle='--')
                        ax.add_patch(arrow)

                    counttext = f"{pro_count} Progressive Carries"

                    # Heading and other texts
                    if col == hcol:
                        ax.set_title(f"{hteamName}\n{counttext}", color=line_color, fontsize=25, fontweight='bold')
                    else:
                        ax.set_title(f"{ateamName}\n{counttext}", color=line_color, fontsize=25, fontweight='bold')

                    return {
                        'Team_Name': team_name,
                        'Total_Progressive_Carries': pro_count,
                        'Progressive_Carries_From_Left': left_pro,
                        'Progressive_Carries_From_Center': mid_pro,
                        'Progressive_Carries_From_Right': right_pro
                    }

                fig, axs = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)

                # Draw progressive carry maps for home and away teams
                Progressvie_Carries_Stats_home = draw_progressive_carry_map(axs[0], hteamName, hcol)
                Progressvie_Carries_Stats_away = draw_progressive_carry_map(axs[1], ateamName, acol)

                # Collect statistics into a list
                Progressvie_Carries_Stats_list = [
                    Progressvie_Carries_Stats_home,
                    Progressvie_Carries_Stats_away
                ]

                # Convert to DataFrame
                Progressvie_Carries_Stats_df = pd.DataFrame(Progressvie_Carries_Stats_list)

                # Display the visualization
                st.header("Progressive Carries Statistics")
                st.pyplot(fig)

                #st.dataframe(Progressvie_Carries_Stats_df)

                # filtering the shots only
                mask4 = (df['type'] == 'Goal') | (df['type'] == 'MissedShots') | (df['type'] == 'SavedShot') | (df['type'] == 'ShotOnPost')
                Shotsdf = df[mask4]
                Shotsdf = Shotsdf.reset_index(drop=True)

                # filtering according to the types of shots
                hShotsdf = Shotsdf[Shotsdf['teamName']==hteamName]
                aShotsdf = Shotsdf[Shotsdf['teamName']==ateamName]
                hSavedf = hShotsdf[(hShotsdf['type']=='SavedShot') & (~hShotsdf['qualifiers'].str.contains(': 82,'))]
                aSavedf = aShotsdf[(aShotsdf['type']=='SavedShot') & (~aShotsdf['qualifiers'].str.contains(': 82,'))]
                hogdf = hShotsdf[(hShotsdf['teamName']==hteamName) & (hShotsdf['qualifiers'].str.contains('OwnGoal'))]
                aogdf = aShotsdf[(aShotsdf['teamName']==ateamName) & (aShotsdf['qualifiers'].str.contains('OwnGoal'))]

                # Center Goal point
                given_point = (105, 34)
                # Calculate distances
                home_shot_distances = np.sqrt((hShotsdf['x'] - given_point[0])**2 + (hShotsdf['y'] - given_point[1])**2)
                home_average_shot_distance = round(home_shot_distances.mean(),2)
                away_shot_distances = np.sqrt((aShotsdf['x'] - given_point[0])**2 + (aShotsdf['y'] - given_point[1])**2)
                away_average_shot_distance = round(away_shot_distances.mean(),2)

                def plot_shotmap(ax):
                    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, linewidth=2, line_color=line_color)
                    pitch.draw(ax=ax)
                    ax.set_ylim(-0.5,68.5)
                    ax.set_xlim(-0.5,105.5)
                    
                    #shooting stats
                    hTotalShots = len(hShotsdf)
                    aTotalShots = len(aShotsdf)
                    hShotsOnT = len(hSavedf) + hgoal_count
                    aShotsOnT = len(aSavedf) + agoal_count
                    hxGpSh = round(hxg/hTotalShots, 2)
                    axGpSh = round(axg/hTotalShots, 2)
                    
                    # without big chances for home team
                    hGoalData = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'Goal') & (~Shotsdf['qualifiers'].str.contains('BigChance'))]
                    hPostData = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'ShotOnPost') & (~Shotsdf['qualifiers'].str.contains('BigChance'))]
                    hSaveData = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'SavedShot') & (~Shotsdf['qualifiers'].str.contains('BigChance'))]
                    hMissData = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'MissedShots') & (~Shotsdf['qualifiers'].str.contains('BigChance'))]
                    # only big chances of home team
                    Big_C_hGoalData = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'Goal') & (Shotsdf['qualifiers'].str.contains('BigChance'))]
                    Big_C_hPostData = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'ShotOnPost') & (Shotsdf['qualifiers'].str.contains('BigChance'))]
                    Big_C_hSaveData = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'SavedShot') & (Shotsdf['qualifiers'].str.contains('BigChance'))]
                    Big_C_hMissData = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'MissedShots') & (Shotsdf['qualifiers'].str.contains('BigChance'))]
                    total_bigC_home = len(Big_C_hGoalData) + len(Big_C_hPostData) + len(Big_C_hSaveData) + len(Big_C_hMissData)
                    bigC_miss_home = len(Big_C_hPostData) + len(Big_C_hSaveData) + len(Big_C_hMissData)
                    # normal shots scatter of home team
                    sc2 = pitch.scatter((105-hPostData.x), (68-hPostData.y), s=200, edgecolors=hcol, c=hcol, marker='o', ax=ax)
                    sc3 = pitch.scatter((105-hSaveData.x), (68-hSaveData.y), s=200, edgecolors=hcol, c='None', hatch='///////', marker='o', ax=ax)
                    sc4 = pitch.scatter((105-hMissData.x), (68-hMissData.y), s=200, edgecolors=hcol, c='None', marker='o', ax=ax)
                    sc1 = pitch.scatter((105-hGoalData.x), (68-hGoalData.y), s=350, edgecolors='green', linewidths=0.6, c='None', marker='football', zorder=3, ax=ax)
                    sc1_og = pitch.scatter((105-hogdf.x), (68-hogdf.y), s=350, edgecolors='orange', linewidths=0.6, c='None', marker='football', zorder=3, ax=ax)
                    # big chances bigger scatter of home team
                    bc_sc2 = pitch.scatter((105-Big_C_hPostData.x), (68-Big_C_hPostData.y), s=500, edgecolors=hcol, c=hcol, marker='o', ax=ax)
                    bc_sc3 = pitch.scatter((105-Big_C_hSaveData.x), (68-Big_C_hSaveData.y), s=500, edgecolors=hcol, c='None', hatch='///////', marker='o', ax=ax)
                    bc_sc4 = pitch.scatter((105-Big_C_hMissData.x), (68-Big_C_hMissData.y), s=500, edgecolors=hcol, c='None', marker='o', ax=ax)
                    bc_sc1 = pitch.scatter((105-Big_C_hGoalData.x), (68-Big_C_hGoalData.y), s=650, edgecolors='green', linewidths=0.6, c='None', marker='football', ax=ax)

                    # without big chances for away team
                    aGoalData = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'Goal') & (~Shotsdf['qualifiers'].str.contains('BigChance'))]
                    aPostData = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'ShotOnPost') & (~Shotsdf['qualifiers'].str.contains('BigChance'))]
                    aSaveData = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'SavedShot') & (~Shotsdf['qualifiers'].str.contains('BigChance'))]
                    aMissData = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'MissedShots') & (~Shotsdf['qualifiers'].str.contains('BigChance'))]
                    # only big chances of away team
                    Big_C_aGoalData = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'Goal') & (Shotsdf['qualifiers'].str.contains('BigChance'))]
                    Big_C_aPostData = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'ShotOnPost') & (Shotsdf['qualifiers'].str.contains('BigChance'))]
                    Big_C_aSaveData = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'SavedShot') & (Shotsdf['qualifiers'].str.contains('BigChance'))]
                    Big_C_aMissData = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'MissedShots') & (Shotsdf['qualifiers'].str.contains('BigChance'))]
                    total_bigC_away = len(Big_C_aGoalData) + len(Big_C_aPostData) + len(Big_C_aSaveData) + len(Big_C_aMissData)
                    bigC_miss_away = len(Big_C_aPostData) + len(Big_C_aSaveData) + len(Big_C_aMissData)
                    # normal shots scatter of away team
                    sc6 = pitch.scatter(aPostData.x, aPostData.y, s=200, edgecolors=acol, c=acol, marker='o', ax=ax)
                    sc7 = pitch.scatter(aSaveData.x, aSaveData.y, s=200, edgecolors=acol, c='None', hatch='///////', marker='o', ax=ax)
                    sc8 = pitch.scatter(aMissData.x, aMissData.y, s=200, edgecolors=acol, c='None', marker='o', ax=ax)
                    sc5 = pitch.scatter(aGoalData.x, aGoalData.y, s=350, edgecolors='green', linewidths=0.6, c='None', marker='football', zorder=3, ax=ax)
                    sc5_og = pitch.scatter((aogdf.x), (aogdf.y), s=350, edgecolors='orange', linewidths=0.6, c='None', marker='football', zorder=3, ax=ax)
                    # big chances bigger scatter of away team
                    bc_sc6 = pitch.scatter(Big_C_aPostData.x, Big_C_aPostData.y, s=700, edgecolors=acol, c=acol, marker='o', ax=ax)
                    bc_sc7 = pitch.scatter(Big_C_aSaveData.x, Big_C_aSaveData.y, s=700, edgecolors=acol, c='None', hatch='///////', marker='o', ax=ax)
                    bc_sc8 = pitch.scatter(Big_C_aMissData.x, Big_C_aMissData.y, s=700, edgecolors=acol, c='None', marker='o', ax=ax)
                    bc_sc5 = pitch.scatter(Big_C_aGoalData.x, Big_C_aGoalData.y, s=850, edgecolors='green', linewidths=0.6, c='None', marker='football', ax=ax)

                    # sometimes the both teams ends the match 0-0, then normalizing the data becomes problem, thats why this part of the code
                    if hgoal_count+agoal_count == 0:
                        hgoal = 10
                        agoal = 10
                    else:
                        hgoal = (hgoal_count/(hgoal_count+agoal_count))*20
                        agoal = (agoal_count/(hgoal_count+agoal_count))*20
                        
                    if total_bigC_home+total_bigC_away == 0:
                        total_bigC_home_n = 10
                        total_bigC_away_n = 10
                    else:
                        total_bigC_home_n = (total_bigC_home/(total_bigC_home+total_bigC_away))*20
                        total_bigC_away_n = (total_bigC_away/(total_bigC_home+total_bigC_away))*20
                        
                    if bigC_miss_home+bigC_miss_away == 0:
                        bigC_miss_home_n = 10
                        bigC_miss_away_n = 10
                    else:
                        bigC_miss_home_n = (bigC_miss_home/(bigC_miss_home+bigC_miss_away))*20
                        bigC_miss_away_n = (bigC_miss_away/(bigC_miss_home+bigC_miss_away))*20

                    if hShotsOnT+aShotsOnT == 0:
                        hShotsOnT_n = 10
                        aShotsOnT_n = 10
                    else:
                        hShotsOnT_n = (hShotsOnT/(hShotsOnT+aShotsOnT))*20
                        aShotsOnT_n = (aShotsOnT/(hShotsOnT+aShotsOnT))*20

                    if hxgot+axgot == 0:
                        hxgot_n = 10
                        axgot_n = 10
                    else:
                        hxgot_n = (hxgot/(hxgot+axgot))*20
                        axgot_n = (axgot/(hxgot+axgot))*20

                    # Stats bar diagram
                    shooting_stats_title = [62, 62-(1*7), 62-(2*7), 62-(3*7), 62-(4*7), 62-(5*7), 62-(6*7), 62-(7*7), 62-(8*7)]
                    shooting_stats_home = [hgoal_count, hxg, hxgot, hTotalShots, hShotsOnT, hxGpSh, total_bigC_home, bigC_miss_home, home_average_shot_distance]
                    shooting_stats_away = [agoal_count, axg, axgot, aTotalShots, aShotsOnT, axGpSh, total_bigC_away, bigC_miss_away, away_average_shot_distance]

                    # normalizing the stats
                    shooting_stats_normalized_home = [hgoal, (hxg/(hxg+axg))*20, hxgot_n,
                                                    (hTotalShots/(hTotalShots+aTotalShots))*20, hShotsOnT_n,
                                                    total_bigC_home_n, bigC_miss_home_n,
                                                    (hxGpSh/(hxGpSh+axGpSh))*20, 
                                                    (home_average_shot_distance/(home_average_shot_distance+away_average_shot_distance))*20]
                    shooting_stats_normalized_away = [agoal, (axg/(hxg+axg))*20, axgot_n,
                                                    (aTotalShots/(hTotalShots+aTotalShots))*20, aShotsOnT_n,
                                                    total_bigC_away_n, bigC_miss_away_n,
                                                    (axGpSh/(hxGpSh+axGpSh))*20,
                                                    (away_average_shot_distance/(home_average_shot_distance+away_average_shot_distance))*20]

                    # definig the start point
                    start_x = 42.5
                    start_x_for_away = [x + 42.5 for x in shooting_stats_normalized_home]
                    ax.barh(shooting_stats_title, shooting_stats_normalized_home, height=5, color=hcol, left=start_x)
                    ax.barh(shooting_stats_title, shooting_stats_normalized_away, height=5, left=start_x_for_away, color=acol)
                    # Turn off axis-related elements
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # plotting the texts
                    ax.text(52.5, 62, "Goals", color=bg_color, fontsize=18, ha='center', va='center', fontweight='bold')
                    ax.text(52.5, 62-(1*7), "xG", color=bg_color, fontsize=18, ha='center', va='center', fontweight='bold')
                    ax.text(52.5, 62-(2*7), "xGOT", color=bg_color, fontsize=18, ha='center', va='center', fontweight='bold')
                    ax.text(52.5, 62-(3*7), "Shots", color=bg_color, fontsize=18, ha='center', va='center', fontweight='bold')
                    ax.text(52.5, 62-(4*7), "On Target", color=bg_color, fontsize=18, ha='center', va='center', fontweight='bold')
                    ax.text(52.5, 62-(5*7), "BigChance", color=bg_color, fontsize=18, ha='center', va='center', fontweight='bold')
                    ax.text(52.5, 62-(6*7), "BigC.Miss", color=bg_color, fontsize=18, ha='center', va='center', fontweight='bold')
                    ax.text(52.5, 62-(7*7), "xG/Shot", color=bg_color, fontsize=18, ha='center', va='center', fontweight='bold')
                    ax.text(52.5, 62-(8*7), "Avg.Dist.", color=bg_color, fontsize=18, ha='center', va='center', fontweight='bold')

                    ax.text(41.5, 62, f"{hgoal_count}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
                    ax.text(41.5, 62-(1*7), f"{hxg}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
                    ax.text(41.5, 62-(2*7), f"{hxgot}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
                    ax.text(41.5, 62-(3*7), f"{hTotalShots}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
                    ax.text(41.5, 62-(4*7), f"{hShotsOnT}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
                    ax.text(41.5, 62-(5*7), f"{total_bigC_home}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
                    ax.text(41.5, 62-(6*7), f"{bigC_miss_home}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
                    ax.text(41.5, 62-(7*7), f"{hxGpSh}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
                    ax.text(41.5, 62-(8*7), f"{home_average_shot_distance}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')

                    ax.text(63.5, 62, f"{agoal_count}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
                    ax.text(63.5, 62-(1*7), f"{axg}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
                    ax.text(63.5, 62-(2*7), f"{axgot}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
                    ax.text(63.5, 62-(3*7), f"{aTotalShots}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
                    ax.text(63.5, 62-(4*7), f"{aShotsOnT}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
                    ax.text(63.5, 62-(5*7), f"{total_bigC_away}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
                    ax.text(63.5, 62-(6*7), f"{bigC_miss_away}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
                    ax.text(63.5, 62-(7*7), f"{axGpSh}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
                    ax.text(63.5, 62-(8*7), f"{away_average_shot_distance}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')

                    # Heading and other texts
                    ax.text(0, 70, f"{hteamName}\n<---shots", color=hcol, size=25, ha='left', fontweight='bold')
                    ax.text(105, 70, f"{ateamName}\nshots--->", color=acol, size=25, ha='right', fontweight='bold')

                    home_data = {
                        'Team_Name': hteamName,
                        'Goals_Scored': hgoal_count,
                        'xG': hxg,
                        'xGOT': hxgot,
                        'Total_Shots': hTotalShots,
                        'Shots_On_Target': hShotsOnT,
                        'BigChances': total_bigC_home,
                        'BigChances_Missed': bigC_miss_home,
                        'xG_per_Shot': hxGpSh,
                        'Average_Shot_Distance': home_average_shot_distance
                    }
                    
                    away_data = {
                        'Team_Name': ateamName,
                        'Goals_Scored': agoal_count,
                        'xG': axg,
                        'xGOT': axgot,
                        'Total_Shots': aTotalShots,
                        'Shots_On_Target': aShotsOnT,
                        'BigChances': total_bigC_away,
                        'BigChances_Missed': bigC_miss_away,
                        'xG_per_Shot': axGpSh,
                        'Average_Shot_Distance': away_average_shot_distance
                    }
                    
                    return [home_data, away_data]

                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)

                # Call the `plot_shotmap` function
                shooting_stats = plot_shotmap(ax)

                # Convert the shooting stats to a DataFrame
                shooting_stats_df = pd.DataFrame(shooting_stats)

                # Display the visualization in Streamlit
                st.header("Shooting Statistics")
                st.pyplot(fig)
                # Display the shooting statistics DataFrame in Streamlit
                #st.dataframe(shooting_stats_df)
                def plot_goalPost(ax):
                    hShotsdf = Shotsdf[Shotsdf['teamName']==hteamName]
                    aShotsdf = Shotsdf[Shotsdf['teamName']==ateamName]
                    # converting the datapoints according to the pitch dimension, because the goalposts are being plotted inside the pitch using pitch's dimension
                    hShotsdf['goalMouthZ'] = hShotsdf['goalMouthZ']*0.75
                    aShotsdf['goalMouthZ'] = (aShotsdf['goalMouthZ']*0.75) + 38

                    hShotsdf['goalMouthY'] = ((37.66 - hShotsdf['goalMouthY'])*12.295) + 7.5
                    aShotsdf['goalMouthY'] = ((37.66 - aShotsdf['goalMouthY'])*12.295) + 7.5

                    # plotting an invisible pitch using the pitch color and line color same color, because the goalposts are being plotted inside the pitch using pitch's dimension
                    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=bg_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_ylim(-0.5,68.5)
                    ax.set_xlim(-0.5,105.5)

                    # away goalpost bars
                    ax.plot([7.5, 7.5], [0, 30], color=line_color, linewidth=5)
                    ax.plot([7.5, 97.5], [30, 30], color=line_color, linewidth=5)
                    ax.plot([97.5, 97.5], [30, 0], color=line_color, linewidth=5)
                    ax.plot([0, 105], [0, 0], color=line_color, linewidth=3)
                    # plotting the away net
                    y_values = np.arange(0, 6) * 6
                    for y in y_values:
                        ax.plot([7.5, 97.5], [y, y], color=line_color, linewidth=2, alpha=0.2)
                    x_values = (np.arange(0, 11) * 9) + 7.5
                    for x in x_values:
                        ax.plot([x, x], [0, 30], color=line_color, linewidth=2, alpha=0.2)
                    # home goalpost bars
                    ax.plot([7.5, 7.5], [38, 68], color=line_color, linewidth=5)
                    ax.plot([7.5, 97.5], [68, 68], color=line_color, linewidth=5)
                    ax.plot([97.5, 97.5], [68, 38], color=line_color, linewidth=5)
                    ax.plot([0, 105], [38, 38], color=line_color, linewidth=3)
                    # plotting the home net
                    y_values = (np.arange(0, 6) * 6) + 38
                    for y in y_values:
                        ax.plot([7.5, 97.5], [y, y], color=line_color, linewidth=2, alpha=0.2)
                    x_values = (np.arange(0, 11) * 9) + 7.5
                    for x in x_values:
                        ax.plot([x, x], [38, 68], color=line_color, linewidth=2, alpha=0.2)

                    # filtering different types of shots without BigChance
                    hSavedf = hShotsdf[(hShotsdf['type']=='SavedShot') & (~hShotsdf['qualifiers'].str.contains(': 82,')) & (~hShotsdf['qualifiers'].str.contains('BigChance'))]
                    hGoaldf = hShotsdf[(hShotsdf['type']=='Goal') & (~hShotsdf['qualifiers'].str.contains('OwnGoal')) & (~hShotsdf['qualifiers'].str.contains('BigChance'))]
                    hPostdf = hShotsdf[(hShotsdf['type']=='ShotOnPost') & (~hShotsdf['qualifiers'].str.contains('BigChance'))]
                    aSavedf = aShotsdf[(aShotsdf['type']=='SavedShot') & (~aShotsdf['qualifiers'].str.contains(': 82,')) & (~aShotsdf['qualifiers'].str.contains('BigChance'))]
                    aGoaldf = aShotsdf[(aShotsdf['type']=='Goal') & (~aShotsdf['qualifiers'].str.contains('OwnGoal')) & (~aShotsdf['qualifiers'].str.contains('BigChance'))]
                    aPostdf = aShotsdf[(aShotsdf['type']=='ShotOnPost') & (~aShotsdf['qualifiers'].str.contains('BigChance'))]
                    # filtering different types of shots with BigChance
                    hSavedf_bc = hShotsdf[(hShotsdf['type']=='SavedShot') & (~hShotsdf['qualifiers'].str.contains(': 82,')) & (hShotsdf['qualifiers'].str.contains('BigChance'))]
                    hGoaldf_bc = hShotsdf[(hShotsdf['type']=='Goal') & (~hShotsdf['qualifiers'].str.contains('OwnGoal')) & (hShotsdf['qualifiers'].str.contains('BigChance'))]
                    hPostdf_bc = hShotsdf[(hShotsdf['type']=='ShotOnPost') & (hShotsdf['qualifiers'].str.contains('BigChance'))]
                    aSavedf_bc = aShotsdf[(aShotsdf['type']=='SavedShot') & (~aShotsdf['qualifiers'].str.contains(': 82,')) & (aShotsdf['qualifiers'].str.contains('BigChance'))]
                    aGoaldf_bc = aShotsdf[(aShotsdf['type']=='Goal') & (~aShotsdf['qualifiers'].str.contains('OwnGoal')) & (aShotsdf['qualifiers'].str.contains('BigChance'))]
                    aPostdf_bc = aShotsdf[(aShotsdf['type']=='ShotOnPost') & (aShotsdf['qualifiers'].str.contains('BigChance'))]

                    # scattering those shots without BigChance
                    sc1 = pitch.scatter(hSavedf.goalMouthY, hSavedf.goalMouthZ, marker='o', c=bg_color, zorder=3, edgecolor=acol, hatch='/////', s=350, ax=ax)
                    sc2 = pitch.scatter(hGoaldf.goalMouthY, hGoaldf.goalMouthZ, marker='football', c=bg_color, zorder=3, edgecolors='green', s=350, ax=ax)
                    sc3 = pitch.scatter(hPostdf.goalMouthY, hPostdf.goalMouthZ, marker='o', c=bg_color, zorder=3, edgecolors='orange', hatch='/////', s=350, ax=ax)
                    sc4 = pitch.scatter(aSavedf.goalMouthY, aSavedf.goalMouthZ, marker='o', c=bg_color, zorder=3, edgecolor=hcol, hatch='/////', s=350, ax=ax)
                    sc5 = pitch.scatter(aGoaldf.goalMouthY, aGoaldf.goalMouthZ, marker='football', c=bg_color, zorder=3, edgecolors='green', s=350, ax=ax)
                    sc6 = pitch.scatter(aPostdf.goalMouthY, aPostdf.goalMouthZ, marker='o', c=bg_color, zorder=3, edgecolors='orange', hatch='/////', s=350, ax=ax)
                    # scattering those shots with BigChance
                    sc1_bc = pitch.scatter(hSavedf_bc.goalMouthY, hSavedf_bc.goalMouthZ, marker='o', c=bg_color, zorder=3, edgecolor=acol, hatch='/////', s=1000, ax=ax)
                    sc2_bc = pitch.scatter(hGoaldf_bc.goalMouthY, hGoaldf_bc.goalMouthZ, marker='football', c=bg_color, zorder=3, edgecolors='green', s=1000, ax=ax)
                    sc3_bc = pitch.scatter(hPostdf_bc.goalMouthY, hPostdf_bc.goalMouthZ, marker='o', c=bg_color, zorder=3, edgecolors='orange', hatch='/////', s=1000, ax=ax)
                    sc4_bc = pitch.scatter(aSavedf_bc.goalMouthY, aSavedf_bc.goalMouthZ, marker='o', c=bg_color, zorder=3, edgecolor=hcol, hatch='/////', s=1000, ax=ax)
                    sc5_bc = pitch.scatter(aGoaldf_bc.goalMouthY, aGoaldf_bc.goalMouthZ, marker='football', c=bg_color, zorder=3, edgecolors='green', s=1000, ax=ax)
                    sc6_bc = pitch.scatter(aPostdf_bc.goalMouthY, aPostdf_bc.goalMouthZ, marker='o', c=bg_color, zorder=3, edgecolors='orange', hatch='/////', s=1000, ax=ax)

                    # Headlines and other texts
                    ax.text(52.5, 70, f"{hteamName} GK saves", color=hcol, fontsize=30, ha='center', fontweight='bold')
                    ax.text(52.5, -2, f"{ateamName} GK saves", color=acol, fontsize=30, ha='center', va='top', fontweight='bold')

                    ax.text(100, 68, f"Saves = {len(aSavedf)+len(aSavedf_bc)}\n\nxGOT faced:\n{axgot}\n\nGoals Prevented:\n{round(axgot - len(aGoaldf) - len(aGoaldf_bc),2)}",
                                    color=hcol, fontsize=16, va='top', ha='left')
                    ax.text(100, 2, f"Saves = {len(hSavedf)+len(hSavedf_bc)}\n\nxGOT faced:\n{hxgot}\n\nGoals Prevented:\n{round(hxgot - len(hGoaldf) - len(hGoaldf_bc),2)}",
                                    color=acol, fontsize=16, va='bottom', ha='left')

                    home_data = {
                        'Team_Name': hteamName,
                        'Shots_Saved': len(hSavedf)+len(hSavedf_bc),
                        'Big_Chance_Saved': len(hSavedf_bc),
                        'Goals_Prevented': round(hxgot - len(hGoaldf) - len(hGoaldf_bc),2)
                    }
                    
                    away_data = {
                        'Team_Name': ateamName,
                        'Shots_Saved': len(aSavedf)+len(aSavedf_bc),
                        'Big_Chance_Saved': len(aSavedf_bc),
                        'Goals_Prevented': round(axgot - len(aGoaldf) - len(aGoaldf_bc),2)
                    }
                    
                    return [home_data, away_data]

                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)

                # Call the `plot_goalPost` function
                goalkeeping_stats = plot_goalPost(ax)

                # Convert the goalkeeping stats to a DataFrame
                goalkeeping_stats_df = pd.DataFrame(goalkeeping_stats)

                # Display the visualization in Streamlit
                st.header("Goalkeeping Statistics")
                st.pyplot(fig)

                # Display the goalkeeping statistics DataFrame in Streamlit
                #st.dataframe(goalkeeping_stats_df)
                u_df = df.copy()
                u_df = u_df[~u_df['qualifiers'].str.contains('CornerTaken')]
                u_df = u_df[['x', 'minute', 'type', 'teamName']]
                u_df = u_df[~u_df['type'].isin(['Start', 'OffsidePass', 'OffsideProvoked', 'Card', 'CornerAwarded', 'End', 
                                                'OffsideGiven', 'SubstitutionOff', 'SubstitutionOn', 'FormationChange', 'FormationSet'])].reset_index(drop=True)
                u_df.loc[u_df['teamName'] == ateamName, 'x'] = 105 - u_df.loc[u_df['teamName'] == ateamName, 'x']

                Momentumdf = u_df.groupby('minute')['x'].mean()
                Momentumdf = Momentumdf.reset_index()
                Momentumdf.columns = ['minute', 'average_x']
                Momentumdf['average_x'] = Momentumdf['average_x'] - 52.5
                def plot_Momentum(ax):
                    # Set colors based on positive or negative values
                    colors = [hcol if x > 0 else acol for x in Momentumdf['average_x']]

                    # making a list of munutes when goals are scored
                    hgoal_list = homedf[(homedf['type'] == 'Goal') & (~homedf['qualifiers'].str.contains('OwnGoal'))]['minute'].tolist()
                    agoal_list = awaydf[(awaydf['type'] == 'Goal') & (~awaydf['qualifiers'].str.contains('OwnGoal'))]['minute'].tolist()
                    hog_list = homedf[(homedf['type'] == 'Goal') & (homedf['qualifiers'].str.contains('OwnGoal'))]['minute'].tolist()
                    aog_list = awaydf[(awaydf['type'] == 'Goal') & (awaydf['qualifiers'].str.contains('OwnGoal'))]['minute'].tolist()
                    hred_list = homedf[homedf['qualifiers'].str.contains('Red|SecondYellow')]['minute'].tolist()
                    ared_list = awaydf[awaydf['qualifiers'].str.contains('Red|SecondYellow')]['minute'].tolist()

                    # plotting scatters when goals are scored
                    highest_x = Momentumdf['average_x'].max()
                    lowest_x = Momentumdf['average_x'].min()
                    highest_minute = Momentumdf['minute'].max()
                    hscatter_y = [highest_x]*len(hgoal_list)
                    ascatter_y = [lowest_x]*len(agoal_list)
                    hogscatter_y = [highest_x]*len(aog_list)
                    aogscatter_y = [lowest_x]*len(hog_list)
                    hred_y = [highest_x]*len(hred_list)
                    ared_y = [lowest_x]*len(ared_list)

                    ax.text((45/2), lowest_x, 'First Half', color='gray', fontsize=20, alpha=0.25, va='center', ha='center')
                    ax.text((45+(45/2)), lowest_x, 'Second Half', color='gray', fontsize=20, alpha=0.25, va='center', ha='center')

                    ax.scatter(hgoal_list, hscatter_y, s=250, c='None', edgecolor='green', hatch='////', marker='o')
                    ax.scatter(agoal_list, ascatter_y, s=250, c='None', edgecolor='green', hatch='////', marker='o')
                    ax.scatter(hog_list, aogscatter_y, s=250, c='None', edgecolor='orange', hatch='////', marker='o')
                    ax.scatter(aog_list, hogscatter_y, s=250, c='None', edgecolor='orange', hatch='////', marker='o')
                    ax.scatter(hred_list, hred_y, s=250, c='None', edgecolor='red', hatch='////', marker='s')
                    ax.scatter(ared_list, ared_y, s=250, c='None', edgecolor='red', hatch='////', marker='s')

                    # Creating the bar plot
                    ax.bar(Momentumdf['minute'], Momentumdf['average_x'], width=1, color=colors)
                    ax.set_xticks(range(0, len(Momentumdf['minute']), 5))
                    ax.axvline(45, color='gray', linewidth=2, linestyle='dotted')
                    # ax.axvline(90, color='gray', linewidth=2, linestyle='dotted')
                    ax.set_facecolor(bg_color)
                    # Hide spines
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    # # Hide ticks
                    ax.tick_params(axis='both', which='both', length=0)
                    ax.tick_params(axis='x', colors=line_color)
                    ax.tick_params(axis='y', colors=bg_color)
                    # Add labels and title
                    ax.set_xlabel('Minute', color=line_color, fontsize=20)
                    # ax.set_ylabel('Avg. xT per minute', color=line_color, fontsize=20)
                    ax.axhline(y=0, color=line_color, alpha=1, linewidth=2)

                    ax.text(highest_minute+1,highest_x, f"{hteamName}\nxT: {hxT}", color=hcol, fontsize=20, va='bottom', ha='left')
                    ax.text(highest_minute+1,lowest_x,  f"{ateamName}\nxT: {axT}", color=acol, fontsize=20, va='top', ha='left')

                    ax.set_title('Match Momentum', color=line_color, fontsize=30, fontweight='bold')
                    
                    home_data = {
                        'Team_Name': hteamName,
                        'xT': hxT
                    }
                    
                    away_data = {
                        'Team_Name': ateamName,
                        'xT': axT
                    }
                    
                    return [home_data, away_data]

                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)

                # Call the `plot_Momentum` function to generate the plot and stats
                xT_stats = plot_Momentum(ax)

                # Convert the xT statistics to a DataFrame
                xT_stats_df = pd.DataFrame(xT_stats)

                # Display the momentum visualization in Streamlit
                st.header("xT Statistics")
                st.pyplot(fig)

                # Display the xT statistics DataFrame in Streamlit
                #st.dataframe(xT_stats_df)
                #Here I have calculated a lot of stats, all of them I couldn't show in the viz because of lack of spaces, but I kept those in the code
                # Passing Stats
                #Possession%
                hpossdf = df[(df['teamName']==hteamName) & (df['type']=='Pass')]
                apossdf = df[(df['teamName']==ateamName) & (df['type']=='Pass')]
                hposs = round((len(hpossdf)/(len(hpossdf)+len(apossdf)))*100,2)
                aposs = round((len(apossdf)/(len(hpossdf)+len(apossdf)))*100,2)
                #Field Tilt%
                hftdf = df[(df['teamName']==hteamName) & (df['isTouch']==1) & (df['x']>=70)]
                aftdf = df[(df['teamName']==ateamName) & (df['isTouch']==1) & (df['x']>=70)]
                hft = round((len(hftdf)/(len(hftdf)+len(aftdf)))*100,2)
                aft = round((len(aftdf)/(len(hftdf)+len(aftdf)))*100,2)
                #Total Passes
                htotalPass = len(df[(df['teamName']==hteamName) & (df['type']=='Pass')])
                atotalPass = len(df[(df['teamName']==ateamName) & (df['type']=='Pass')])
                #Accurate Pass
                hAccPass = len(df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['outcomeType']=='Successful')])
                aAccPass = len(df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['outcomeType']=='Successful')])
                #Accurate Pass (without defensive third)
                hAccPasswdt = len(df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['outcomeType']=='Successful') & (df['endX']>35)])
                aAccPasswdt = len(df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['outcomeType']=='Successful') & (df['endX']>35)])
                #LongBall
                hLongB = len(df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Longball')) & (~df['qualifiers'].str.contains('Corner')) & (~df['qualifiers'].str.contains('Cross'))])
                aLongB = len(df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Longball')) & (~df['qualifiers'].str.contains('Corner')) & (~df['qualifiers'].str.contains('Cross'))])
                #Accurate LongBall
                hAccLongB = len(df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Longball')) & (df['outcomeType']=='Successful') & (~df['qualifiers'].str.contains('Corner')) & (~df['qualifiers'].str.contains('Cross'))])
                aAccLongB = len(df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Longball')) & (df['outcomeType']=='Successful') & (~df['qualifiers'].str.contains('Corner')) & (~df['qualifiers'].str.contains('Cross'))])
                #Crosses
                hCrss= len(df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Cross'))])
                aCrss= len(df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Cross'))])
                #Accurate Crosses
                hAccCrss= len(df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Cross')) & (df['outcomeType']=='Successful')])
                aAccCrss= len(df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Cross')) & (df['outcomeType']=='Successful')])
                #Freekick
                hfk= len(df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Freekick'))])
                afk= len(df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Freekick'))])
                #Corner
                hCor= len(df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Corner'))])
                aCor= len(df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Corner'))])
                #ThrowIn
                htins= len(df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('ThrowIn'))])
                atins= len(df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('ThrowIn'))])
                #GoalKicks
                hglkk= len(df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('GoalKick'))])
                aglkk= len(df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('GoalKick'))])
                #Dribbling
                htotalDrb = len(df[(df['teamName']==hteamName) & (df['type']=='TakeOn') & (df['qualifiers'].str.contains('Offensive'))])
                atotalDrb = len(df[(df['teamName']==ateamName) & (df['type']=='TakeOn') & (df['qualifiers'].str.contains('Offensive'))])
                #Accurate TakeOn
                hAccDrb = len(df[(df['teamName']==hteamName) & (df['type']=='TakeOn') & (df['qualifiers'].str.contains('Offensive')) & (df['outcomeType']=='Successful')])
                aAccDrb = len(df[(df['teamName']==ateamName) & (df['type']=='TakeOn') & (df['qualifiers'].str.contains('Offensive')) & (df['outcomeType']=='Successful')])
                #GoalKick Length
                home_goalkick = df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('GoalKick'))]
                away_goalkick = df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('GoalKick'))]
                import ast
                if len(home_goalkick) != 0:
                    # Convert 'qualifiers' column from string to list of dictionaries
                    home_goalkick['qualifiers'] = home_goalkick['qualifiers'].apply(ast.literal_eval)
                    # Function to extract value of 'Length'
                    def extract_length(qualifiers):
                        for item in qualifiers:
                            if 'displayName' in item['type'] and item['type']['displayName'] == 'Length':
                                return float(item['value'])
                        return None
                    # Apply the function to the 'qualifiers' column
                    home_goalkick['length'] = home_goalkick['qualifiers'].apply(extract_length).astype(float)
                    hglkl = round(home_goalkick['length'].mean(),2)

                else:
                    hglkl = 0

                if len(away_goalkick) != 0:
                    # Convert 'qualifiers' column from string to list of dictionaries
                    away_goalkick['qualifiers'] = away_goalkick['qualifiers'].apply(ast.literal_eval)
                    # Function to extract value of 'Length'
                    def extract_length(qualifiers):
                        for item in qualifiers:
                            if 'displayName' in item['type'] and item['type']['displayName'] == 'Length':
                                return float(item['value'])
                        return None
                    # Apply the function to the 'qualifiers' column
                    away_goalkick['length'] = away_goalkick['qualifiers'].apply(extract_length).astype(float)
                    aglkl = round(away_goalkick['length'].mean(),2)

                else:
                    aglkl = 0


                # Defensive Stats

                #Tackles
                htkl = len(df[(df['teamName']==hteamName) & (df['type']=='Tackle')])
                atkl = len(df[(df['teamName']==ateamName) & (df['type']=='Tackle')])
                #Tackles Won
                htklw = len(df[(df['teamName']==hteamName) & (df['type']=='Tackle') & (df['outcomeType']=='Successful')])
                atklw = len(df[(df['teamName']==ateamName) & (df['type']=='Tackle') & (df['outcomeType']=='Successful')])
                #Interceptions
                hintc= len(df[(df['teamName']==hteamName) & (df['type']=='Interception')])
                aintc= len(df[(df['teamName']==ateamName) & (df['type']=='Interception')])
                #Clearances
                hclr= len(df[(df['teamName']==hteamName) & (df['type']=='Clearance')])
                aclr= len(df[(df['teamName']==ateamName) & (df['type']=='Clearance')])
                #Aerials
                harl= len(df[(df['teamName']==hteamName) & (df['type']=='Aerial')])
                aarl= len(df[(df['teamName']==ateamName) & (df['type']=='Aerial')])
                #Aerials Wins
                harlw= len(df[(df['teamName']==hteamName) & (df['type']=='Aerial') & (df['outcomeType']=='Successful')])
                aarlw= len(df[(df['teamName']==ateamName) & (df['type']=='Aerial') & (df['outcomeType']=='Successful')])
                #BallRecovery
                hblrc= len(df[(df['teamName']==hteamName) & (df['type']=='BallRecovery')])
                ablrc= len(df[(df['teamName']==ateamName) & (df['type']=='BallRecovery')])
                #BlockedPass
                hblkp= len(df[(df['teamName']==hteamName) & (df['type']=='BlockedPass')])
                ablkp= len(df[(df['teamName']==ateamName) & (df['type']=='BlockedPass')])
                #OffsideGiven
                hofs= len(df[(df['teamName']==hteamName) & (df['type']=='OffsideGiven')])
                aofs= len(df[(df['teamName']==ateamName) & (df['type']=='OffsideGiven')])
                #Fouls
                hfoul= len(df[(df['teamName']==hteamName) & (df['type']=='Foul')])
                afoul= len(df[(df['teamName']==ateamName) & (df['type']=='Foul')])

                # PPDA
                home_def_acts = df[(df['teamName']==hteamName) & (df['type'].str.contains('Interception|Foul|Challenge|BlockedPass|Tackle')) & (df['x']>35)]
                away_def_acts = df[(df['teamName']==ateamName) & (df['type'].str.contains('Interception|Foul|Challenge|BlockedPass|Tackle')) & (df['x']>35)]
                home_pass = df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['outcomeType']=='Successful') & (df['x']<70)]
                away_pass = df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['outcomeType']=='Successful') & (df['x']<70)]
                home_ppda = round((len(away_pass)/len(home_def_acts)), 2)
                away_ppda = round((len(home_pass)/len(away_def_acts)), 2)

                # Average Passes per Sequence
                pass_df_home = df[(df['type'] == 'Pass') & (df['teamName']==hteamName)]
                pass_counts_home = pass_df_home.groupby('possession_id').size()
                PPS_home = pass_counts_home.mean().round()
                pass_df_away = df[(df['type'] == 'Pass') & (df['teamName']==ateamName)]
                pass_counts_away = pass_df_away.groupby('possession_id').size()
                PPS_away = pass_counts_away.mean().round()

                # Number of Sequence with 10+ Passes
                possessions_with_10_or_more_passes = pass_counts_home[pass_counts_home >= 10]
                pass_seq_10_more_home = possessions_with_10_or_more_passes.count()
                possessions_with_10_or_more_passes = pass_counts_away[pass_counts_away >= 10]
                pass_seq_10_more_away = possessions_with_10_or_more_passes.count()

                path_eff1 = [path_effects.Stroke(linewidth=1.5, foreground=line_color), path_effects.Normal()]

                def plotting_match_stats(ax):
                    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=bg_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    ax.set_ylim(-5, 68.5)

                    # plotting the headline box
                    head_y = [62,68,68,62]
                    head_x = [0,0,105,105]
                    ax.fill(head_x, head_y, 'orange')
                    ax.text(52.5,64.5, "Match Stats", ha='center', va='center', color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)

                    # Stats bar diagram
                    stats_title = [58, 58-(1*6), 58-(2*6), 58-(3*6), 58-(4*6), 58-(5*6), 58-(6*6), 58-(7*6), 58-(8*6), 58-(9*6), 58-(10*6)] # y co-ordinate values of the bars
                    stats_home = [hposs, hft, htotalPass, hLongB, htkl, hintc, hclr, harl, home_ppda, PPS_home, pass_seq_10_more_home]
                    stats_away = [aposs, aft, atotalPass, aLongB, atkl, aintc, aclr, aarl, away_ppda, PPS_away, pass_seq_10_more_away]

                    stats_normalized_home = [-(hposs/(hposs+aposs))*50, -(hft/(hft+aft))*50, -(htotalPass/(htotalPass+atotalPass))*50,
                                            -(hLongB/(hLongB+aLongB))*50, -(htkl/(htkl+atkl))*50,       # put a (-) sign before each value so that the
                                            -(hintc/(hintc+aintc))*50, -(hclr/(hclr+aclr))*50, -(harl/(harl+aarl))*50, -(home_ppda/(home_ppda+away_ppda))*50,
                                            -(PPS_home/(PPS_home+PPS_away))*50, -(pass_seq_10_more_home/(pass_seq_10_more_home+pass_seq_10_more_away))*50]          # home stats bar shows in the opposite of away
                    stats_normalized_away = [(aposs/(hposs+aposs))*50, (aft/(hft+aft))*50, (atotalPass/(htotalPass+atotalPass))*50,
                                            (aLongB/(hLongB+aLongB))*50, (atkl/(htkl+atkl))*50,
                                            (aintc/(hintc+aintc))*50, (aclr/(hclr+aclr))*50, (aarl/(harl+aarl))*50, (away_ppda/(home_ppda+away_ppda))*50,
                                            (PPS_away/(PPS_home+PPS_away))*50, (pass_seq_10_more_away/(pass_seq_10_more_home+pass_seq_10_more_away))*50]

                    start_x = 52.5
                    ax.barh(stats_title, stats_normalized_home, height=4, color=col1, left=start_x)
                    ax.barh(stats_title, stats_normalized_away, height=4, left=start_x, color=col2)
                    # Turn off axis-related elements
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Plotting the texts
                    ax.text(52.5, 58, "Possession", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
                    ax.text(52.5, 58-(1*6), "Field Tilt", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
                    ax.text(52.5, 58-(2*6), "Passes (Acc.)", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
                    ax.text(52.5, 58-(3*6), "LongBalls (Acc.)", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
                    ax.text(52.5, 58-(4*6), "Tackles (Wins)", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
                    ax.text(52.5, 58-(5*6), "Interceptions", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
                    ax.text(52.5, 58-(6*6), "Clearence", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
                    ax.text(52.5, 58-(7*6), "Aerials (Wins)", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
                    ax.text(52.5, 58-(8*6), "PPDA", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
                    ax.text(52.5, 58-(9*6), "Pass/Sequence", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
                    ax.text(52.5, 58-(10*6), "10+Pass Seq.", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)

                    ax.text(0, 58, f"{round(hposs)}%", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
                    ax.text(0, 58-(1*6), f"{round(hft)}%", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
                    ax.text(0, 58-(2*6), f"{htotalPass}({hAccPass})", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
                    ax.text(0, 58-(3*6), f"{hLongB}({hAccLongB})", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
                    ax.text(0, 58-(4*6), f"{htkl}({htklw})", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
                    ax.text(0, 58-(5*6), f"{hintc}", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
                    ax.text(0, 58-(6*6), f"{hclr}", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
                    ax.text(0, 58-(7*6), f"{harl}({harlw})", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
                    ax.text(0, 58-(8*6), f"{home_ppda}", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
                    ax.text(0, 58-(9*6), f"{int(PPS_home)}", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
                    ax.text(0, 58-(10*6), f"{pass_seq_10_more_home}", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')

                    ax.text(105, 58, f"{round(aposs)}%", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
                    ax.text(105, 58-(1*6), f"{round(aft)}%", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
                    ax.text(105, 58-(2*6), f"{atotalPass}({aAccPass})", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
                    ax.text(105, 58-(3*6), f"{aLongB}({aAccLongB})", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
                    ax.text(105, 58-(4*6), f"{atkl}({atklw})", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
                    ax.text(105, 58-(5*6), f"{aintc}", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
                    ax.text(105, 58-(6*6), f"{aclr}", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
                    ax.text(105, 58-(7*6), f"{aarl}({aarlw})", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
                    ax.text(105, 58-(8*6), f"{away_ppda}", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
                    ax.text(105, 58-(9*6), f"{int(PPS_away)}", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
                    ax.text(105, 58-(10*6), f"{pass_seq_10_more_away}", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')

                    home_data = {
                        'Team_Name': hteamName,
                        'Possession_%': hposs,
                        'Field_Tilt_%': hft,
                        'Total_Passes': htotalPass,
                        'Accurate_Passes': hAccPass,
                        'Longballs': hLongB,
                        'Accurate_Longballs': hAccLongB,
                        'Corners': hCor,
                        'Avg.GoalKick_Length': hglkl,
                        'Tackles': htkl,
                        'Tackles_Won': htklw,
                        'Interceptions': hintc,
                        'Clearances': hclr,
                        'Aerial_Duels': harl,
                        'Aerial_Duels_Won': harlw,
                        'Passes_Per_Defensive_Actions(PPDA)': home_ppda,
                        'Average_Passes_Per_Sequences': PPS_home,
                        '10+_Passing_Sequences': pass_seq_10_more_home
                    }
                    
                    away_data = {
                        'Team_Name': ateamName,
                        'Possession_%': aposs,
                        'Field_Tilt_%': aft,
                        'Total_Passes': atotalPass,
                        'Accurate_Passes': aAccPass,
                        'Longballs': aLongB,
                        'Accurate_Longballs': aAccLongB,
                        'Corners': aCor,
                        'Avg.GoalKick_Length': aglkl,
                        'Tackles': atkl,
                        'Tackles_Won': atklw,
                        'Interceptions': aintc,
                        'Clearances': aclr,
                        'Aerial_Duels': aarl,
                        'Aerial_Duels_Won': aarlw,
                        'Passes_Per_Defensive_Actions(PPDA)': away_ppda,
                        'Average_Passes_Per_Sequences': PPS_away,
                        '10+_Passing_Sequences': pass_seq_10_more_away
                    }
                    
                    return [home_data, away_data]

                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)

                # Call the `plotting_match_stats` function to generate the plot and stats
                general_match_stats = plotting_match_stats(ax)

                # Convert the match statistics to a DataFrame
                general_match_stats_df = pd.DataFrame(general_match_stats)

                # Display the match stats visualization in Streamlit
                st.header("General Match Statistics")
                st.pyplot(fig)

                # Display the general match statistics DataFrame in Streamlit
                #st.dataframe(general_match_stats_df)
                def Final_third_entry(ax, team_name, col):
                    dfpass = df[(df['teamName']==team_name) & (df['type']=='Pass') & (df['x']<70) & (df['endX']>=70) & (df['outcomeType']=='Successful') &
                                (~df['qualifiers'].str.contains('Freekick'))]
                    dfcarry = df[(df['teamName']==team_name) & (df['type']=='Carry') & (df['x']<70) & (df['endX']>=70)]
                    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2,
                                        corner_arcs=True)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    # ax.set_ylim(-0.5, 68.5)

                    if team_name == ateamName:
                        ax.invert_xaxis()
                        ax.invert_yaxis()

                    pass_count = len(dfpass) + len(dfcarry)

                    # calculating the counts
                    left_entry = len(dfpass[dfpass['y']>=45.33]) + len(dfcarry[dfcarry['y']>=45.33])
                    mid_entry = len(dfpass[(dfpass['y']>=22.67) & (dfpass['y']<45.33)]) + len(dfcarry[(dfcarry['y']>=22.67) & (dfcarry['y']<45.33)])
                    right_entry = len(dfpass[(dfpass['y']>=0) & (dfpass['y']<22.67)]) + len(dfcarry[(dfcarry['y']>=0) & (dfcarry['y']<22.67)])
                    left_percentage = round((left_entry/pass_count)*100)
                    mid_percentage = round((mid_entry/pass_count)*100)
                    right_percentage = round((right_entry/pass_count)*100)

                    ax.hlines(22.67, xmin=0, xmax=70, colors=line_color, linestyle='dashed', alpha=0.35)
                    ax.hlines(45.33, xmin=0, xmax=70, colors=line_color, linestyle='dashed', alpha=0.35)
                    ax.vlines(70, ymin=-2, ymax=70, colors=line_color, linestyle='dashed', alpha=0.55)

                    # showing the texts in the pitch
                    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
                    if col == hcol:
                        ax.text(8, 11.335, f'{right_entry}\n({right_percentage}%)', color=hcol, fontsize=24, va='center', ha='center', bbox=bbox_props)
                        ax.text(8, 34, f'{mid_entry}\n({mid_percentage}%)', color=hcol, fontsize=24, va='center', ha='center', bbox=bbox_props)
                        ax.text(8, 56.675, f'{left_entry}\n({left_percentage}%)', color=hcol, fontsize=24, va='center', ha='center', bbox=bbox_props)
                    else:
                        ax.text(8, 11.335, f'{right_entry}\n({right_percentage}%)', color=acol, fontsize=24, va='center', ha='center', bbox=bbox_props)
                        ax.text(8, 34, f'{mid_entry}\n({mid_percentage}%)', color=acol, fontsize=24, va='center', ha='center', bbox=bbox_props)
                        ax.text(8, 56.675, f'{left_entry}\n({left_percentage}%)', color=acol, fontsize=24, va='center', ha='center', bbox=bbox_props)

                    # plotting the passes
                    pro_pass = pitch.lines(dfpass.x, dfpass.y, dfpass.endX, dfpass.endY, lw=3.5, comet=True, color=col, ax=ax, alpha=0.5)
                    # plotting some scatters at the end of each pass
                    pro_pass_end = pitch.scatter(dfpass.endX, dfpass.endY, s=35, edgecolor=col, linewidth=1, color=bg_color, zorder=2, ax=ax)
                    # plotting carries
                    for index, row in dfcarry.iterrows():
                        arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', color=col, zorder=4, mutation_scale=20, 
                                                        alpha=1, linewidth=2, linestyle='--')
                        ax.add_patch(arrow)

                    counttext = f"{pass_count} Final Third Entries"

                    # Heading and other texts
                    if col == hcol:
                        ax.set_title(f"{hteamName}\n{counttext}", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
                        ax.text(87.5, 70, '<---------------- Final third ---------------->', color=line_color, ha='center', va='center')
                        pitch.lines(53, -2, 73, -2, lw=3, transparent=True, comet=True, color=col, ax=ax, alpha=0.5)
                        ax.scatter(73,-2, s=35, edgecolor=col, linewidth=1, color=bg_color, zorder=2)
                        arrow = patches.FancyArrowPatch((83, -2), (103, -2), arrowstyle='->', color=col, zorder=4, mutation_scale=20, 
                                                        alpha=1, linewidth=2, linestyle='--')
                        ax.add_patch(arrow)
                        ax.text(63, -5, f'Entry by Pass: {len(dfpass)}', fontsize=15, color=line_color, ha='center', va='center')
                        ax.text(93, -5, f'Entry by Carry: {len(dfcarry)}', fontsize=15, color=line_color, ha='center', va='center')
                        
                    else:
                        ax.set_title(f"{ateamName}\n{counttext}", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
                        ax.text(87.5, -2, '<---------------- Final third ---------------->', color=line_color, ha='center', va='center')
                        pitch.lines(53, 70, 73, 70, lw=3, transparent=True, comet=True, color=col, ax=ax, alpha=0.5)
                        ax.scatter(73,70, s=35, edgecolor=col, linewidth=1, color=bg_color, zorder=2)
                        arrow = patches.FancyArrowPatch((83, 70), (103, 70), arrowstyle='->', color=col, zorder=4, mutation_scale=20, 
                                                        alpha=1, linewidth=2, linestyle='--')
                        ax.add_patch(arrow)
                        ax.text(63, 73, f'Entry by Pass: {len(dfpass)}', fontsize=15, color=line_color, ha='center', va='center')
                        ax.text(93, 73, f'Entry by Carry: {len(dfcarry)}', fontsize=15, color=line_color, ha='center', va='center')

                    return {
                        'Team_Name': team_name,
                        'Total_Final_Third_Entries': pass_count,
                        'Final_Third_Entries_From_Left': left_entry,
                        'Final_Third_Entries_From_Center': mid_entry,
                        'Final_Third_Entries_From_Right': right_entry,
                        'Entry_By_Pass': len(dfpass),
                        'Entry_By_Carry': len(dfcarry)
                    }

                fig, axs = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)

                # Generate visualizations and stats for both home and away teams
                final_third_entry_stats_home = Final_third_entry(axs[0], hteamName, hcol)
                final_third_entry_stats_away = Final_third_entry(axs[1], ateamName, acol)

                # Collect the stats into a list and convert to a DataFrame
                final_third_entry_stats_list = [
                    final_third_entry_stats_home,
                    final_third_entry_stats_away
                ]
                final_third_entry_stats_df = pd.DataFrame(final_third_entry_stats_list)

                # Display the visualizations in Streamlit
                st.header("Final Third Entry Statistics")
                st.pyplot(fig)

                # Display the final third entry statistics DataFrame in Streamlit
                #st.dataframe(final_third_entry_stats_df)
                def zone14hs(ax, team_name, col):
                    
                    dfhp = df[(df['teamName']==team_name) & (df['type']=='Pass') & (df['outcomeType']=='Successful') & 
                            (~df['qualifiers'].str.contains('CornerTaken|Freekick'))]
                    
                    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color,  linewidth=2,
                                        corner_arcs=True)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    ax.set_facecolor(bg_color)
                    if team_name == ateamName:
                        ax.invert_xaxis()
                        ax.invert_yaxis()

                    # setting the count varibale
                    z14 = 0
                    hs = 0
                    lhs = 0
                    rhs = 0

                    path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]
                    # iterating ecah pass and according to the conditions plotting only zone14 and half spaces passes
                    for index, row in dfhp.iterrows():
                        if row['endX'] >= 70 and row['endX'] <= 88.54 and row['endY'] >= 22.66 and row['endY'] <= 45.32:
                            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color='orange', comet=True, lw=3, zorder=3, ax=ax, alpha=0.75)
                            ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor='orange', zorder=4)
                            z14 += 1
                        if row['endX'] >= 70 and row['endY'] >= 11.33 and row['endY'] <= 22.66:
                            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=col, comet=True, lw=3, zorder=3, ax=ax, alpha=0.75)
                            ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor=col, zorder=4)
                            hs += 1
                            rhs += 1
                        if row['endX'] >= 70 and row['endY'] >= 45.32 and row['endY'] <= 56.95:
                            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=col, comet=True, lw=3, zorder=3, ax=ax, alpha=0.75)
                            ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor=col, zorder=4)
                            hs += 1
                            lhs += 1

                    # coloring those zones in the pitch
                    y_z14 = [22.66, 22.66, 45.32, 45.32]
                    x_z14 = [70, 88.54, 88.54, 70]
                    ax.fill(x_z14, y_z14, 'orange', alpha=0.2, label='Zone14')

                    y_rhs = [11.33, 11.33, 22.66, 22.66]
                    x_rhs = [70, 105, 105, 70]
                    ax.fill(x_rhs, y_rhs, col, alpha=0.2, label='HalfSpaces')

                    y_lhs = [45.32, 45.32, 56.95, 56.95]
                    x_lhs = [70, 105, 105, 70]
                    ax.fill(x_lhs, y_lhs, col, alpha=0.2, label='HalfSpaces')

                    # showing the counts in an attractive way
                    z14name = "Zone14"
                    hsname = "HalfSp"
                    z14count = f"{z14}"
                    hscount = f"{hs}"
                    ax.scatter(16.46, 13.85, color=col, s=15000, edgecolor=line_color, linewidth=2, alpha=1, marker='h')
                    ax.scatter(16.46, 54.15, color='orange', s=15000, edgecolor=line_color, linewidth=2, alpha=1, marker='h')
                    ax.text(16.46, 13.85-4, hsname, fontsize=20, color=line_color, ha='center', va='center', path_effects=path_eff)
                    ax.text(16.46, 54.15-4, z14name, fontsize=20, color=line_color, ha='center', va='center', path_effects=path_eff)
                    ax.text(16.46, 13.85+2, hscount, fontsize=40, color=line_color, ha='center', va='center', path_effects=path_eff)
                    ax.text(16.46, 54.15+2, z14count, fontsize=40, color=line_color, ha='center', va='center', path_effects=path_eff)

                    # Headings and other texts
                    if col == hcol:
                        ax.set_title(f"{hteamName}\nZone14 & Halfsp. Pass", color=line_color, fontsize=25, fontweight='bold')
                    else:
                        ax.set_title(f"{ateamName}\nZone14 & Halfsp. Pass", color=line_color, fontsize=25, fontweight='bold')

                    return {
                        'Team_Name': team_name,
                        'Total_Passes_Into_Zone14': z14,
                        'Passes_Into_Halfspaces': hs,
                        'Passes_Into_Left_Halfspaces': lhs,
                        'Passes_Into_Right_Halfspaces': rhs
                    }

                fig, axs = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)

                # Generate zonal passing visualizations and stats for both home and away teams
                zonal_passing_stats_home = zone14hs(axs[0], hteamName, hcol)
                zonal_passing_stats_away = zone14hs(axs[1], ateamName, acol)

                # Collect the stats into a list and convert to a DataFrame
                zonal_passing_stats_list = [
                    zonal_passing_stats_home,
                    zonal_passing_stats_away
                ]
                zonal_passing_stats_df = pd.DataFrame(zonal_passing_stats_list)

                # Display the visualizations in Streamlit
                st.header("Zonal Passing Statistics")
                st.pyplot(fig)
                # Display the zonal passing statistics DataFrame in Streamlit
                #st.dataframe(zonal_passing_stats_df)
                # setting the custom colormap
                pearl_earring_cmaph = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",  [bg_color, hcol], N=20)
                pearl_earring_cmapa = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",  [bg_color, acol], N=20)

                path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]

                def Pass_end_zone(ax, team_name, cm):
                    pez = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful')]
                    pitch = Pitch(pitch_type='uefa', line_color=line_color, goal_type='box', goal_alpha=.5, corner_arcs=True, line_zorder=2, pitch_color=bg_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    if team_name == ateamName:
                        ax.invert_xaxis()
                        ax.invert_yaxis()

                    pearl_earring_cmap = cm
                    # binning the data points
                    # bin_statistic = pitch.bin_statistic_positional(df.endX, df.endY, statistic='count', positional='full', normalize=True)
                    bin_statistic = pitch.bin_statistic(pez.endX, pez.endY, bins=(6, 5), normalize=True)
                    pitch.heatmap(bin_statistic, ax=ax, cmap=pearl_earring_cmap, edgecolors=bg_color)
                    labels = pitch.label_heatmap(bin_statistic, color=line_color, fontsize=25, ax=ax, ha='center', va='center', str_format='{:.0%}', path_effects=path_eff)

                    # Headings and other texts
                    if team_name == hteamName:
                        ax.set_title(f"{hteamName}\nPass End Zone", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
                    else:
                        ax.set_title(f"{ateamName}\nPass End Zone", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
                # Define custom colormaps for home and away teams
                pearl_earring_cmaph = LinearSegmentedColormap.from_list("Pearl Earring - Home", [bg_color, hcol], N=20)
                pearl_earring_cmapa = LinearSegmentedColormap.from_list("Pearl Earring - Away", [bg_color, acol], N=20)

                # Create subplots for Pass End Zone visualizations
                fig, axs = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)

                # Generate the Pass End Zone plots for home and away teams
                Pass_end_zone(axs[0], hteamName, pearl_earring_cmaph)
                Pass_end_zone(axs[1], ateamName, pearl_earring_cmapa)

                # Display the visualization in Streamlit
                st.header("Pass End Zone")
                st.pyplot(fig)

                # Add headers for better organization


                def Chance_creating_zone(ax, team_name, cm, col):
                    ccp = df[(df['qualifiers'].str.contains('KeyPass')) & (df['teamName']==team_name)]
                    pitch = Pitch(pitch_type='uefa', line_color=line_color, corner_arcs=True, line_zorder=2, pitch_color=bg_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    if team_name == ateamName:
                        ax.invert_xaxis()
                        ax.invert_yaxis()

                    cc = 0
                    pearl_earring_cmap = cm
                    # bin_statistic = pitch.bin_statistic_positional(df.x, df.y, statistic='count', positional='full', normalize=False)
                    bin_statistic = pitch.bin_statistic(ccp.x, ccp.y, bins=(6,5), statistic='count', normalize=False)
                    pitch.heatmap(bin_statistic, ax=ax, cmap=pearl_earring_cmap, edgecolors='#f8f8f8')
                    # pitch.scatter(ccp.x, ccp.y, c='gray', s=5, ax=ax)
                    for index, row in ccp.iterrows():
                        if 'IntentionalGoalAssist' in row['qualifiers']:
                            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=green, comet=True, lw=3, zorder=3, ax=ax)
                            ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor=green, zorder=4)
                            cc += 1
                    else :
                        pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=violet, comet=True, lw=3, zorder=3, ax=ax)
                        ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor=violet, zorder=4)
                        cc += 1
                    labels = pitch.label_heatmap(bin_statistic, color=line_color, fontsize=25, ax=ax, ha='center', va='center', str_format='{:.0f}', path_effects=path_eff)
                    teamName = team_name

                    # Headings and other texts
                    if col == hcol:
                        ax.text(105,-3.5, "violet = key pass\ngreen = assist", color=hcol, size=15, ha='right', va='center')
                        ax.text(52.5,70, f"Total Chances Created = {cc}", color=col, fontsize=15, ha='center', va='center')
                        ax.set_title(f"{hteamName}\nChance Creating Zone", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
                    else:
                        ax.text(105,71.5, "violet = key pass\ngreen = assist", color=acol, size=15, ha='left', va='center')
                        ax.text(52.5,-2, f"Total Chances Created = {cc}", color=col, fontsize=15, ha='center', va='center')
                        ax.set_title(f"{ateamName}\nChance Creating Zone", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)

                    return {
                        'Team_Name': team_name,
                        'Total_Chances_Created': cc
                    }

                # Define custom colormaps for home and away teams
                pearl_earring_cmaph = LinearSegmentedColormap.from_list("Pearl Earring - Home", [bg_color, hcol], N=20)
                pearl_earring_cmapa = LinearSegmentedColormap.from_list("Pearl Earring - Away", [bg_color, acol], N=20)

                # Create subplots for Chance Creating Zone visualizations
                fig, axs = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)

                # Generate visualizations and stats for home and away teams
                chance_creating_stats_home = Chance_creating_zone(axs[0], hteamName, pearl_earring_cmaph, hcol)
                chance_creating_stats_away = Chance_creating_zone(axs[1], ateamName, pearl_earring_cmapa, acol)

                # Collect stats into a list and convert to a DataFrame
                chance_creating_stats_list = [
                    chance_creating_stats_home,
                    chance_creating_stats_away
                ]
                chance_creating_stats_df = pd.DataFrame(chance_creating_stats_list)

                # Display the visualizations in Streamlit
                st.header("Chance Creating Zone Statistics")

                st.pyplot(fig)

                # Display the chance creating statistics DataFrame in Streamlit
                #st.dataframe(chance_creating_stats_df)

                def box_entry(ax):
                    bentry = df[((df['type']=='Pass')|(df['type']=='Carry')) & (df['outcomeType']=='Successful') & (df['endX']>=88.5) &
                                ~((df['x']>=88.5) & (df['y']>=13.6) & (df['y']<=54.6)) & (df['endY']>=13.6) & (df['endY']<=54.4) &
                            (~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn'))]
                    hbentry = bentry[bentry['teamName']==hteamName]
                    abentry = bentry[bentry['teamName']==ateamName]

                    hrigt = hbentry[hbentry['y']<68/3]
                    hcent = hbentry[(hbentry['y']>=68/3) & (hbentry['y']<=136/3)]
                    hleft = hbentry[hbentry['y']>136/3]

                    arigt = abentry[(abentry['y']<68/3)]
                    acent = abentry[(abentry['y']>=68/3) & (abentry['y']<=136/3)]
                    aleft = abentry[(abentry['y']>136/3)]

                    pitch = Pitch(pitch_type='uefa', line_color=line_color, corner_arcs=True, line_zorder=2, pitch_color=bg_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    ax.set_ylim(-0.5, 68.5)
                    hdfpass = 0
                    hdfcarry = 0
                    adfpass = 0
                    adfcarry = 0
                    for index, row in bentry.iterrows():
                        if row['teamName'] == hteamName:
                            color = hcol
                            x, y, endX, endY = 105 - row['x'], 68 - row['y'], 105 - row['endX'], 68 - row['endY']
                            
                            if row['type'] == 'Pass':
                                # Draw pass line and scatter point
                                pitch.lines(x, y, endX, endY, lw=0.5, comet=True, color=color, ax=ax, alpha=0.5)
                                pitch.scatter(endX, endY, s=5, edgecolor=color, linewidth=0.5, color=bg_color, zorder=2, ax=ax)
                                hdfpass += 1  # Increment pass count
                                
                            elif row['type'] == 'Carry':
                                # Draw carry arrow
                                arrow = patches.FancyArrowPatch(
                                    (x, y), (endX, endY), arrowstyle='->', color=color, zorder=4, 
                                    mutation_scale=5, alpha=1, linewidth=0.5, linestyle='--'
                                )
                                ax.add_patch(arrow)
                                hdfcarry += 1  # Increment carry count
                    for index, row in bentry.iterrows():
                        if row['teamName'] == ateamName:
                            color = acol
                            x, y, endX, endY = row['x'], row['y'], row['endX'], row['endY']
                            
                            if row['type'] == 'Pass':
                                # Draw pass line and scatter point
                                pitch.lines(x, y, endX, endY, lw=0.5, comet=True, color=color, ax=ax, alpha=0.5)
                                pitch.scatter(endX, endY, s=5, edgecolor=color, linewidth=0.5, color=bg_color, zorder=2, ax=ax)
                                adfpass += 1  # Increment pass count
                                
                            elif row['type'] == 'Carry':
                                # Draw carry arrow
                                arrow = patches.FancyArrowPatch(
                                    (x, y), (endX, endY), arrowstyle='->', color=color, zorder=4, 
                                    mutation_scale=5, alpha=1, linewidth=0.5, linestyle='--'
                                )
                                ax.add_patch(arrow)
                                adfcarry += 1  # Increment carry count
                    
                    ax.text(0, 69, f'{hteamName}\nBox Entries: {len(hbentry)}', color=hcol, fontsize=25, fontweight='bold', ha='left', va='bottom')
                    ax.text(105, 69, f'{ateamName}\nBox Entries: {len(abentry)}', color=acol, fontsize=25, fontweight='bold', ha='right', va='bottom')
                    
                    ax.scatter(46, 6, s=2000, marker='s', color=hcol, zorder=3)
                    ax.scatter(46, 34, s=2000, marker='s', color=hcol, zorder=3)
                    ax.scatter(46, 62, s=2000, marker='s', color=hcol, zorder=3)
                    ax.text(46, 6, f'{len(hleft)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
                    ax.text(46, 34, f'{len(hcent)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
                    ax.text(46, 62, f'{len(hrigt)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
                    
                    ax.scatter(59.5, 6, s=2000, marker='s', color=acol, zorder=3)
                    ax.scatter(59.5, 34, s=2000, marker='s', color=acol, zorder=3)
                    ax.scatter(59.5, 62, s=2000, marker='s', color=acol, zorder=3)
                    ax.text(59.5, 6, f'{len(arigt)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
                    ax.text(59.5, 34, f'{len(acent)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
                    ax.text(59.5, 62, f'{len(aleft)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
                    ax.text(63, -3, f'Entry by Pass: {(adfpass)}', fontsize=12, color=acol, ha='center', va='center')
                    ax.text(93, -3, f'Entry by Carry: {(adfcarry)}', fontsize=12, color=acol, ha='center', va='center')
                    ax.text(10, -3, f'Entry by Pass: {(hdfpass)}', fontsize=12, color=hcol, ha='center', va='center')
                    ax.text(40, -3, f'Entry by Carry: {(hdfcarry)}', fontsize=12, color=hcol, ha='center', va='center')
                    home_data = {
                            'Team_Name': hteamName,
                            'Total_Box_Entries': len(hbentry),
                            'Box_Entry_From_Left': len(hleft),
                            'Box_Entry_From_Center': len(hcent),
                            'Box_Entry_From_Right': len(hrigt)
                        }
                    
                    away_data = {
                            'Team_Name': ateamName,
                            'Total_Box_Entries': len(abentry),
                            'Box_Entry_From_Left': len(aleft),
                            'Box_Entry_From_Center': len(acent),
                            'Box_Entry_From_Right': len(arigt)
                        }
                    return [home_data, away_data]


                # Create a subplot for the Box Entry visualization
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)

                # Generate the visualization and stats
                box_entry_stats = box_entry(ax)

                # Convert the box entry statistics to a DataFrame
                box_entry_stats_df = pd.DataFrame(box_entry_stats)


                # Display the visualization in Streamlit
                st.header("Box Entry Statistics")

                st.pyplot(fig)

                def Crosses(ax):
                    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_ylim(-0.5,68.5)
                    ax.set_xlim(-0.5,105.5)

                    home_cross = df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Cross')) & (~df['qualifiers'].str.contains('Corner'))]
                    away_cross = df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Cross')) & (~df['qualifiers'].str.contains('Corner'))]

                    hsuc = 0
                    hunsuc = 0
                    asuc = 0
                    aunsuc = 0

                    # iterating through each pass and coloring according to successful or not
                    for index, row in home_cross.iterrows():
                        if row['outcomeType'] == 'Successful':
                            arrow = patches.FancyArrowPatch((105-row['x'], 68-row['y']), (105-row['endX'], 68-row['endY']), arrowstyle='->', mutation_scale=15, color=hcol, linewidth=1.5, zorder=3, alpha=1)
                            ax.add_patch(arrow)
                            hsuc += 1
                        else:
                            arrow = patches.FancyArrowPatch((105-row['x'], 68-row['y']), (105-row['endX'], 68-row['endY']), arrowstyle='->', mutation_scale=10, color=line_color, linewidth=1, zorder=2, alpha=.25)
                            ax.add_patch(arrow)
                            hunsuc += 1

                    for index, row in away_cross.iterrows():
                        if row['outcomeType'] == 'Successful':
                            arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', mutation_scale=15, color=acol, linewidth=1.5, zorder=3, alpha=1)
                            ax.add_patch(arrow)
                            asuc += 1
                        else:
                            arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', mutation_scale=10, color=line_color, linewidth=1, zorder=2, alpha=.25)
                            ax.add_patch(arrow)
                            aunsuc += 1

                    # Headlines and other texts
                    home_left = len(home_cross[home_cross['y']>=34])
                    home_right = len(home_cross[home_cross['y']<34])
                    away_left = len(away_cross[away_cross['y']>=34])
                    away_right = len(away_cross[away_cross['y']<34])

                    ax.text(51, 2, f"Crosses from\nLeftwing: {home_left}", color=hcol, fontsize=15, va='bottom', ha='right')
                    ax.text(51, 66, f"Crosses from\nRightwing: {home_right}", color=hcol, fontsize=15, va='top', ha='right')
                    ax.text(54, 66, f"Crosses from\nLeftwing: {away_left}", color=acol, fontsize=15, va='top', ha='left')
                    ax.text(54, 2, f"Crosses from\nRightwing: {away_right}", color=acol, fontsize=15, va='bottom', ha='left')

                    ax.text(0,-2, f"Successful: {hsuc}", color=hcol, fontsize=20, ha='left', va='top')
                    ax.text(0,-5.5, f"Unsuccessful: {hunsuc}", color=line_color, fontsize=20, ha='left', va='top')
                    ax.text(105,-2, f"Successful: {asuc}", color=acol, fontsize=20, ha='right', va='top')
                    ax.text(105,-5.5, f"Unsuccessful: {aunsuc}", color=line_color, fontsize=20, ha='right', va='top')

                    ax.text(0, 70, f"{hteamName}\n<---Crosses", color=hcol, size=25, ha='left', fontweight='bold')
                    ax.text(105, 70, f"{ateamName}\nCrosses--->", color=acol, size=25, ha='right', fontweight='bold')

                    home_data = {
                        'Team_Name': hteamName,
                        'Total_Cross': hsuc + hunsuc,
                        'Successful_Cross': hsuc,
                        'Unsuccessful_Cross': hunsuc,
                        'Cross_From_LeftWing': home_left,
                        'Cross_From_RightWing': home_right
                    }
                    
                    away_data = {
                        'Team_Name': ateamName,
                        'Total_Cross': asuc + aunsuc,
                        'Successful_Cross': asuc,
                        'Unsuccessful_Cross': aunsuc,
                        'Cross_From_LeftWing': away_left,
                        'Cross_From_RightWing': away_right
                    }
                    
                    return [home_data, away_data]

                # Create a subplot for the Crosses visualization
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)

                # Generate the visualization and stats
                cross_stats = Crosses(ax)

                # Convert the cross statistics to a DataFrame
                cross_stats_df = pd.DataFrame(cross_stats)

                # Display the visualization in Streamlit
                st.header("Crosses Statistics")

                st.pyplot(fig)
                # Display the cross statistics DataFrame in Streamlit
                #st.dataframe(cross_stats_df)

                def HighTO(ax):
                    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_ylim(-0.5,68.5)
                    ax.set_xlim(-0.5,105.5)
                    highTO = df.copy()
                    highTO['Distance'] = ((highTO['x'] - 105)**2 + (highTO['y'] - 34)**2)**0.5
                
                    agoal_count = 0
                    # Iterate through the DataFrame
                    for i in range(len(highTO)):
                        if ((highTO.loc[i, 'type'] in ['BallRecovery', 'Interception']) and 
                            (highTO.loc[i, 'teamName'] == ateamName) and 
                            (highTO.loc[i, 'Distance'] <= 40)):
                            
                            possession_id = highTO.loc[i, 'possession_id']
                            
                            # Check the following rows within the same possession
                            j = i + 1
                            while j < len(highTO) and highTO.loc[j, 'possession_id'] == possession_id and highTO.loc[j, 'teamName']==ateamName:
                                if highTO.loc[j, 'type'] == 'Goal' and highTO.loc[j, 'teamName']==ateamName:
                                    ax.scatter(highTO.loc[i, 'x'],highTO.loc[i, 'y'], s=600, marker='*', color='green', edgecolor='k', zorder=3)
                                    # print(highTO.loc[i, 'type'])
                                    agoal_count += 1
                                    break
                                j += 1
                
                    ashot_count = 0
                    # Iterate through the DataFrame
                    for i in range(len(highTO)):
                        if ((highTO.loc[i, 'type'] in ['BallRecovery', 'Interception']) and 
                            (highTO.loc[i, 'teamName'] == ateamName) and 
                            (highTO.loc[i, 'Distance'] <= 40)):
                            
                            possession_id = highTO.loc[i, 'possession_id']
                            
                            # Check the following rows within the same possession
                            j = i + 1
                            while j < len(highTO) and highTO.loc[j, 'possession_id'] == possession_id and highTO.loc[j, 'teamName']==ateamName:
                                if ('Shot' in highTO.loc[j, 'type']) and (highTO.loc[j, 'teamName']==ateamName):
                                    ax.scatter(highTO.loc[i, 'x'],highTO.loc[i, 'y'], s=150, color=acol, edgecolor=bg_color, zorder=2)
                                    ashot_count += 1
                                    break
                                j += 1
                    
                    aht_count = 0
                    p_list = []
                    # Iterate through the DataFrame
                    for i in range(len(highTO)):
                        if ((highTO.loc[i, 'type'] in ['BallRecovery', 'Interception']) and 
                            (highTO.loc[i, 'teamName'] == ateamName) and 
                            (highTO.loc[i, 'Distance'] <= 40)):
                            
                            # Check the following rows
                            j = i + 1
                            if ((highTO.loc[j, 'teamName']==ateamName) and
                                (highTO.loc[j, 'type']!='Dispossessed') and (highTO.loc[j, 'type']!='OffsidePass')):
                                ax.scatter(highTO.loc[i, 'x'],highTO.loc[i, 'y'], s=100, color='None', edgecolor=acol)
                                aht_count += 1
                                p_list.append(highTO.loc[i, 'shortName'])
                    hgoal_count = 0
                    # Iterate through the DataFrame
                    for i in range(len(highTO)):
                        if ((highTO.loc[i, 'type'] in ['BallRecovery', 'Interception']) and 
                            (highTO.loc[i, 'teamName'] == hteamName) and 
                            (highTO.loc[i, 'Distance'] <= 40)):
                            
                            possession_id = highTO.loc[i, 'possession_id']
                            
                            # Check the following rows within the same possession
                            j = i + 1
                            while j < len(highTO) and highTO.loc[j, 'possession_id'] == possession_id and highTO.loc[j, 'teamName']==hteamName:
                                if highTO.loc[j, 'type'] == 'Goal' and highTO.loc[j, 'teamName']==hteamName:
                                    ax.scatter(105-highTO.loc[i, 'x'],68-highTO.loc[i, 'y'], s=600, marker='*', color='green', edgecolor='k', zorder=3)
                                    # print(highTO.loc[i, 'name'])
                                    hgoal_count += 1
                                    break
                                j += 1
                
                    hshot_count = 0
                    # Iterate through the DataFrame
                    for i in range(len(highTO)):
                        if ((highTO.loc[i, 'type'] in ['BallRecovery', 'Interception']) and 
                            (highTO.loc[i, 'teamName'] == hteamName) and 
                            (highTO.loc[i, 'Distance'] <= 40)):
                            
                            possession_id = highTO.loc[i, 'possession_id']
                            
                            # Check the following rows within the same possession
                            j = i + 1
                            while j < len(highTO) and highTO.loc[j, 'possession_id'] == possession_id and highTO.loc[j, 'teamName']==hteamName:
                                if ('Shot' in highTO.loc[j, 'type']) and (highTO.loc[j, 'teamName']==hteamName):
                                    ax.scatter(105-highTO.loc[i, 'x'],68-highTO.loc[i, 'y'], s=150, color=hcol, edgecolor=bg_color, zorder=2)
                                    hshot_count += 1
                                    break
                                j += 1
                    
                    hht_count = 0
                    p_list = []
                    # Iterate through the DataFrame
                    for i in range(len(highTO)):
                        if ((highTO.loc[i, 'type'] in ['BallRecovery', 'Interception']) and 
                            (highTO.loc[i, 'teamName'] == hteamName) and 
                            (highTO.loc[i, 'Distance'] <= 40)):
                            
                            # Check the following rows
                            j = i + 1
                            if ((highTO.loc[j, 'teamName']==hteamName) and
                                (highTO.loc[j, 'type']!='Dispossessed') and (highTO.loc[j, 'type']!='OffsidePass')):
                                ax.scatter(105-highTO.loc[i, 'x'],68-highTO.loc[i, 'y'], s=100, color='None', edgecolor=hcol)
                                hht_count += 1
                                p_list.append(highTO.loc[i, 'shortName'])
                
                    # Plotting the half circle
                    left_circle = plt.Circle((0,34), 40, color=hcol, fill=True, alpha=0.25, linestyle='dashed')
                    ax.add_artist(left_circle)
                    right_circle = plt.Circle((105,34), 40, color=acol, fill=True, alpha=0.25, linestyle='dashed')
                    ax.add_artist(right_circle)
                    # Set the aspect ratio to be equal
                    ax.set_aspect('equal', adjustable='box')
                    # Headlines and other texts
                    ax.text(0, 70, f"{hteamName}\nHigh Turnover: {hht_count}", color=hcol, size=25, ha='left', fontweight='bold')
                    ax.text(105, 70, f"{ateamName}\nHigh Turnover: {aht_count}", color=acol, size=25, ha='right', fontweight='bold')
                    ax.text(0,  -3, '<---Attacking Direction', color=hcol, fontsize=13, ha='left', va='center')
                    ax.text(105,-3, 'Attacking Direction--->', color=acol, fontsize=13, ha='right', va='center')
                    ax.text(40, 66.5, f'Shot Ending High Turnovers: {hshot_count}', fontsize=8, color=hcol, ha='center', va='center')
                    ax.text(40, 63.5, f'Goal Ending High Turnovers: {hgoal_count}', fontsize=8, color="green", ha='center', va='center')
                    ax.text(65, 4, f'Shot Ending High Turnovers: {ashot_count}', fontsize=8, color=acol, ha='center', va='center')
                    ax.text(65, 1, f'Goal Ending High Turnovers: {agoal_count}', fontsize=8, color="green", ha='center', va='center')
                    
                    return []

                # Create a subplot for visualization
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
                # Generate visualization and statistics
                HighTO(ax) 
                st.header("High Turnover")
                st.pyplot(fig)
                   


                #st.dataframe(high_turnover_stats_df)


                def plot_congestion(ax):
                    pcmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",  [acol, 'gray', hcol], N=20)
                    df1 = df[(df['teamName']==hteamName) & (df['isTouch']==1) & (~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn'))]
                    df2 = df[(df['teamName']==ateamName) & (df['isTouch']==1) & (~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn'))]
                    df2['x'] = 105-df2['x']
                    df2['y'] =  68-df2['y']
                    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2, line_zorder=6)
                    pitch.draw(ax=ax)
                    ax.set_ylim(-0.5,68.5)
                    ax.set_xlim(-0.5,105.5)

                    bin_statistic1 = pitch.bin_statistic(df1.x, df1.y, bins=(6,5), statistic='count', normalize=False)
                    bin_statistic2 = pitch.bin_statistic(df2.x, df2.y, bins=(6,5), statistic='count', normalize=False)

                    # Assuming 'cx' and 'cy' are as follows:
                    cx = np.array([[ 8.75, 26.25, 43.75, 61.25, 78.75, 96.25],
                            [ 8.75, 26.25, 43.75, 61.25, 78.75, 96.25],
                            [ 8.75, 26.25, 43.75, 61.25, 78.75, 96.25],
                            [ 8.75, 26.25, 43.75, 61.25, 78.75, 96.25],
                            [ 8.75, 26.25, 43.75, 61.25, 78.75, 96.25]])

                    cy = np.array([[61.2, 61.2, 61.2, 61.2, 61.2, 61.2],
                            [47.6, 47.6, 47.6, 47.6, 47.6, 47.6],
                            [34.0, 34.0, 34.0, 34.0, 34.0, 34.0],
                            [20.4, 20.4, 20.4, 20.4, 20.4, 20.4],
                            [ 6.8,  6.8,  6.8,  6.8,  6.8,  6.8]])

                    # Flatten the arrays
                    cx_flat = cx.flatten()
                    cy_flat = cy.flatten()

                    # Create a DataFrame
                    df_cong = pd.DataFrame({'cx': cx_flat, 'cy': cy_flat})

                    hd_values = []


                    # Loop through the 2D arrays
                    for i in range(bin_statistic1['statistic'].shape[0]):
                        for j in range(bin_statistic1['statistic'].shape[1]):
                            stat1 = bin_statistic1['statistic'][i, j]
                            stat2 = bin_statistic2['statistic'][i, j]
                        
                            if (stat1 / (stat1 + stat2)) > 0.55:
                                hd_values.append(1)
                            elif (stat1 / (stat1 + stat2)) < 0.45:
                                hd_values.append(0)
                            else:
                                hd_values.append(0.5)

                    df_cong['hd']=hd_values
                    bin_stat = pitch.bin_statistic(df_cong.cx, df_cong.cy, bins=(6,5), values=df_cong['hd'], statistic='sum', normalize=False)
                    pitch.heatmap(bin_stat, ax=ax, cmap=pcmap, edgecolors='#000000', lw=0, zorder=3, alpha=0.85)

                    ax_text(52.5, 71, s=f"<{hteamName}>  |  Contested  |  <{ateamName}>", highlight_textprops=[{'color':hcol}, {'color':acol}],
                            color='gray', fontsize=18, ha='center', va='center', ax=ax)
                    ax.set_title("Team's Dominating Zone", color=line_color, fontsize=30, fontweight='bold', y=1.075)
                    ax.text(0,  -3, 'Attacking Direction--->', color=hcol, fontsize=13, ha='left', va='center')
                    ax.text(105,-3, '<---Attacking Direction', color=acol, fontsize=13, ha='right', va='center')

                    ax.vlines(1*(105/6), ymin=0, ymax=68, color=bg_color, lw=2, ls='--', zorder=5)
                    ax.vlines(2*(105/6), ymin=0, ymax=68, color=bg_color, lw=2, ls='--', zorder=5)
                    ax.vlines(3*(105/6), ymin=0, ymax=68, color=bg_color, lw=2, ls='--', zorder=5)
                    ax.vlines(4*(105/6), ymin=0, ymax=68, color=bg_color, lw=2, ls='--', zorder=5)
                    ax.vlines(5*(105/6), ymin=0, ymax=68, color=bg_color, lw=2, ls='--', zorder=5)

                    ax.hlines(1*(68/5), xmin=0, xmax=105, color=bg_color, lw=2, ls='--', zorder=5)
                    ax.hlines(2*(68/5), xmin=0, xmax=105, color=bg_color, lw=2, ls='--', zorder=5)
                    ax.hlines(3*(68/5), xmin=0, xmax=105, color=bg_color, lw=2, ls='--', zorder=5)
                    ax.hlines(4*(68/5), xmin=0, xmax=105, color=bg_color, lw=2, ls='--', zorder=5)
                    
                    return

                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)

                # Generate the congestion plot
                plot_congestion(ax)


                st.header("Congestion Plot Visualization")
                st.pyplot(fig)
                home_unique_players = homedf['name'].unique()
                away_unique_players = awaydf['name'].unique()
                def get_short_name(full_name):
                    if pd.isna(full_name):
                        return full_name
                    parts = full_name.split()
                    if len(parts) == 1:
                        return full_name  # No need for short name if there's only one word
                    elif len(parts) == 2:
                        return parts[0][0] + ". " + parts[1]
                    else:
                        return parts[0][0] + ". " + parts[1][0] + ". " + " ".join(parts[2:])

                # Top Ball Progressor
                # Initialize an empty dictionary to store home players different type of pass counts
                home_progressor_counts = {'name': home_unique_players, 'Progressive Passes': [], 'Progressive Carries': [], 'LineBreaking Pass': []}
                for name in home_unique_players:
                    home_progressor_counts['Progressive Passes'].append(len(df[(df['name'] == name) & (df['prog_pass'] >= 9.144) & (df['x']>=35) & (df['outcomeType']=='Successful') & (~df['qualifiers'].str.contains('CornerTaken|Freekick'))]))
                    home_progressor_counts['Progressive Carries'].append(len(df[(df['name'] == name) & (df['prog_carry'] >= 9.144) & (df['endX']>=35)]))
                    home_progressor_counts['LineBreaking Pass'].append(len(df[(df['name'] == name) & (df['type'] == 'Pass') & (df['qualifiers'].str.contains('Throughball'))]))
                home_progressor_df = pd.DataFrame(home_progressor_counts)
                home_progressor_df['total'] = home_progressor_df['Progressive Passes']+home_progressor_df['Progressive Carries']+home_progressor_df['LineBreaking Pass']
                home_progressor_df = home_progressor_df.sort_values(by='total', ascending=False)
                home_progressor_df.reset_index(drop=True, inplace=True)
                home_progressor_df = home_progressor_df.head(5)
                home_progressor_df['shortName'] = home_progressor_df['name'].apply(get_short_name)

                # Initialize an empty dictionary to store away players different type of pass counts
                away_progressor_counts = {'name': away_unique_players, 'Progressive Passes': [], 'Progressive Carries': [], 'LineBreaking Pass': []}
                for name in away_unique_players:
                    away_progressor_counts['Progressive Passes'].append(len(df[(df['name'] == name) & (df['prog_pass'] >= 9.144) & (df['x']>=35) & (df['outcomeType']=='Successful') & (~df['qualifiers'].str.contains('CornerTaken|Freekick'))]))
                    away_progressor_counts['Progressive Carries'].append(len(df[(df['name'] == name) & (df['prog_carry'] >= 9.144) & (df['endX']>=35)]))
                    away_progressor_counts['LineBreaking Pass'].append(len(df[(df['name'] == name) & (df['type'] == 'Pass') & (df['qualifiers'].str.contains('Throughball'))]))
                away_progressor_df = pd.DataFrame(away_progressor_counts)
                away_progressor_df['total'] = away_progressor_df['Progressive Passes']+away_progressor_df['Progressive Carries']+away_progressor_df['LineBreaking Pass']
                away_progressor_df = away_progressor_df.sort_values(by='total', ascending=False)
                away_progressor_df.reset_index(drop=True, inplace=True)
                away_progressor_df = away_progressor_df.head(5)
                away_progressor_df['shortName'] = away_progressor_df['name'].apply(get_short_name)


                # Top Threate Creators
                # Initialize an empty dictionary to store home players different type of Carries counts
                home_xT_counts = {'name': home_unique_players, 'xT from Pass': [], 'xT from Carry': []}
                for name in home_unique_players:
                    home_xT_counts['xT from Pass'].append((df[(df['name'] == name) & (df['type'] == 'Pass') & (df['xT']>=0) & (df['outcomeType']=='Successful') & (~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn'))])['xT'].sum().round(2))
                    home_xT_counts['xT from Carry'].append((df[(df['name'] == name) & (df['type'] == 'Carry') & (df['xT']>=0)])['xT'].sum().round(2))
                home_xT_df = pd.DataFrame(home_xT_counts)
                home_xT_df['total'] = home_xT_df['xT from Pass']+home_xT_df['xT from Carry']
                home_xT_df = home_xT_df.sort_values(by='total', ascending=False)
                home_xT_df.reset_index(drop=True, inplace=True)
                home_xT_df = home_xT_df.head(5)
                home_xT_df['shortName'] = home_xT_df['name'].apply(get_short_name)

                # Initialize an empty dictionary to store home players different type of Carries counts
                away_xT_counts = {'name': away_unique_players, 'xT from Pass': [], 'xT from Carry': []}
                for name in away_unique_players:
                    away_xT_counts['xT from Pass'].append((df[(df['name'] == name) & (df['type'] == 'Pass') & (df['xT']>=0) & (df['outcomeType']=='Successful') & (~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn'))])['xT'].sum().round(2))
                    away_xT_counts['xT from Carry'].append((df[(df['name'] == name) & (df['type'] == 'Carry') & (df['xT']>=0)])['xT'].sum().round(2))
                away_xT_df = pd.DataFrame(away_xT_counts)
                away_xT_df['total'] = away_xT_df['xT from Pass']+away_xT_df['xT from Carry']
                away_xT_df = away_xT_df.sort_values(by='total', ascending=False)
                away_xT_df.reset_index(drop=True, inplace=True)
                away_xT_df = away_xT_df.head(5)
                away_xT_df['shortName'] = away_xT_df['name'].apply(get_short_name)


                # Shot Sequence Involvement
                df_no_carry = df[df['type']!='Carry']
                # Initialize an empty dictionary to store home players different type of shot sequence counts
                home_shot_seq_counts = {'name': home_unique_players, 'Shots': [], 'Shot Assist': [], 'Buildup to shot': []}
                # Putting counts in those lists
                for name in home_unique_players:
                    home_shot_seq_counts['Shots'].append(len(df[(df['name'] == name) & ((df['type']=='MissedShots') | (df['type']=='SavedShot') | (df['type']=='ShotOnPost') | (df['type']=='Goal'))]))
                    home_shot_seq_counts['Shot Assist'].append(len(df[(df['name'] == name) & (df['type'] == 'Pass') & (df['qualifiers'].str.contains('KeyPass'))]))
                    home_shot_seq_counts['Buildup to shot'].append(len(df_no_carry[(df_no_carry['name'] == name) & (df_no_carry['type'] == 'Pass') & (df_no_carry['qualifiers'].shift(-1).str.contains('KeyPass'))]))
                # converting that list into a dataframe
                home_sh_sq_df = pd.DataFrame(home_shot_seq_counts)
                home_sh_sq_df['total'] = home_sh_sq_df['Shots']+home_sh_sq_df['Shot Assist']+home_sh_sq_df['Buildup to shot']
                home_sh_sq_df = home_sh_sq_df.sort_values(by='total', ascending=False)
                home_sh_sq_df.reset_index(drop=True, inplace=True)
                home_sh_sq_df = home_sh_sq_df.head(5)
                home_sh_sq_df['shortName'] = home_sh_sq_df['name'].apply(get_short_name)

                # Initialize an empty dictionary to store away players different type of shot sequence counts
                away_shot_seq_counts = {'name': away_unique_players, 'Shots': [], 'Shot Assist': [], 'Buildup to shot': []}
                for name in away_unique_players:
                    away_shot_seq_counts['Shots'].append(len(df[(df['name'] == name) & ((df['type']=='MissedShots') | (df['type']=='SavedShot') | (df['type']=='ShotOnPost') | (df['type']=='Goal'))]))
                    away_shot_seq_counts['Shot Assist'].append(len(df[(df['name'] == name) & (df['type'] == 'Pass') & (df['qualifiers'].str.contains('KeyPass'))]))
                    away_shot_seq_counts['Buildup to shot'].append(len(df_no_carry[(df_no_carry['name'] == name) & (df_no_carry['type'] == 'Pass') & (df_no_carry['qualifiers'].shift(-1).str.contains('KeyPass'))]))
                away_sh_sq_df = pd.DataFrame(away_shot_seq_counts)
                away_sh_sq_df['total'] = away_sh_sq_df['Shots']+away_sh_sq_df['Shot Assist']+away_sh_sq_df['Buildup to shot']
                away_sh_sq_df = away_sh_sq_df.sort_values(by='total', ascending=False)
                away_sh_sq_df.reset_index(drop=True, inplace=True)
                away_sh_sq_df = away_sh_sq_df.head(5)
                away_sh_sq_df['shortName'] = away_sh_sq_df['name'].apply(get_short_name)


                # Top Defenders
                # Initialize an empty dictionary to store home players different type of defensive actions counts
                home_defensive_actions_counts = {'name': home_unique_players, 'Tackles': [], 'Interceptions': [], 'Clearance': []}
                for name in home_unique_players:
                    home_defensive_actions_counts['Tackles'].append(len(df[(df['name'] == name) & (df['type'] == 'Tackle') & (df['outcomeType']=='Successful')]))
                    home_defensive_actions_counts['Interceptions'].append(len(df[(df['name'] == name) & (df['type'] == 'Interception')]))
                    home_defensive_actions_counts['Clearance'].append(len(df[(df['name'] == name) & (df['type'] == 'Clearance')]))
                home_defender_df = pd.DataFrame(home_defensive_actions_counts)
                home_defender_df['total'] = home_defender_df['Tackles']+home_defender_df['Interceptions']+home_defender_df['Clearance']
                home_defender_df = home_defender_df.sort_values(by='total', ascending=False)
                home_defender_df.reset_index(drop=True, inplace=True)
                home_defender_df = home_defender_df.head(5)
                home_defender_df['shortName'] = home_defender_df['name'].apply(get_short_name)

                # Initialize an empty dictionary to store away players different type of defensive actions counts
                away_defensive_actions_counts = {'name': away_unique_players, 'Tackles': [], 'Interceptions': [], 'Clearance': []}
                for name in away_unique_players:
                    away_defensive_actions_counts['Tackles'].append(len(df[(df['name'] == name) & (df['type'] == 'Tackle') & (df['outcomeType']=='Successful')]))
                    away_defensive_actions_counts['Interceptions'].append(len(df[(df['name'] == name) & (df['type'] == 'Interception')]))
                    away_defensive_actions_counts['Clearance'].append(len(df[(df['name'] == name) & (df['type'] == 'Clearance')]))
                away_defender_df = pd.DataFrame(away_defensive_actions_counts)
                away_defender_df['total'] = away_defender_df['Tackles']+away_defender_df['Interceptions']+away_defender_df['Clearance']
                away_defender_df = away_defender_df.sort_values(by='total', ascending=False)
                away_defender_df.reset_index(drop=True, inplace=True)
                away_defender_df = away_defender_df.head(5)
                away_defender_df['shortName'] = away_defender_df['name'].apply(get_short_name)
                # Get unique players
                unique_players = df['name'].unique()


                # Top Ball Progressor
                # Initialize an empty dictionary to store home players different type of pass counts
                progressor_counts = {'name': unique_players, 'Progressive Passes': [], 'Progressive Carries': [], 'LineBreaking Pass': []}
                for name in unique_players:
                    progressor_counts['Progressive Passes'].append(len(df[(df['name'] == name) & (df['prog_pass'] >= 9.144) & (df['x']>=35) & (df['outcomeType']=='Successful') & (~df['qualifiers'].str.contains('CornerTaken|Freekick'))]))
                    progressor_counts['Progressive Carries'].append(len(df[(df['name'] == name) & (df['prog_carry'] >= 9.144) & (df['endX']>=35)]))
                    progressor_counts['LineBreaking Pass'].append(len(df[(df['name'] == name) & (df['type'] == 'Pass') & (df['qualifiers'].str.contains('Throughball'))]))
                progressor_df = pd.DataFrame(progressor_counts)
                progressor_df['total'] = progressor_df['Progressive Passes']+progressor_df['Progressive Carries']+progressor_df['LineBreaking Pass']
                progressor_df = progressor_df.sort_values(by='total', ascending=False)
                progressor_df.reset_index(drop=True, inplace=True)
                progressor_df = progressor_df.head(10)
                progressor_df['shortName'] = progressor_df['name'].apply(get_short_name)




                # Top Threate Creators
                # Initialize an empty dictionary to store home players different type of Carries counts
                xT_counts = {'name': unique_players, 'xT from Pass': [], 'xT from Carry': []}
                for name in unique_players:
                    xT_counts['xT from Pass'].append((df[(df['name'] == name) & (df['type'] == 'Pass') & (df['xT']>=0) & (df['outcomeType']=='Successful') & (~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn'))])['xT'].sum().round(2))
                    xT_counts['xT from Carry'].append((df[(df['name'] == name) & (df['type'] == 'Carry') & (df['xT']>=0)])['xT'].sum().round(2))
                xT_df = pd.DataFrame(xT_counts)
                xT_df['total'] = xT_df['xT from Pass']+xT_df['xT from Carry']
                xT_df = xT_df.sort_values(by='total', ascending=False)
                xT_df.reset_index(drop=True, inplace=True)
                xT_df = xT_df.head(10)
                xT_df['shortName'] = xT_df['name'].apply(get_short_name)




                # Shot Sequence Involvement
                df_no_carry = df[df['type']!='Carry']
                # Initialize an empty dictionary to store home players different type of shot sequence counts
                shot_seq_counts = {'name': unique_players, 'Shots': [], 'Shot Assist': [], 'Buildup to shot': []}
                # Putting counts in those lists
                for name in unique_players:
                    shot_seq_counts['Shots'].append(len(df[(df['name'] == name) & ((df['type']=='MissedShots') | (df['type']=='SavedShot') | (df['type']=='ShotOnPost') | (df['type']=='Goal'))]))
                    shot_seq_counts['Shot Assist'].append(len(df[(df['name'] == name) & (df['type'] == 'Pass') & (df['qualifiers'].str.contains('KeyPass'))]))
                    shot_seq_counts['Buildup to shot'].append(len(df_no_carry[(df_no_carry['name'] == name) & (df_no_carry['type'] == 'Pass') & (df_no_carry['qualifiers'].shift(-1).str.contains('KeyPass'))]))
                # converting that list into a dataframe
                sh_sq_df = pd.DataFrame(shot_seq_counts)
                sh_sq_df['total'] = sh_sq_df['Shots']+sh_sq_df['Shot Assist']+sh_sq_df['Buildup to shot']
                sh_sq_df = sh_sq_df.sort_values(by='total', ascending=False)
                sh_sq_df.reset_index(drop=True, inplace=True)
                sh_sq_df = sh_sq_df.head(10)
                sh_sq_df['shortName'] = sh_sq_df['name'].apply(get_short_name)




                # Top Defenders
                # Initialize an empty dictionary to store home players different type of defensive actions counts
                defensive_actions_counts = {'name': unique_players, 'Tackles': [], 'Interceptions': [], 'Clearance': []}
                for name in unique_players:
                    defensive_actions_counts['Tackles'].append(len(df[(df['name'] == name) & (df['type'] == 'Tackle') & (df['outcomeType']=='Successful')]))
                    defensive_actions_counts['Interceptions'].append(len(df[(df['name'] == name) & (df['type'] == 'Interception')]))
                    defensive_actions_counts['Clearance'].append(len(df[(df['name'] == name) & (df['type'] == 'Clearance')]))
                defender_df = pd.DataFrame(defensive_actions_counts)
                defender_df['total'] = defender_df['Tackles']+defender_df['Interceptions']+defender_df['Clearance']
                defender_df = defender_df.sort_values(by='total', ascending=False)
                defender_df.reset_index(drop=True, inplace=True)
                defender_df = defender_df.head(10)
                defender_df['shortName'] = defender_df['name'].apply(get_short_name)
                def home_player_passmap(ax):
                    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    ax.set_ylim(-0.5, 68.5)

                    # taking the top home passer and plotting his passmap
                    home_player_name = home_progressor_df['name'].iloc[0]

                    acc_pass = df[(df['name']==home_player_name) & (df['type']=='Pass') & (df['outcomeType']=='Successful')]
                    pro_pass = acc_pass[(acc_pass['prog_pass']>=9.11) & (acc_pass['x']>=35) & (~acc_pass['qualifiers'].str.contains('CornerTaken|Freekick'))]
                    pro_carry = df[(df['name']==home_player_name) & (df['prog_carry']>=9.11) & (df['endX']>=35)]
                    key_pass = acc_pass[acc_pass['qualifiers'].str.contains('KeyPass')]
                    g_assist = acc_pass[acc_pass['qualifiers'].str.contains('GoalAssist')]

                    pitch.lines(acc_pass.x, acc_pass.y, acc_pass.endX, acc_pass.endY, color=line_color, lw=2, alpha=0.15, comet=True, zorder=2, ax=ax)
                    pitch.lines(pro_pass.x, pro_pass.y, pro_pass.endX, pro_pass.endY, color=hcol      , lw=3, alpha=1,    comet=True, zorder=3, ax=ax)
                    pitch.lines(key_pass.x, key_pass.y, key_pass.endX, key_pass.endY, color=violet,     lw=4, alpha=1,    comet=True, zorder=4, ax=ax)
                    pitch.lines(g_assist.x, g_assist.y, g_assist.endX, g_assist.endY, color='green',      lw=4, alpha=1,    comet=True, zorder=5, ax=ax)

                    ax.scatter(acc_pass.endX, acc_pass.endY, s=30, color=bg_color,    edgecolor='gray', alpha=1, zorder=2)
                    ax.scatter(pro_pass.endX, pro_pass.endY, s=40, color=bg_color,  edgecolor= hcol,  alpha=1, zorder=3)
                    ax.scatter(key_pass.endX, key_pass.endY, s=50, color=bg_color,  edgecolor=violet, alpha=1, zorder=4)
                    ax.scatter(g_assist.endX, g_assist.endY, s=50, color=bg_color,  edgecolor= 'green', alpha=1, zorder=5)

                    for index, row in pro_carry.iterrows():
                        arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', color=hcol, zorder=4, mutation_scale=20, 
                                                        alpha=0.9, linewidth=2, linestyle='--')
                        ax.add_patch(arrow) 


                    home_name_show = home_progressor_df['shortName'].iloc[0]
                    ax.set_title(f"{home_name_show} PassMap", color=hcol, fontsize=25, fontweight='bold', y=1.03)
                    ax.text(0,-3, f'Prog. Pass: {len(pro_pass)}          Prog. Carry: {len(pro_carry)}', fontsize=15, color=hcol, ha='left', va='center')
                    ax_text(105,-3, s=f'Key Pass: {len(key_pass)}          <Assist: {len(g_assist)}>', fontsize=15, color=violet, ha='right', va='center',
                            highlight_textprops=[{'color':'green'}], ax=ax)

                def away_player_passmap(ax):
                    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    ax.set_ylim(-0.5, 68.5)
                    ax.invert_xaxis()
                    ax.invert_yaxis()

                    # taking the top away passer and plotting his passmap
                    away_player_name = away_progressor_df['name'].iloc[0]
                    
                    acc_pass = df[(df['name']==away_player_name) & (df['type']=='Pass') & (df['outcomeType']=='Successful')]
                    pro_pass = acc_pass[(acc_pass['prog_pass']>=9.11) & (acc_pass['x']>=35) & (~acc_pass['qualifiers'].str.contains('CornerTaken|Freekick'))]
                    pro_carry = df[(df['name']==away_player_name) & (df['prog_carry']>=9.11) & (df['endX']>=35)]
                    key_pass = acc_pass[acc_pass['qualifiers'].str.contains('KeyPass')]
                    g_assist = acc_pass[acc_pass['qualifiers'].str.contains('GoalAssist')]

                    pitch.lines(acc_pass.x, acc_pass.y, acc_pass.endX, acc_pass.endY, color=line_color, lw=2, alpha=0.15, comet=True, zorder=2, ax=ax)
                    pitch.lines(pro_pass.x, pro_pass.y, pro_pass.endX, pro_pass.endY, color=acol      , lw=3, alpha=1,    comet=True, zorder=3, ax=ax)
                    pitch.lines(key_pass.x, key_pass.y, key_pass.endX, key_pass.endY, color=violet,     lw=4, alpha=1,    comet=True, zorder=4, ax=ax)
                    pitch.lines(g_assist.x, g_assist.y, g_assist.endX, g_assist.endY, color='green',      lw=4, alpha=1,    comet=True, zorder=5, ax=ax)

                    ax.scatter(acc_pass.endX, acc_pass.endY, s=30, color=bg_color,    edgecolor='gray', alpha=1, zorder=2)
                    ax.scatter(pro_pass.endX, pro_pass.endY, s=40, color=bg_color,  edgecolor= acol,  alpha=1, zorder=3)
                    ax.scatter(key_pass.endX, key_pass.endY, s=50, color=bg_color,  edgecolor=violet, alpha=1, zorder=4)
                    ax.scatter(g_assist.endX, g_assist.endY, s=50, color=bg_color,  edgecolor= 'green', alpha=1, zorder=5)

                    for index, row in pro_carry.iterrows():
                        arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', color=acol, zorder=4, mutation_scale=20, 
                                                        alpha=0.9, linewidth=2, linestyle='--')
                        ax.add_patch(arrow) 


                    away_name_show = away_progressor_df['shortName'].iloc[0]
                    ax.set_title(f"{away_name_show} PassMap", color=acol, fontsize=25, fontweight='bold', y=1.03)
                    ax.text(0,71, f'Prog. Pass: {len(pro_pass)}          Prog. Carry: {len(pro_carry)}', fontsize=15, color=acol, ha='right', va='center')
                    ax_text(105,71, s=f'Key Pass: {len(key_pass)}          <Assist: {len(g_assist)}>', fontsize=15, color=violet, ha='left', va='center',
                            highlight_textprops=[{'color':'green'}], ax=ax)
                st.markdown("<h1 style='text-align: center;'>Top Players of the Match</h1>", unsafe_allow_html=True)
                st.header("Home Player Pass Map")

                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)

                # Generate the Home Player Pass Map
                home_player_passmap(ax)

                # Display the visualization in Streamlit
                st.pyplot(fig)

                # Add a header for context
                st.header("Away Player Pass Map")

                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)

                # Generate the Home Player Pass Map
                away_player_passmap(ax)
                # Display the visualization in Streamlit
                st.pyplot(fig)

                # Add a header for context
                def get_short_name(full_name):
                    
                    if not isinstance(full_name, str):
                        return None  # Return None if the input is not a string
                    parts = full_name.split()
                    if len(parts) == 1:
                        return full_name  # No need for short name if there's only one word
                    elif len(parts) == 2:
                        return parts[0][0] + ". " + parts[1]
                    else:
                        return parts[0][0] + ". " + parts[1][0] + ". " + " ".join(parts[2:])

                def home_passes_recieved(ax):
                    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    ax.set_ylim(-0.5, 68.5)

                    # plotting the home center forward pass receiving
                    name = home_average_locs_and_count_df.loc[home_average_locs_and_count_df['position'] == 'FW', 'name'].tolist()[0]
                    name_show = get_short_name(name)
                    filtered_rows = df[(df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (df['name'].shift(-1) == name)]
                    keypass_recieved_df = filtered_rows[filtered_rows['qualifiers'].str.contains('KeyPass')]
                    assist_recieved_df = filtered_rows[filtered_rows['qualifiers'].str.contains('IntentionalGoalAssist')]
                    pr = len(filtered_rows)
                    kpr = len(keypass_recieved_df)
                    ar = len(assist_recieved_df)

                    lc1 = pitch.lines(filtered_rows.x, filtered_rows.y, filtered_rows.endX, filtered_rows.endY, lw=3, transparent=True, comet=True,color=hcol, ax=ax, alpha=0.5)
                    lc2 = pitch.lines(keypass_recieved_df.x, keypass_recieved_df.y, keypass_recieved_df.endX, keypass_recieved_df.endY, lw=4, transparent=True, comet=True,color=violet, ax=ax, alpha=0.75)
                    lc3 = pitch.lines(assist_recieved_df.x, assist_recieved_df.y, assist_recieved_df.endX, assist_recieved_df.endY, lw=4, transparent=True, comet=True,color='green', ax=ax, alpha=0.75)
                    sc1 = pitch.scatter(filtered_rows.endX, filtered_rows.endY, s=30, edgecolor=hcol, linewidth=1, color=bg_color, zorder=2, ax=ax)
                    sc2 = pitch.scatter(keypass_recieved_df.endX, keypass_recieved_df.endY, s=40, edgecolor=violet, linewidth=1.5, color=bg_color, zorder=2, ax=ax)
                    sc3 = pitch.scatter(assist_recieved_df.endX, assist_recieved_df.endY, s=50, edgecolors='green', linewidths=1, marker='football', c=bg_color, zorder=2, ax=ax)

                    avg_endY = filtered_rows['endY'].median()
                    avg_endX = filtered_rows['endX'].median()
                    ax.axvline(x=avg_endX, ymin=0, ymax=68, color='gray', linestyle='--', alpha=0.6, linewidth=2)
                    ax.axhline(y=avg_endY, xmin=0, xmax=105, color='gray', linestyle='--', alpha=0.6, linewidth=2)
                    ax.set_title(f"{name_show} Passes Recieved", color=hcol, fontsize=25, fontweight='bold', y=1.03)
                    highlight_text=[{'color':violet}, {'color':'green'}]
                    ax_text(52.5,-3, f'Passes Recieved:{pr+kpr} | <Keypasses Recieved:{kpr}> | <Assist Received: {ar}>', color=line_color, fontsize=15, ha='center', 
                            va='center', highlight_textprops=highlight_text, ax=ax)

                    return

                def away_passes_recieved(ax):
                    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    ax.set_ylim(-0.5, 68.5)
                    ax.invert_xaxis()
                    ax.invert_yaxis()

                    # plotting the away center forward pass receiving
                    name = away_average_locs_and_count_df.loc[away_average_locs_and_count_df['position'] == 'FW', 'name'].tolist()[0]
                    name_show = get_short_name(name)
                    filtered_rows = df[(df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (df['name'].shift(-1) == name)]
                    keypass_recieved_df = filtered_rows[filtered_rows['qualifiers'].str.contains('KeyPass')]
                    assist_recieved_df = filtered_rows[filtered_rows['qualifiers'].str.contains('IntentionalGoalAssist')]
                    pr = len(filtered_rows)
                    kpr = len(keypass_recieved_df)
                    ar = len(assist_recieved_df)

                    lc1 = pitch.lines(filtered_rows.x, filtered_rows.y, filtered_rows.endX, filtered_rows.endY, lw=3, transparent=True, comet=True,color=acol, ax=ax, alpha=0.5)
                    lc2 = pitch.lines(keypass_recieved_df.x, keypass_recieved_df.y, keypass_recieved_df.endX, keypass_recieved_df.endY, lw=4, transparent=True, comet=True,color=violet, ax=ax, alpha=0.75)
                    lc3 = pitch.lines(assist_recieved_df.x, assist_recieved_df.y, assist_recieved_df.endX, assist_recieved_df.endY, lw=4, transparent=True, comet=True,color='green', ax=ax, alpha=0.75)
                    sc1 = pitch.scatter(filtered_rows.endX, filtered_rows.endY, s=30, edgecolor=acol, linewidth=1, color=bg_color, zorder=2, ax=ax)
                    sc2 = pitch.scatter(keypass_recieved_df.endX, keypass_recieved_df.endY, s=40, edgecolor=violet, linewidth=1.5, color=bg_color, zorder=2, ax=ax)
                    sc3 = pitch.scatter(assist_recieved_df.endX, assist_recieved_df.endY, s=50, edgecolors='green', linewidths=1, marker='football', c=bg_color, zorder=2, ax=ax)

                    avg_endX = filtered_rows['endX'].median()
                    avg_endY = filtered_rows['endY'].median()
                    ax.axvline(x=avg_endX, ymin=0, ymax=68, color='gray', linestyle='--', alpha=0.6, linewidth=2)
                    ax.axhline(y=avg_endY, xmin=0, xmax=105, color='gray', linestyle='--', alpha=0.6, linewidth=2)
                    ax.set_title(f"{name_show} Passes Recieved", color=acol, fontsize=25, fontweight='bold', y=1.03)
                    highlight_text=[{'color':violet}, {'color':'green'}]
                    ax_text(52.5,71, f'Passes Recieved:{pr+kpr} | <Keypasses Recieved:{kpr}> | <Assist Received: {ar}>', color=line_color, fontsize=15, ha='center', 
                            va='center', highlight_textprops=highlight_text, ax=ax)

                    return
                st.header("Home Passes Received")
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
                home_passes_recieved(ax)

                st.pyplot(fig)
                st.header("Away Passes Received")
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
                away_passes_recieved(ax)

                st.pyplot(fig)
                def home_player_def_acts(ax):
                    
                    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=2, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    ax.set_ylim(-12,68.5)

                    # taking the top home defender and plotting his defensive actions
                    home_player_name = home_defender_df['name'].iloc[0]
                    home_playerdf = df[(df['name']==home_player_name)]

                    hp_tk = home_playerdf[home_playerdf['type']=='Tackle']
                    hp_intc = home_playerdf[(home_playerdf['type']=='Interception') | (home_playerdf['type']=='BlockedPass')]
                    hp_br = home_playerdf[home_playerdf['type']=='BallRecovery']
                    hp_cl = home_playerdf[home_playerdf['type']=='Clearance']
                    hp_fl = home_playerdf[home_playerdf['type']=='Foul']
                    hp_ar = home_playerdf[(home_playerdf['type']=='Aerial') & (home_playerdf['qualifiers'].str.contains('Defensive'))]

                    sc1 = pitch.scatter(hp_tk.x, hp_tk.y, s=250, c=hcol, lw=2.5, edgecolor=hcol, marker='+', hatch='/////', ax=ax)
                    sc2 = pitch.scatter(hp_intc.x, hp_intc.y, s=250, c='None', lw=2.5, edgecolor=hcol, marker='s', hatch='/////', ax=ax)
                    sc3 = pitch.scatter(hp_br.x, hp_br.y, s=250, c='None', lw=2.5, edgecolor=hcol, marker='o', hatch='/////', ax=ax)
                    sc4 = pitch.scatter(hp_cl.x, hp_cl.y, s=250, c='None', lw=2.5, edgecolor=hcol, marker='d', hatch='/////', ax=ax)
                    sc5 = pitch.scatter(hp_fl.x, hp_fl.y, s=250, c=hcol, lw=2.5, edgecolor=hcol, marker='x', hatch='/////', ax=ax)
                    sc6 = pitch.scatter(hp_ar.x, hp_ar.y, s=250, c='None', lw=2.5, edgecolor=hcol, marker='^', hatch='/////', ax=ax)

                    sc7 =  pitch.scatter(2, -4, s=150, c=hcol, lw=2.5, edgecolor=hcol, marker='+', hatch='/////', ax=ax)
                    sc8 =  pitch.scatter(2, -10, s=150, c='None', lw=2.5, edgecolor=hcol, marker='s', hatch='/////', ax=ax)
                    sc9 =  pitch.scatter(41, -4, s=150, c='None', lw=2.5, edgecolor=hcol, marker='o', hatch='/////', ax=ax)
                    sc10 = pitch.scatter(41, -10, s=150, c='None', lw=2.5, edgecolor=hcol, marker='d', hatch='/////', ax=ax)
                    sc11 = pitch.scatter(103, -4, s=150, c=hcol, lw=2.5, edgecolor=hcol, marker='x', hatch='/////', ax=ax)
                    sc12 = pitch.scatter(103, -10, s=150, c='None', lw=2.5, edgecolor=hcol, marker='^', hatch='/////', ax=ax)

                    ax.text(5, -3, f"Tackle: {len(hp_tk)}\n\nInterception: {len(hp_intc)}", color=hcol, ha='left', va='top', fontsize=13)
                    ax.text(43, -3, f"BallRecovery: {len(hp_br)}\n\nClearance: {len(hp_cl)}", color=hcol, ha='left', va='top', fontsize=13)
                    ax.text(100, -3, f"{len(hp_fl)} Fouls\n\n{len(hp_ar)} Aerials", color=hcol, ha='right', va='top', fontsize=13)

                    home_name_show = home_defender_df['shortName'].iloc[0]
                    ax.set_title(f"{home_name_show} Defensive Actions", color=hcol, fontsize=25, fontweight='bold')

                def away_player_def_acts(ax):
                    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=2, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    ax.set_ylim(-0.5,80)
                    ax.invert_xaxis()
                    ax.invert_yaxis()

                    # taking the top home defender and plotting his defensive actions
                    away_player_name = away_defender_df['name'].iloc[0]
                    away_playerdf = df[(df['name']==away_player_name)]

                    ap_tk = away_playerdf[away_playerdf['type']=='Tackle']
                    ap_intc = away_playerdf[(away_playerdf['type']=='Interception') | (away_playerdf['type']=='BlockedPass')]
                    ap_br = away_playerdf[away_playerdf['type']=='BallRecovery']
                    ap_cl = away_playerdf[away_playerdf['type']=='Clearance']
                    ap_fl = away_playerdf[away_playerdf['type']=='Foul']
                    ap_ar = away_playerdf[(away_playerdf['type']=='Aerial') & (away_playerdf['qualifiers'].str.contains('Defensive'))]

                    sc1 = pitch.scatter(ap_tk.x, ap_tk.y, s=250, c=acol, lw=2.5, edgecolor=acol, marker='+', hatch='/////', ax=ax)
                    sc2 = pitch.scatter(ap_intc.x, ap_intc.y, s=250, c='None', lw=2.5, edgecolor=acol, marker='s', hatch='/////', ax=ax)
                    sc3 = pitch.scatter(ap_br.x, ap_br.y, s=250, c='None', lw=2.5, edgecolor=acol, marker='o', hatch='/////', ax=ax)
                    sc4 = pitch.scatter(ap_cl.x, ap_cl.y, s=250, c='None', lw=2.5, edgecolor=acol, marker='d', hatch='/////', ax=ax)
                    sc5 = pitch.scatter(ap_fl.x, ap_fl.y, s=250, c=acol, lw=2.5, edgecolor=acol, marker='x', hatch='/////', ax=ax)
                    sc6 = pitch.scatter(ap_ar.x, ap_ar.y, s=250, c='None', lw=2.5, edgecolor=acol, marker='^', hatch='/////', ax=ax)

                    sc7 =  pitch.scatter(2, 72, s=150, c=acol, lw=2.5, edgecolor=acol, marker='+', hatch='/////', ax=ax)
                    sc8 =  pitch.scatter(2, 78, s=150, c='None', lw=2.5, edgecolor=acol, marker='s', hatch='/////', ax=ax)
                    sc9 =  pitch.scatter(41, 72, s=150, c='None', lw=2.5, edgecolor=acol, marker='o', hatch='/////', ax=ax)
                    sc10 = pitch.scatter(41, 78, s=150, c='None', lw=2.5, edgecolor=acol, marker='d', hatch='/////', ax=ax)
                    sc11 = pitch.scatter(103, 72, s=150, c=acol, lw=2.5, edgecolor=acol, marker='x', hatch='/////', ax=ax)
                    sc12 = pitch.scatter(103, 78, s=150, c='None', lw=2.5, edgecolor=acol, marker='^', hatch='/////', ax=ax)

                    ax.text(5, 71, f"Tackle: {len(ap_tk)}\n\nInterception: {len(ap_intc)}", color=acol, ha='right', va='top', fontsize=13)
                    ax.text(43, 71, f"BallRecovery: {len(ap_br)}\n\nClearance: {len(ap_cl)}", color=acol, ha='right', va='top', fontsize=13)
                    ax.text(100, 71, f"{len(ap_fl)} Fouls\n\n{len(ap_ar)} Aerials", color=acol, ha='left', va='top', fontsize=13)

                    away_name_show = away_defender_df['shortName'].iloc[0]
                    ax.set_title(f"{away_name_show} Defensive Actions", color=acol, fontsize=25, fontweight='bold')

                st.header("Home Player Defensive Actions")
                # Create a subplot for the Away Player Defensive Actions visualization
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
                # Generate the Away Player Defensive Actions plot
                home_player_def_acts(ax)
                # Display the visualization in Streamlit
                st.pyplot(fig)
                st.header("Away Player Defensive Actions")
                # Create a subplot for the Away Player Defensive Actions visualization
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
                # Generate the Away Player Defensive Actions plot
                away_player_def_acts(ax)
                # Display the visualization in Streamlit
                st.pyplot(fig)
                def home_gk(ax):
                    df_gk = df[(df['teamName']==hteamName) & (df['position']=='GK')]
                    gk_pass = df_gk[df_gk['type']=='Pass']
                    op_pass = df_gk[(df_gk['type']=='Pass') & (~df_gk['qualifiers'].str.contains('GoalKick|FreekickTaken'))]
                    sp_pass = df_gk[(df_gk['type']=='Pass') &  (df_gk['qualifiers'].str.contains('GoalKick|FreekickTaken'))]
                    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    ax.set_ylim(-0.5, 68.5)
                    gk_name = df_gk['shortName'].unique()[0]
                    op_succ = sp_succ = 0
                    for index, row in op_pass.iterrows():
                        if row['outcomeType']=='Successful':
                            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=hcol, lw=4, comet=True, alpha=0.5, zorder=2, ax=ax)
                            ax.scatter(row['endX'], row['endY'], s=40, color=hcol, edgecolor=line_color, zorder=3)
                            op_succ += 1
                        if row['outcomeType']=='Unsuccessful':
                            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=hcol, lw=4, comet=True, alpha=0.5, zorder=2, ax=ax)
                            ax.scatter(row['endX'], row['endY'], s=40, color=bg_color, edgecolor=hcol, zorder=3)
                    for index, row in sp_pass.iterrows():
                        if row['outcomeType']=='Successful':
                            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=violet, lw=4, comet=True, alpha=0.5, zorder=2, ax=ax)
                            ax.scatter(row['endX'], row['endY'], s=40, color=violet, edgecolor=line_color, zorder=3)
                            sp_succ += 1
                        if row['outcomeType']=='Unsuccessful':
                            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=violet, lw=4, comet=True, alpha=0.35, zorder=2, ax=ax)
                            ax.scatter(row['endX'], row['endY'], s=40, color=bg_color, edgecolor=violet, zorder=3)

                    gk_pass['length'] = np.sqrt((gk_pass['x']-gk_pass['endX'])**2 + (gk_pass['y']-gk_pass['endY'])**2)
                    sp_pass['length'] = np.sqrt((sp_pass['x']-sp_pass['endX'])**2 + (sp_pass['y']-sp_pass['endY'])**2)
                    avg_len = gk_pass['length'].mean().round(2)
                    avg_hgh = sp_pass['endX'].mean().round(2)
                    
                    ax.set_title(f'{gk_name} PassMap', color=hcol, fontsize=25, fontweight='bold', y=1.07)
                    ax.text(52.5, -3, f'Avg. Passing Length: {avg_len}m     |     Avg. Goalkick Length: {avg_hgh}m', color=line_color, fontsize=15, ha='center', va='center')
                    ax_text(52.5, 70, s=f'<Open-play Pass (Acc.): {len(op_pass)} ({op_succ})>     |     <GoalKick/Freekick (Acc.): {len(sp_pass)} ({sp_succ})>', 
                            fontsize=15, highlight_textprops=[{'color':hcol}, {'color':violet}], ha='center', va='center', ax=ax)

                    return

                def away_gk(ax):
                    df_gk = df[(df['teamName']==ateamName) & (df['position']=='GK')]
                    gk_pass = df_gk[df_gk['type']=='Pass']
                    op_pass = df_gk[(df_gk['type']=='Pass') & (~df_gk['qualifiers'].str.contains('GoalKick|FreekickTaken'))]
                    sp_pass = df_gk[(df_gk['type']=='Pass') &  (df_gk['qualifiers'].str.contains('GoalKick|FreekickTaken'))]
                    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    ax.set_ylim(-0.5, 68.5)
                    ax.invert_xaxis()
                    ax.invert_yaxis()
                    gk_name = df_gk['shortName'].unique()[0]
                    op_succ = sp_succ = 0
                    for index, row in op_pass.iterrows():
                        if row['outcomeType']=='Successful':
                            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=acol, lw=4, comet=True, alpha=0.5, zorder=2, ax=ax)
                            ax.scatter(row['endX'], row['endY'], s=40, color=acol, edgecolor=line_color, zorder=3)
                            op_succ += 1
                        if row['outcomeType']=='Unsuccessful':
                            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=acol, lw=4, comet=True, alpha=0.5, zorder=2, ax=ax)
                            ax.scatter(row['endX'], row['endY'], s=40, color=bg_color, edgecolor=acol, zorder=3)
                    for index, row in sp_pass.iterrows():
                        if row['outcomeType']=='Successful':
                            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=violet, lw=4, comet=True, alpha=0.5, zorder=2, ax=ax)
                            ax.scatter(row['endX'], row['endY'], s=40, color=violet, edgecolor=line_color, zorder=3)
                            sp_succ += 1
                        if row['outcomeType']=='Unsuccessful':
                            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=violet, lw=4, comet=True, alpha=0.35, zorder=2, ax=ax)
                            ax.scatter(row['endX'], row['endY'], s=40, color=bg_color, edgecolor=violet, zorder=3)

                    gk_pass['length'] = np.sqrt((gk_pass['x']-gk_pass['endX'])**2 + (gk_pass['y']-gk_pass['endY'])**2)
                    sp_pass['length'] = np.sqrt((sp_pass['x']-sp_pass['endX'])**2 + (sp_pass['y']-sp_pass['endY'])**2)
                    avg_len = gk_pass['length'].mean().round(2)
                    avg_hgh = gk_pass['endX'].mean().round(2)

                    ax.set_title(f'{gk_name} PassMap', color=acol, fontsize=25, fontweight='bold', y=1.07)
                    ax.text(52.5, 71, f'Avg. Passing Length: {avg_len}m     |     Avg. Goalkick Length: {avg_hgh}m', color=line_color, fontsize=15, ha='center', va='center')
                    ax_text(52.5, -2, s=f'<Open-play Pass (Acc.): {len(op_pass)} ({op_succ})>     |     <GoalKick/Freekick (Acc.): {len(sp_pass)} ({sp_succ})>', 
                            fontsize=15, highlight_textprops=[{'color':acol}, {'color':violet}], ha='center', va='center', ax=ax)

                    return
                st.header("Home Goalkeeper Actions")
                # Create a subplot for the Away Player Defensive Actions visualization
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
                # Generate the Away Player Defensive Actions plot
                home_gk(ax)
                # Display the visualization in Streamlit
                st.pyplot(fig)
                st.header("Away Goalkeeper Actions")
                # Create a subplot for the Away Player Defensive Actions visualization
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
                # Generate the Away Player Defensive Actions plot
                away_gk(ax)
                # Display the visualization in Streamlit
                st.pyplot(fig)
                from matplotlib.ticker import MaxNLocator
                def sh_sq_bar(ax):
                    top10_sh_sq = sh_sq_df.nsmallest(10, 'total')['shortName'].tolist()

                    shsq_sh = sh_sq_df.nsmallest(10, 'total')['Shots'].tolist()
                    shsq_sa = sh_sq_df.nsmallest(10, 'total')['Shot Assist'].tolist()
                    shsq_bs = sh_sq_df.nsmallest(10, 'total')['Buildup to shot'].tolist()

                    left1 = [w + x for w, x in zip(shsq_sh, shsq_sa)]

                    ax.barh(top10_sh_sq, shsq_sh, label='Shot', color=col1, left=0)
                    ax.barh(top10_sh_sq, shsq_sa, label='Shot Assist', color=violet, left=shsq_sh)
                    ax.barh(top10_sh_sq, shsq_bs, label='Buildup to Shot', color=col2, left=left1)

                # Add counts in the middle of the bars (if count > 0)
                    for i, player in enumerate(top10_sh_sq):
                        for j, count in enumerate([shsq_sh[i], shsq_sa[i], shsq_bs[i]]):
                            if count > 0:
                                x_position = sum([shsq_sh[i], shsq_sa[i]][:j]) + count / 2
                                ax.text(x_position, i, str(count), ha='center', va='center', color=bg_color, fontsize=18, fontweight='bold')

                    max_x = sh_sq_df['total'].iloc()[0]
                    x_coord = [2 * i for i in range(1, int(max_x/2))]
                    for x in x_coord:
                        ax.axvline(x=x, color='gray', linestyle='--', zorder=2, alpha=0.5)

                    ax.set_facecolor(bg_color)
                    ax.tick_params(axis='x', colors=line_color, labelsize=15)
                    ax.tick_params(axis='y', colors=line_color, labelsize=15)
                    ax.xaxis.label.set_color(line_color)
                    ax.yaxis.label.set_color(line_color)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(bg_color)

                    ax.set_title(f"Shot Sequence Involvement", color=line_color, fontsize=25, fontweight='bold')
                    ax.legend(fontsize=13)
                    st.header("Shot Sequence Involvement")

                    # Create a subplot for the Shot Sequence Bar Chart
                    fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)

                    # Generate the Shot Sequence Bar Chart
                    sh_sq_bar(ax)

                    # Display the visualization in Streamlit
                    st.pyplot(fig)
                def passer_bar(ax):
                    top10_passers = progressor_df.nsmallest(10, 'total')['shortName'].tolist()

                    passers_pp = progressor_df.nsmallest(10, 'total')['Progressive Passes'].tolist()
                    passers_tp = progressor_df.nsmallest(10, 'total')['Progressive Carries'].tolist()
                    passers_kp = progressor_df.nsmallest(10, 'total')['LineBreaking Pass'].tolist()

                    left1 = [w + x for w, x in zip(passers_pp, passers_tp)]

                    ax.barh(top10_passers, passers_pp, label='Prog. Pass', color=col1, left=0)
                    ax.barh(top10_passers, passers_tp, label='Prog. Carries', color=col2, left=passers_pp)
                    ax.barh(top10_passers, passers_kp, label='Through Pass', color=violet, left=left1)

                    # Add counts in the middle of the bars (if count > 0)
                    for i, player in enumerate(top10_passers):
                        for j, count in enumerate([passers_pp[i], passers_tp[i], passers_kp[i]]):
                            if count > 0:
                                x_position = sum([passers_pp[i], passers_tp[i]][:j]) + count / 2
                                ax.text(x_position, i, str(count), ha='center', va='center', color=bg_color, fontsize=18, fontweight='bold')

                    max_x = progressor_df['total'].iloc()[0]
                    x_coord = [2 * i for i in range(1, int(max_x/2))]
                    for x in x_coord:
                        ax.axvline(x=x, color='gray', linestyle='--', zorder=2, alpha=0.5)

                    ax.set_facecolor(bg_color)
                    ax.tick_params(axis='x', colors=line_color, labelsize=15)
                    ax.tick_params(axis='y', colors=line_color, labelsize=15)
                    ax.xaxis.label.set_color(line_color)
                    ax.yaxis.label.set_color(line_color)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(bg_color)

                    ax.set_title(f"Top10 Ball Progressors", color=line_color, fontsize=25, fontweight='bold')
                    ax.legend(fontsize=13)

                st.header("Top 10 Ball Progressors")

                    # Create a subplot for the Ball Progressors Bar Chart
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)

                    # Generate the Ball Progressors Bar Chart
                passer_bar(ax)

                    # Display the visualization in Streamlit
                st.pyplot(fig)
                def defender_bar(ax):
                    top10_defenders = defender_df.nsmallest(10, 'total')['shortName'].tolist()

                    defender_tk = defender_df.nsmallest(10, 'total')['Tackles'].tolist()
                    defender_in = defender_df.nsmallest(10, 'total')['Interceptions'].tolist()
                    defender_ar = defender_df.nsmallest(10, 'total')['Clearance'].tolist()

                    left1 = [w + x for w, x in zip(defender_tk, defender_in)]

                    ax.barh(top10_defenders, defender_tk, label='Tackle', color=col1, left=0)
                    ax.barh(top10_defenders, defender_in, label='Interception', color=violet, left=defender_tk)
                    ax.barh(top10_defenders, defender_ar, label='Clearance', color=col2, left=left1)

                    # Add counts in the middle of the bars (if count > 0)
                    for i, player in enumerate(top10_defenders):
                        for j, count in enumerate([defender_tk[i], defender_in[i], defender_ar[i]]):
                            if count > 0:
                                x_position = sum([defender_tk[i], defender_in[i]][:j]) + count / 2
                                ax.text(x_position, i, str(count), ha='center', va='center', color=bg_color, fontsize=18, fontweight='bold')

                    max_x = defender_df['total'].iloc()[0]
                    x_coord = [2 * i for i in range(1, int(max_x/2))]
                    for x in x_coord:
                        ax.axvline(x=x, color='gray', linestyle='--', zorder=2, alpha=0.5)

                    ax.set_facecolor(bg_color)
                    ax.tick_params(axis='x', colors=line_color, labelsize=15)
                    ax.tick_params(axis='y', colors=line_color, labelsize=15)
                    ax.xaxis.label.set_color(line_color)
                    ax.yaxis.label.set_color(line_color)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(bg_color)


                    ax.set_title(f"Top10 Defenders", color=line_color, fontsize=25, fontweight='bold')
                    ax.legend(fontsize=13)
                st.header("Top 10 Defenders")

                # Create a subplot for the Defenders Bar Chart
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)

                # Generate the Defenders Bar Chart
                defender_bar(ax)

                # Display the visualization in Streamlit
                st.pyplot(fig)

                def threat_creators(ax):
                    top10_xT = xT_df.nsmallest(10, 'total')['shortName'].tolist()

                    xT_pass = xT_df.nsmallest(10, 'total')['xT from Pass'].tolist()
                    xT_carry = xT_df.nsmallest(10, 'total')['xT from Carry'].tolist()

                    left1 = [w + x for w, x in zip(xT_pass, xT_carry)]

                    ax.barh(top10_xT, xT_pass, label='xT from pass', color=col1, left=0)
                    ax.barh(top10_xT, xT_carry, label='xT from carry', color=violet, left=xT_pass)

                    # Add counts in the middle of the bars (if count > 0)
                    for i, player in enumerate(top10_xT):
                        for j, count in enumerate([xT_pass[i], xT_carry[i]]):
                            if count > 0:
                                x_position = sum([xT_pass[i], xT_carry[i]][:j]) + count / 2
                                ax.text(x_position, i, str(count), ha='center', va='center', color=line_color, fontsize=15, rotation=45)

                    # max_x = xT_df['total'].iloc()[0]
                    # x_coord = [2 * i for i in range(1, int(max_x/2))]
                    # for x in x_coord:
                    #     ax.axvline(x=x, color='gray', linestyle='--', zorder=2, alpha=0.5)

                    ax.set_facecolor(bg_color)
                    ax.tick_params(axis='x', colors=line_color, labelsize=15)
                    ax.tick_params(axis='y', colors=line_color, labelsize=15)
                    ax.xaxis.label.set_color(line_color)
                    ax.yaxis.label.set_color(line_color)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(bg_color)


                    ax.set_title(f"Top10 Threatening Players", color=line_color, fontsize=25, fontweight='bold')
                    ax.legend(fontsize=13)
                st.header("Top 10 Threatening Players")

                # Create a subplot for the Threat Creators Bar Chart
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)

                # Generate the Threat Creators Bar Chart
                threat_creators(ax)

                # Display the visualization in Streamlit
                st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please select a team, a match, and press 'Go' to generate the report.")

