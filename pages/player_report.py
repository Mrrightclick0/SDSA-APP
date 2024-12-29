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

# Print the modified DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
def app():
    st.title("Player Report")

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

    # Paths to data files
    event_data_file = os.path.join("data", "all_stat.csv")
    shot_data_file = os.path.join("data", "all_shot.csv")

    try:
        # Handle potential formatting issues in CSV files
        event_data = pd.read_csv(event_data_file, on_bad_lines='skip')
        shot_data = pd.read_csv(shot_data_file, on_bad_lines='skip')
        st.success("Data loaded successfully!")
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    except pd.errors.ParserError as e:
        st.error(f"Parsing error: {e}. Ensure the CSV files are properly formatted.")
        st.stop()

    # Validate required columns
    required_event_columns = ['teamName', 'name']
    required_shot_columns = ['teamName']
    if not all(col in event_data.columns for col in required_event_columns):
        st.error(f"Event data file is missing required columns: {required_event_columns}")
        st.stop()
    if not all(col in shot_data.columns for col in required_shot_columns):
        st.error(f"Shot data file is missing required columns: {required_shot_columns}")
        st.stop()

    # Extract unique teams
    teams = sorted(event_data['teamName'].unique())

    # Streamlit Layout
    st.title("Player Analysis")

    # Sidebar for Team and Player Selection
    with st.sidebar:
        st.header("Player Selection")
        
        # Team Selection
        selected_team = st.selectbox("Select a Team", [""] + teams)

    if selected_team:
        # Filter data for the selected team
        team_event_data = event_data[event_data['teamName'] == selected_team]
        team_shot_data = shot_data[shot_data['teamName'] == selected_team]
        
        # Extract players from the selected team
        team_event_data['name'] = team_event_data['name'].astype(str)  # Convert all values to strings
        players = sorted(team_event_data['name'].unique())  # Sort the unique player names
        # Player Selection
        selected_player = st.selectbox("Select a Player", [""] + players)

        if selected_player:
            # Filter data for the selected player
            player_event_data = team_event_data[team_event_data['name'] == selected_player]

            # Remove rows with missing names or team names
            df = player_event_data.dropna(subset=['name', 'teamName'])

            # Ensure required columns are present
            required_columns = ['teamName', 'oppositionTeamName']
            if not all(col in df.columns for col in required_columns):
                st.error(f"The event data file for {selected_team} does not contain the required columns: {required_columns}")
            else:
                # Extract home and away team names
                hteamName = selected_team 
                ateamNames = df[df['teamName'] != hteamName]['teamName'].unique().tolist()

    # Filter DataFrames for home and away teams
                hteam_df = df[df['teamName'] == hteamName]  # Data for the selected home team
                ateam_df = df[df['teamName'].isin(ateamNames)]

                # Extract player information
                columns_to_extract = ['playerId', 'shirtNo', 'name', 'position', 'isFirstEleven', 'teamName', 'oppositionTeamName']
                players_df = df[columns_to_extract]

                # Ensure shirt numbers are strings or 'nan'
                players_df['shirtNo'] = players_df['shirtNo'].apply(lambda x: str(int(x)) if pd.notna(x) else 'nan')

                # Filter players belonging to the home team (hteamName)
                players_df = players_df[players_df['teamName'] == hteamName]

                # Remove duplicates based on playerId
                players_df = players_df.drop_duplicates(subset='playerId')

                # Team-specific data
                homedf = df[df['teamName'] == hteamName]
                awaydf = df[df['teamName'] != hteamName]

                # Metrics
                hxT = homedf['xT'].sum().round(2)
                axT = awaydf['xT'].sum().round(2)

                hgoal_count = len(homedf[(homedf['teamName'] == hteamName) & (homedf['type'] == 'Goal') & (~homedf['qualifiers'].str.contains('OwnGoal'))])
                agoal_count = len(awaydf[(awaydf['teamName'] != hteamName) & (awaydf['type'] == 'Goal') & (~awaydf['qualifiers'].str.contains('OwnGoal'))])
                hgoal_count += len(awaydf[(awaydf['teamName'] != hteamName) & (awaydf['type'] == 'Goal') & (awaydf['qualifiers'].str.contains('OwnGoal'))])
                agoal_count += len(homedf[(homedf['teamName'] == hteamName) & (homedf['type'] == 'Goal') & (homedf['qualifiers'].str.contains('OwnGoal'))])

                # Filter shots for the selected home team (user-selected team)
                hshots_xgdf = team_shot_data[
                    (team_shot_data['fullName'] == selected_player)&(team_shot_data['oppositeTeam'] != hteamName)

                ]                # Filter shots for the away team (teams not selected by the user)S
                ashots_xgdf = team_shot_data[(team_shot_data['teamName'] == hteamName)]
                hxg = round(hshots_xgdf['expectedGoals'].sum(), 2)
                axg = round(ashots_xgdf['expectedGoals'].sum(), 2)
                hxgot = round(hshots_xgdf['expectedGoalsOnTarget'].sum(), 2)
                axgot = round(ashots_xgdf['expectedGoalsOnTarget'].sum(), 2)

                # Display Metrics
                # Display Player DataFrame
            
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

                home_passes_between_df, home_average_locs_and_count_df = get_passes_between_df(hteamName, passes_df, players_df)

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
                    


                    
                    ax.text(avgph-1, -5, f"{avgph_show}m", fontsize=15, color=line_color, ha='right')
                    ax.text(105, -5, f"verticality: {verticality}%", fontsize=15, color=line_color, ha='right')

                    # Headlines and other texts
                    if teamName == hteamName:
                        ax.text(2,66, "circle = starter\nbox = sub", color=hcol, size=12, ha='left', va='top')
                        ax.set_title(f"{selected_player}\nAverage Location", color=line_color, size=25, fontweight='bold')

                    else:
                        ax.text(2,2, "circle = starter\nbox = sub", color=acol, size=12, ha='right', va='top')

                    return {
                        'Team_Name': teamName,
                        'Defense_Line_Height': def_line_h,
                        'Vericality_%': verticality,
                        'Most_pass_combination_from': most_pass_from,
                        'Most_pass_combination_to': most_pass_to,
                        'Most_passes_in_combination': most_pass_count,
                    } 

                fig,axs=plt.subplots(figsize=(20,10), facecolor=bg_color)
                pass_network_stats_home = pass_network_visualization(axs, home_passes_between_df, home_average_locs_and_count_df, hcol, hteamName)
                pass_network_stats_list = [pass_network_stats_home]
                pass_network_stats_df = pd.DataFrame(pass_network_stats_list)

                
                    # Display the visualization
                st.header("Average Player Location")
                st.pyplot(fig)
                def get_defensive_action_df(events_dict):
                    # filter only defensive actions
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
                defensive_home_average_locs_and_count_df = defensive_home_average_locs_and_count_df[defensive_home_average_locs_and_count_df['position'] != 'GK']

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
                    
                
                    
                    ax.text(dah-1, -5, f"{dah_show}m", fontsize=15, color=line_color, ha='right', va='center')

                    # Headlines and other texts
                    if team_name == hteamName:
                        ax.text(105, -5, f'Compact:{compactness}%', fontsize=15, color=line_color, ha='right', va='center')
                        ax.text(2,66, "circle = starter\nbox = sub", color='gray', size=12, ha='left', va='top')
                        ax.set_title(f"{selected_player}\nDefensive Action Heatmap", color=line_color, fontsize=25, fontweight='bold')

                    return {
                        'Team_Name': team_name,
                        'Average_Defensive_Action_Height': dah,
                        'Forward_Line_Pressing_Height': fwd_line_h
                    }

                fig,axs=plt.subplots(figsize=(20,10), facecolor=bg_color)
                defensive_block_stats_home = defensive_block(axs, defensive_home_average_locs_and_count_df, hteamName, hcol)
                defensive_block_stats_home_list = [defensive_block_stats_home]
                defensive_block_stats_home_df = pd.DataFrame(defensive_block_stats_home_list)
                defensive_block_stats_list = []
                defensive_block_stats_list.append(defensive_block_stats_home)
                defensive_block_stats_df = pd.DataFrame(defensive_block_stats_list)
                    # Display the visualization

                st.header("Defensive Action")
                st.pyplot(fig)
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

                    pro_count = len(dfpro)

                    # calculating the counts
                    if pro_count > 0:
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
                    

                        # plotting the passes
                        pro_pass = pitch.lines(dfpro.x, dfpro.y, dfpro.endX, dfpro.endY, lw=3.5, comet=True, color=col, ax=ax, alpha=0.5)
                        # plotting some scatters at the end of each pass
                        pro_pass_end = pitch.scatter(dfpro.endX, dfpro.endY, s=35, edgecolor=col, linewidth=1, color=bg_color, zorder=2, ax=ax)

                        counttext = f"{pro_count} Progressive Passes"

                        # Heading and other texts
                        if col == hcol:
                            ax.set_title(f"{selected_player}\n{counttext}", color=line_color, fontsize=25, fontweight='bold')

                        return {
                            'Team_Name': team_name,
                            'Total_Progressive_Passes': pro_count,
                            'Progressive_Passes_From_Left': left_pro,
                            'Progressive_Passes_From_Center': mid_pro,
                            'Progressive_Passes_From_Right': right_pro
                        }

                    fig,axs=plt.subplots(figsize=(20,10), facecolor=bg_color)
                    Progressvie_Passes_Stats_home = draw_progressive_pass_map(axs, hteamName, hcol)
                    Progressvie_Passes_Stats_list = []
                    Progressvie_Passes_Stats_list.append(Progressvie_Passes_Stats_home)
                    Progressvie_Passes_Stats_df = pd.DataFrame(Progressvie_Passes_Stats_list)
                    # Display the visualization
                    st.header("Progressive Passes")
                    st.pyplot(fig)

                def draw_progressive_carry_map(ax, team_name, col):
                    dfpro = df[(df['teamName']==team_name) & (df['prog_carry']>=9.11) & (df['endX']>=35)]
                    # df_pen = df[(df['teamName']==team_name) & (df['type']=='Pass') & (df['endX']>=88.5) & (df['endY']>=13.6) & (df['endY']<=54.4) &
                    #             ~((df['x']>=88.5) & (df['y']>=13.6) & (df['y']<=54.6))]
                    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2,
                                        corner_arcs=True)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    # ax.set_ylim(-5, 68.5)

                    pro_count = len(dfpro)
                    if pro_count > 0:
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
                            ax.set_title(f"{selected_player}\n{counttext}", color=line_color, fontsize=25, fontweight='bold')

                        return {
                            'Team_Name': team_name,
                            'Total_Progressive_Carries': pro_count,
                            'Progressive_Carries_From_Left': left_pro,
                            'Progressive_Carries_From_Center': mid_pro,
                            'Progressive_Carries_From_Right': right_pro
                        }

                    fig,axs=plt.subplots(figsize=(20,10), facecolor=bg_color)
                    Progressvie_Carries_Stats_home = draw_progressive_carry_map(axs, hteamName, hcol)
                    Progressvie_Carries_Stats_list = []
                    Progressvie_Carries_Stats_list.append(Progressvie_Carries_Stats_home)
                    Progressvie_Carries_Stats_df = pd.DataFrame(Progressvie_Carries_Stats_list)
                    # Display the visualization
                    st.header("Progressive Carries")
                    st.pyplot(fig)
                # filtering the shots only
                mask4 = (df['type'] == 'Goal') | (df['type'] == 'MissedShots') | (df['type'] == 'SavedShot') | (df['type'] == 'ShotOnPost')
                Shotsdf = df[mask4]
                Shotsdf = Shotsdf.reset_index(drop=True)

                # filtering according to the types of shots
                hShotsdf = Shotsdf[Shotsdf['teamName']==hteamName]
                aShotsdf = Shotsdf[Shotsdf['oppositionTeamName']== hteamName]
                hSavedf = hShotsdf[(hShotsdf['type']=='SavedShot') & (~hShotsdf['qualifiers'].str.contains(': 82,'))]
                aSavedf = aShotsdf[(aShotsdf['type']=='SavedShot') & (~aShotsdf['qualifiers'].str.contains(': 82,'))]
                hogdf = hShotsdf[(hShotsdf['teamName']==hteamName) & (hShotsdf['qualifiers'].str.contains('OwnGoal'))]
                # Center Goal point
                given_point = (105, 34)
                # Calculate distances
                home_shot_distances = np.sqrt((hShotsdf['x'] - given_point[0])**2 + (hShotsdf['y'] - given_point[1])**2)
                home_average_shot_distance = round(home_shot_distances.mean(),2)
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
                    # sometimes the both teams ends the match 0-0, then normalizing the data becomes problem, thats why this part of the code
                    if hgoal_count+agoal_count == 0:
                        hgoal = 10
                        agoal = 10
                    else:
                        hgoal = (hgoal_count/(hgoal_count+agoal_count))*20
                        agoal = (agoal_count/(hgoal_count+agoal_count))*20
                        
                    if total_bigC_home == 0:
                        total_bigC_home_n = 10
                        total_bigC_away_n = 10
                    else:
                        total_bigC_home_n = (total_bigC_home/(total_bigC_home))*20
                        
                    if bigC_miss_home == 0:
                        bigC_miss_home_n = 10
                        bigC_miss_away_n = 10
                    else:
                        bigC_miss_home_n = (bigC_miss_home/(bigC_miss_home))*20

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
                        # normalizing the stats
                    shooting_stats_normalized_home = [(hgoal/(hgoal))*20, (hxg/(hxg))*20, (hxgot_n/(hxgot_n))*20,
                                                        (hTotalShots/(hTotalShots))*20, (hShotsOnT_n/(hShotsOnT_n))*20,
                                                        (total_bigC_home_n/(total_bigC_home_n))*20, (bigC_miss_home_n/(bigC_miss_home_n))*20,
                                                        (hxGpSh/(hxGpSh))*20, 
                                                        (home_average_shot_distance/(home_average_shot_distance))*20]
                        
                        # definig the start point
                    start_x = 42.5
                    start_x_for_away = [x + 42.5 for x in shooting_stats_normalized_home]
                    ax.barh(shooting_stats_title, shooting_stats_normalized_home, height=5, color=hcol, left=start_x)
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

                        # Heading and other texts
                    ax.text(0, 70, f"{selected_player}\n<---shots", color=hcol, size=25, ha='left', fontweight='bold')
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
                        
                        
                    return [home_data]

                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)

                    # Call the `plot_shotmap` function
                shooting_stats = plot_shotmap(ax)

                    # Convert the shooting stats to a DataFrame
                shooting_stats_df = pd.DataFrame(shooting_stats)

                    # Display the visualization in Streamlit
                st.header("Shooting Statistics")
                st.pyplot(fig)
                def plot_goalPost(ax):
                        hShotsdf = Shotsdf[Shotsdf['teamName']==hteamName]
                        aShotsdf = Shotsdf[Shotsdf['oppositionTeamName']== hteamName]
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
                        ax.text(52.5, -2, f"Aginst GK ", color=hcol, fontsize=30, ha='center', va='top', fontweight='bold')
                        ax.text(100, 2, f"Saves = {len(hSavedf)+len(hSavedf_bc)}\n\nxGOT faced:\n{hxgot}\n\nGoals Prevented:\n{round(hxgot - len(hGoaldf) - len(hGoaldf_bc),2)}",
                                        color=hcol, fontsize=16, va='bottom', ha='left')

                        home_data = {
                            'Team_Name': hteamName,
                            'Shots_Saved': len(hSavedf)+len(hSavedf_bc),
                            'Big_Chance_Saved': len(hSavedf_bc),
                            'Goals_Prevented': round(hxgot - len(hGoaldf) - len(hGoaldf_bc),2)
                        }
                        
                        away_data = {
                            'Team_Name'
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
                st.pyplot(fig)
                def Final_third_entry(ax, team_name, col):
                    dfpass = df[(df['teamName']==team_name) & (df['type']=='Pass') & (df['x']<70) & (df['endX']>=70) & (df['outcomeType']=='Successful') &
                                (~df['qualifiers'].str.contains('Freekick'))]
                    dfcarry = df[(df['teamName']==team_name) & (df['type']=='Carry') & (df['x']<70) & (df['endX']>=70)]
                    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2,
                                        corner_arcs=True)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    # ax.set_ylim(-0.5, 68.5)

                    pass_count = len(dfpass) + len(dfcarry)
                    if pass_count > 0:

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
                            ax.set_title(f"{selected_player}\n{counttext}", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
                            ax.text(87.5, 70, '<---------------- Final third ---------------->', color=line_color, ha='center', va='center')
                            pitch.lines(53, -2, 73, -2, lw=3, transparent=True, comet=True, color=col, ax=ax, alpha=0.5)
                            ax.scatter(73,-2, s=35, edgecolor=col, linewidth=1, color=bg_color, zorder=2)
                            arrow = patches.FancyArrowPatch((83, -2), (103, -2), arrowstyle='->', color=col, zorder=4, mutation_scale=20, 
                                                            alpha=1, linewidth=2, linestyle='--')
                            ax.add_patch(arrow)
                            ax.text(63, -5, f'Entry by Pass: {len(dfpass)}', fontsize=15, color=line_color, ha='center', va='center')
                            ax.text(93, -5, f'Entry by Carry: {len(dfcarry)}', fontsize=15, color=line_color, ha='center', va='center')
                        return {
                            'Team_Name': team_name,
                            'Total_Final_Third_Entries': pass_count,
                            'Final_Third_Entries_From_Left': left_entry,
                            'Final_Third_Entries_From_Center': mid_entry,
                            'Final_Third_Entries_From_Right': right_entry,
                            'Entry_By_Pass': len(dfpass),
                            'Entry_By_Carry': len(dfcarry)
                        }

                    fig,axs=plt.subplots(figsize=(20,10), facecolor=bg_color)
                    final_third_entry_stats_home = Final_third_entry(axs, hteamName, hcol)
                    final_third_entry_stats_list = []
                    final_third_entry_stats_list.append(final_third_entry_stats_home)
                    final_third_entry_stats_df = pd.DataFrame(final_third_entry_stats_list)
                    st.header("Final Third Entry")
                    st.pyplot(fig) 
                def zone14hs(ax, team_name, col):
                    dfhp = df[(df['teamName']==team_name) & (df['type']=='Pass') & (df['outcomeType']=='Successful') & 
                            (~df['qualifiers'].str.contains('CornerTaken|Freekick'))]
                    
                    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color,  linewidth=2,
                                        corner_arcs=True)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    ax.set_facecolor(bg_color)

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
                        ax.set_title(f"{selected_player}\nZone14 & Halfsp. Pass", color=line_color, fontsize=25, fontweight='bold')

                    return {
                        'Team_Name': team_name,
                        'Total_Passes_Into_Zone14': z14,
                        'Passes_Into_Halfspaces': hs,
                        'Passes_Into_Left_Halfspaces': lhs,
                        'Passes_Into_Right_Halfspaces': rhs
                    }

                fig,axs=plt.subplots(figsize=(20,10), facecolor=bg_color)
                zonal_passing_stats_home = zone14hs(axs, hteamName, hcol)
                zonal_passing_stats_list = []
                zonal_passing_stats_list.append(zonal_passing_stats_home)
                zonal_passing_stats_df = pd.DataFrame(zonal_passing_stats_list)
    # Display the visualizations in Streamlit
                st.header("Zonal Passing Statistics")
                st.pyplot(fig)
                # setting the custom colormap
                pearl_earring_cmaph = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",  [bg_color, hcol], N=20)
                pearl_earring_cmapa = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",  [bg_color, acol], N=20)

                path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]

                def Pass_end_zone(ax, team_name, cm):
                    pez = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful')]
                    pitch = Pitch(pitch_type='uefa', line_color=line_color, goal_type='box', goal_alpha=.5, corner_arcs=True, line_zorder=2, pitch_color=bg_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    pearl_earring_cmap = cm
                    # binning the data points
                    # bin_statistic = pitch.bin_statistic_positional(df.endX, df.endY, statistic='count', positional='full', normalize=True)
                    bin_statistic = pitch.bin_statistic(pez.endX, pez.endY, bins=(6, 5), normalize=True)
                    pitch.heatmap(bin_statistic, ax=ax, cmap=pearl_earring_cmap, edgecolors=bg_color)
                    pitch.scatter(pez.endX, pez.endY, c='gray', alpha=0.5, s=5, ax=ax)
                    labels = pitch.label_heatmap(bin_statistic, color=line_color, fontsize=25, ax=ax, ha='center', va='center', str_format='{:.0%}', path_effects=path_eff)

                    # Headings and other texts
                    if team_name == hteamName:
                        ax.set_title(f"{selected_player}\nPass End Zone", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)

                
                # setting the custom colormap
                pearl_earring_cmaph = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors", [bg_color, hcol], N=20)
                path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]
                fig,axs=plt.subplots(figsize=(20,10), facecolor=bg_color)
                Pass_end_zone(axs, hteamName, pearl_earring_cmaph)
                # Display the visualization in Streamlit
                st.header("Pass End Zone")
                st.pyplot(fig)

                def Chance_creating_zone(ax, team_name, cm, col):
                    ccp = df[(df['qualifiers'].str.contains('KeyPass')) & (df['teamName']==team_name)]
                    pitch = Pitch(pitch_type='uefa', line_color=line_color, corner_arcs=True, line_zorder=2, pitch_color=bg_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
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
                    if cc > 0: 
                        pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=violet, comet=True, lw=3, zorder=3, ax=ax)
                        ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor=violet, zorder=4)
                        cc += 1
                    labels = pitch.label_heatmap(bin_statistic, color=line_color, fontsize=25, ax=ax, ha='center', va='center', str_format='{:.0f}', path_effects=path_eff)
                    teamName = team_name

                    # Headings and other texts
                    if col == hcol:
                        ax.text(105,-3.5, "violet = key pass\ngreen = assist", color=hcol, size=15, ha='right', va='center')
                        ax.text(52.5,70, f"Total Chances Created = {cc}", color=col, fontsize=15, ha='center', va='center')
                        ax.set_title(f"{selected_player}\nChance Creating Zone", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
                    return {
                        'Team_Name': team_name,
                        'Total_Chances_Created': cc
                    }

                fig,axs=plt.subplots(figsize=(20,10), facecolor=bg_color)
                chance_creating_stats_home = Chance_creating_zone(axs, hteamName, pearl_earring_cmaph, hcol)
                chance_creating_stats_list = []
                chance_creating_stats_list.append(chance_creating_stats_home)
                chance_creating_stats_df = pd.DataFrame(chance_creating_stats_list)
                st.header("Chance Creating Zone Statistics")

                st.pyplot(fig)  
                def box_entry(ax):
                    bentry = df[((df['type']=='Pass')|(df['type']=='Carry')) & (df['outcomeType']=='Successful') & (df['endX']>=88.5) &
                                ~((df['x']>=88.5) & (df['y']>=13.6) & (df['y']<=54.6)) & (df['endY']>=13.6) & (df['endY']<=54.4) &
                            (~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn'))]
                    hbentry = bentry[bentry['teamName']==hteamName]

                    hrigt = hbentry[hbentry['y']<68/3]
                    hcent = hbentry[(hbentry['y']>=68/3) & (hbentry['y']<=136/3)]
                    hleft = hbentry[hbentry['y']>136/3]
                    pitch = Pitch(pitch_type='uefa', line_color=line_color, corner_arcs=True, line_zorder=2, pitch_color=bg_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_xlim(-0.5, 105.5)
                    ax.set_ylim(-0.5, 68.5)

                    for index, row in bentry.iterrows():
                        if row['teamName'] == hteamName:
                            color = hcol
                            x, y, endX, endY = 105 - row['x'], 68 - row['y'], 105 - row['endX'], 68 - row['endY']
                        else:
                            continue  # Skip rows that don't match either team name
                        if row['type'] == 'Pass':
                            pitch.lines(x, y, endX, endY, lw=3.5, comet=True, color=color, ax=ax, alpha=0.5)
                            pitch.scatter(endX, endY, s=35, edgecolor=color, linewidth=1, color=bg_color, zorder=2, ax=ax)
                        elif row['type'] == 'Carry':
                            arrow = patches.FancyArrowPatch((x, y), (endX, endY), arrowstyle='->', color=color, zorder=4, mutation_scale=20, 
                                                            alpha=1, linewidth=2, linestyle='--')
                            ax.add_patch(arrow)

                    
                    ax.text(0, 69, f'{selected_player}\nBox Entries: {len(hbentry)}', color=hcol, fontsize=25, fontweight='bold', ha='left', va='bottom')
                    ax.scatter(46, 6, s=2000, marker='s', color=hcol, zorder=3)
                    ax.scatter(46, 34, s=2000, marker='s', color=hcol, zorder=3)
                    ax.scatter(46, 62, s=2000, marker='s', color=hcol, zorder=3)
                    ax.text(46, 6, f'{len(hleft)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
                    ax.text(46, 34, f'{len(hcent)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
                    ax.text(46, 62, f'{len(hrigt)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
                    home_data = {
                        'Team_Name': hteamName,
                        'Total_Box_Entries': len(hbentry),
                        'Box_Entry_From_Left': len(hleft),
                        'Box_Entry_From_Center': len(hcent),
                        'Box_Entry_From_Right': len(hrigt)
                    }
                    return [home_data]

                fig,ax=plt.subplots(figsize=(10,10), facecolor=bg_color)
                box_entry_stats = box_entry(ax)
                box_entry_stats_df = pd.DataFrame(box_entry_stats)
                st.header("Box Entry Statistics")

                st.pyplot(fig)

                def Crosses(ax):
                    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
                    pitch.draw(ax=ax)
                    ax.set_ylim(-0.5,68.5)
                    ax.set_xlim(-0.5,105.5)

                    home_cross = df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Cross')) & (~df['qualifiers'].str.contains('Corner'))]

                    hsuc = 0
                    hunsuc = 0
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
                    # Headlines and other texts
                    home_left = len(home_cross[home_cross['y']>=34])
                    home_right = len(home_cross[home_cross['y']<34])

                    ax.text(51, 2, f"Crosses from\nLeftwing: {home_left}", color=hcol, fontsize=15, va='bottom', ha='right')
                    ax.text(51, 66, f"Crosses from\nRightwing: {home_right}", color=hcol, fontsize=15, va='top', ha='right')

                    ax.text(0,-2, f"Successful: {hsuc}", color=hcol, fontsize=20, ha='left', va='top')
                    ax.text(0,-5.5, f"Unsuccessful: {hunsuc}", color=line_color, fontsize=20, ha='left', va='top')

                    ax.text(0, 70, f"{selected_player}\n<---Crosses", color=hcol, size=25, ha='left', fontweight='bold')
                    home_data = {
                        'Team_Name': hteamName,
                        'Total_Cross': hsuc + hunsuc,
                        'Successful_Cross': hsuc,
                        'Unsuccessful_Cross': hunsuc,
                        'Cross_From_LeftWing': home_left,
                        'Cross_From_RightWing': home_right
                    }

                    
                    return [home_data]

                fig,ax=plt.subplots(figsize=(10,10), facecolor=bg_color)
                cross_stats = Crosses(ax)
                cross_stats_df = pd.DataFrame(cross_stats)
                st.header("Crosses Statistics")

                st.pyplot(fig)

