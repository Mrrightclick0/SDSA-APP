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
    st.title("Stat Finder")

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
                def players_passing_stats(pname):
                    dfpass = df[(df['type']=='Pass') & (df['name']==pname)]
                    acc_pass = dfpass[dfpass['outcomeType']=='Successful']
                    iac_pass = dfpass[dfpass['outcomeType']=='Unsuccessful']
                    
                    if len(dfpass) != 0:
                        accurate_pass_perc = round((len(acc_pass)/len(dfpass))*100, 2)
                    else:
                        accurate_pass_perc = 0
                    
                    pro_pass = acc_pass[(acc_pass['prog_pass']>=9.11) & (acc_pass['x']>=35) &
                                        (~acc_pass['qualifiers'].str.contains('CornerTaken|Freekick'))]
                    Thr_ball = dfpass[(dfpass['qualifiers'].str.contains('Throughball')) & (~dfpass['qualifiers'].str.contains('CornerTaken|Freekick'))]
                    Thr_ball_acc = Thr_ball[Thr_ball['outcomeType']=='Successful']
                    Lng_ball = dfpass[(dfpass['qualifiers'].str.contains('Longball')) & (~dfpass['qualifiers'].str.contains('CornerTaken|Freekick'))]
                    Lng_ball_acc = Lng_ball[Lng_ball['outcomeType']=='Successful']
                    Crs_pass = dfpass[(dfpass['qualifiers'].str.contains('Cross')) & (~dfpass['qualifiers'].str.contains('CornerTaken|Freekick'))]
                    Crs_pass_acc = Crs_pass[Crs_pass['outcomeType']=='Successful']
                    key_pass = dfpass[dfpass['qualifiers'].str.contains('KeyPass')]
                    big_chnc = dfpass[dfpass['qualifiers'].str.contains('BigChanceCreated')]
                    df_no_carry = df[df['type']!='Carry'].reset_index(drop=True)
                    pre_asst = df_no_carry[(df_no_carry['qualifiers'].shift(-1).str.contains('IntentionalGoalAssist')) & (df_no_carry['type']=='Pass') & 
                                        (df_no_carry['outcomeType']=='Successful') &  (df_no_carry['name']==pname)]
                    shot_buildup = df_no_carry[(df_no_carry['qualifiers'].shift(-1).str.contains('KeyPass')) & (df_no_carry['type']=='Pass') & 
                                        (df_no_carry['outcomeType']=='Successful') &  (df_no_carry['name']==pname)]
                    g_assist = dfpass[dfpass['qualifiers'].str.contains('IntentionalGoalAssist')]
                    fnl_thd = acc_pass[(acc_pass['endX']>=70) & (~acc_pass['qualifiers'].str.contains('CornerTaken|Freekick'))]
                    pen_box = acc_pass[(acc_pass['endX']>=88.5) & (acc_pass['endY']>=13.6) & (acc_pass['endY']<=54.4) & 
                                    (~acc_pass['qualifiers'].str.contains('CornerTaken|Freekick'))]
                    frwd_pass = dfpass[(dfpass['pass_or_carry_angle']>= -85) & (dfpass['pass_or_carry_angle']<= 85) & 
                                    (~dfpass['qualifiers'].str.contains('CornerTaken|Freekick'))]
                    back_pass = dfpass[((dfpass['pass_or_carry_angle']>= 95) & (dfpass['pass_or_carry_angle']<= 180) | 
                                        (dfpass['pass_or_carry_angle']>= -180) & (dfpass['pass_or_carry_angle']<= -95)) &
                                    (~dfpass['qualifiers'].str.contains('CornerTaken|Freekick'))]
                    side_pass = dfpass[((dfpass['pass_or_carry_angle']>= 85) & (dfpass['pass_or_carry_angle']<= 95) | 
                                        (dfpass['pass_or_carry_angle']>= -95) & (dfpass['pass_or_carry_angle']<= -85)) & 
                                    (~dfpass['qualifiers'].str.contains('CornerTaken|Freekick'))]
                    frwd_pass_acc = frwd_pass[frwd_pass['outcomeType']=='Successful']
                    back_pass_acc = back_pass[back_pass['outcomeType']=='Successful']
                    side_pass_acc = side_pass[side_pass['outcomeType']=='Successful']
                    corners = dfpass[dfpass['qualifiers'].str.contains('CornerTaken')]
                    corners_acc = corners[corners['outcomeType']=='Successful']
                    freekik = dfpass[dfpass['qualifiers'].str.contains('Freekick')]
                    freekik_acc = freekik[freekik['outcomeType']=='Successful']
                    thins = dfpass[dfpass['qualifiers'].str.contains('ThrowIn')]
                    thins_acc = thins[thins['outcomeType']=='Successful']
                    lngball = dfpass[(dfpass['qualifiers'].str.contains('Longball')) & (~dfpass['qualifiers'].str.contains('CornerTaken|Freekick'))]
                    lngball_acc = lngball[lngball['outcomeType']=='Successful']

                    if len(frwd_pass) != 0:
                        Forward_Pass_Accuracy = round((len(frwd_pass_acc)/len(frwd_pass))*100, 2)
                    else:
                        Forward_Pass_Accuracy = 0
                    
                    df_xT_inc = (dfpass[dfpass['xT']>0])['xT'].sum().round(2)
                    df_xT_dec = (dfpass[dfpass['xT']<0])['xT'].sum().round(2)
                    total_xT = dfpass['xT'].sum().round(2)
                    
                    return {
                        'Name': pname,
                        'Total_Passes': len(dfpass),
                        'Accurate_Passes': len(acc_pass),
                        'Miss_Pass': len(iac_pass),
                        'Passing_Accuracy': accurate_pass_perc,
                        'Progressive_Passes': len(pro_pass),
                        'Chances_Created': len(key_pass),
                        'Big_Chances_Created': len(big_chnc),
                        'Assists': len(g_assist),
                        'Pre-Assists': len(pre_asst),
                        'Buil-up_to_Shot': len(shot_buildup),
                        'Final_Third_Passes': len(fnl_thd),
                        'Passes_In_Penalty_Box': len(pen_box),
                        'Through_Pass_Attempts': len(Thr_ball),
                        'Accurate_Through_Passes': len(Thr_ball_acc),
                        'Crosses_Attempts': len(Crs_pass),
                        'Accurate_Crosses': len(Crs_pass_acc),
                        'Longballs_Attempts': len(lngball),
                        'Accurate_Longballs': len(lngball_acc),
                        'Corners_Taken': len(corners),
                        'Accurate_Corners': len(corners_acc),
                        'Freekicks_Taken': len(freekik),
                        'Accurate_Freekicks': len(freekik_acc),
                        'ThrowIns_Taken': len(thins),
                        'Accurate_ThrowIns': len(thins_acc),
                        'Forward_Pass_Attempts': len(frwd_pass),
                        'Accurate_Forward_Pass': len(frwd_pass_acc),
                        'Side_Pass_Attempts': len(side_pass),
                        'Accurate_Side_Pass': len(side_pass_acc),
                        'Back_Pass_Attempts': len(back_pass),
                        'Accurate_Back_Pass': len(back_pass_acc),
                        'xT_Increased_From_Pass': df_xT_inc,
                        'xT_Decreased_From_Pass': df_xT_dec,
                        'Total_xT_From_Pass': total_xT 
                    }


                pnames = df['name'].unique()

                # Create a list of dictionaries to store the counts for each player
                data = []

                for pname in pnames:
                    counts = players_passing_stats(pname)
                    data.append(counts)

                # Convert the list of dictionaries to a DataFrame
                passing_stats_df = pd.DataFrame(data)

                # Sort the DataFrame by 'pr_count' in descending order
                passing_stats_df = passing_stats_df.sort_values(by='xT_Decreased_From_Pass', ascending=False).reset_index(drop=True)
                cleaned_df = df.dropna(subset=['name', 'teamName'])
                def player_carry_stats(pname):
                    df_carry = df[(df['type']=='Carry') & (df['name']==pname)]
                    led_shot1 = df[(df['type']=='Carry') & (df['name']==pname) & (df['qualifiers'].shift(-1).str.contains('KeyPass'))]
                    led_shot2 = df[(df['type']=='Carry') & (df['name']==pname) & (df['type'].shift(-1).str.contains('Shot'))]
                    led_shot = pd.concat([led_shot1, led_shot2])
                    led_goal1 = df[(df['type']=='Carry') & (df['name']==pname) & (df['qualifiers'].shift(-1).str.contains('IntentionalGoalAssist'))]
                    led_goal2 = df[(df['type']=='Carry') & (df['name']==pname) & (df['type'].shift(-1)=='Goal')]
                    led_goal = pd.concat([led_goal1, led_goal2])
                    pro_carry = df_carry[(df_carry['prog_carry']>=9.11) & (df_carry['x']>=35)]
                    fth_carry = df_carry[(df_carry['x']<70) & (df_carry['endX']>=70)]
                    box_entry = df_carry[(df_carry['endX']>=88.5) & (df_carry['endY']>=13.6) & (df_carry['endY']<=54.4) &
                                ~((df_carry['x']>=88.5) & (df_carry['y']>=13.6) & (df_carry['y']<=54.6))]
                    disp = df[(df['type']=='Carry') & (df['name']==pname) & (df['type'].shift(-1)=='Dispossessed')]
                    df_to = df[(df['type']=='TakeOn') & (df['name']==pname)]
                    t_ons = df_to[df_to['outcomeType']=='Successful']
                    t_onu = df_to[df_to['outcomeType']=='Unsuccessful']
                    df_xT_inc = df_carry[df_carry['xT']>0]
                    df_xT_dec = df_carry[df_carry['xT']<0]
                    total_xT = df_carry['xT'].sum().round(2)
                    df_carry = df_carry.copy()
                    df_carry.loc[:, 'Length'] = np.sqrt((df_carry['x'] - df_carry['endX'])**2 + (df_carry['y'] - df_carry['endY'])**2)
                    median_length = round(df_carry['Length'].median(),2)
                    total_length = round(df_carry['Length'].sum(),2)
                    if len(df_to)!=0:
                        success_rate = round((len(t_ons)/len(df_to))*100, 2)
                    else:
                        success_rate = 0
                    
                    return {
                        'Name': pname,
                        'Total_Carries': len(df_carry),
                        'Progressive_Carries': len(pro_carry),
                        'Carries_Led_to_Shot': len(led_shot),
                        'Carries_Led_to_Goal': len(led_goal),
                        'Carrier_Dispossessed': len(disp),
                        'Carries_Into_Final_Third': len(fth_carry),
                        'Carries_Into_Penalty_Box': len(box_entry),
                        'Avg._Carry_Length': median_length,
                        'Total_Carry_Length': total_length,
                        'xT_Increased_From_Carries': df_xT_inc['xT'].sum().round(2),
                        'xT_Decreased_From_Carries': df_xT_dec['xT'].sum().round(2),
                        'Total_xT_From_Carries': total_xT,
                        'TakeOn_Attempts': len(df_to),
                        'Successful_TakeOns': len(t_ons)
                    }

                pnames = df['name'].unique()

                # Create a list of dictionaries to store the counts for each player
                data = []

                for pname in pnames:
                    counts = player_carry_stats(pname)
                    data.append(counts)

                # Convert the list of dictionaries to a DataFrame
                carrying_stats_df = pd.DataFrame(data)

                # Sort the DataFrame by 'pr_count' in descending order
                carrying_stats_df = carrying_stats_df.sort_values(by='Total_Carries', ascending=False).reset_index(drop=True)
                def player_shooting_stats(pname,):
                    # goal = df[(df['name']==pname) & (df['type']=='Goal') & (~df['qualifiers'].str.contains('BigChance'))]
                    # miss = df[(df['name']==pname) & (df['type']=='MissedShots') & (~df['qualifiers'].str.contains('BigChance'))]
                    # save = df[(df['name']==pname) & (df['type']=='SavedShot') & (~df['qualifiers'].str.contains('BigChance'))]
                    # post = df[(df['name']==pname) & (df['type']=='ShotOnPost') & (~df['qualifiers'].str.contains('BigChance'))]
                  

                    
                    op_sh = team_shot_data[(team_shot_data['playerName']==pname) & (team_shot_data['situation'].str.contains('RegularPlay|FastBreak|IndividualPlay')) & 
                                    (team_shot_data['isOwnGoal']==0)]

                    goal = team_shot_data[(team_shot_data['playerName']==pname) & (team_shot_data['eventType']=='Goal') & (team_shot_data['isOwnGoal']==0)]
                    miss = team_shot_data[(team_shot_data['playerName']==pname) & (team_shot_data['eventType']=='Miss')]
                    save = team_shot_data[(team_shot_data['playerName']==pname) & (team_shot_data['eventType']=='AttemptSaved') & (team_shot_data['isBlocked']==0)]
                    blok = team_shot_data[(team_shot_data['playerName']==pname) & (team_shot_data['eventType']=='AttemptSaved') & (team_shot_data['isBlocked']==1)]
                    post = team_shot_data[(team_shot_data['playerName']==pname) & (team_shot_data['eventType']=='Post')]

                    goal_bc = df[(df['name']==pname) & (df['type']=='Goal') & (df['qualifiers'].str.contains('BigChance'))]
                    miss_bc = df[(df['name']==pname) & (df['type']=='MissedShots') & (df['qualifiers'].str.contains('BigChance'))]
                    save_bc = df[(df['name']==pname) & (df['type']=='SavedShot') & (df['qualifiers'].str.contains('BigChance'))]
                    post_bc = df[(df['name']==pname) & (df['type']=='ShotOnPost') & (df['qualifiers'].str.contains('BigChance'))]

                    shots = df[(df['name']==pname) & ((df['type']=='Goal') | (df['type']=='MissedShots') | (df['type']=='SavedShot') | (df['type']=='ShotOnPost')) &
                            (~df['qualifiers'].str.contains('OwnGoal'))]
                    out_box = shots[shots['qualifiers'].str.contains('OutOfBox')]
                    shots = shots.copy()
                    shots.loc[:, 'Length'] = np.sqrt((shots['x'] - 105)**2 + (shots['y'] - 34)**2)
                    avg_dist = round(shots['Length'].mean(), 2)
                    xG = team_shot_data[(team_shot_data['playerName']==pname)]['expectedGoals'].sum().round(2)
                    xGOT = team_shot_data[(team_shot_data['playerName']==pname)]['expectedGoalsOnTarget'].sum().round(2)

                    return {
                        'Name': pname, 
                        'Total_Shots': len(shots),
                        'Open_Play_Shots': len(op_sh),
                        'Goals': len(goal),
                        'Shot_On_Post': len(post),
                        'Shot_On_Target': len(save),
                        'Shot_Off_Target': len(miss),
                        'Shot_Blocked': len(blok),
                        'Big_Chances': len(goal_bc)+len(miss_bc)+len(save_bc)+len(post_bc),
                        'Big_Chances_Missed': len(miss_bc)+len(save_bc)+len(post_bc),
                        'Shots_From_Outside_The_Box': len(out_box),
                        'Shots_From_Inside_The_Box': len(shots) - len(out_box),
                        'Avg._Shot_Distance': avg_dist,
                        'xG': xG,
                        'xGOT': xGOT
                    }

                pnames = df['name'].unique()

                # Create a list of dictionaries to store the counts for each player
                data = []

                for pname in pnames:
                    counts = player_shooting_stats(pname)
                    data.append(counts)

                # Convert the list of dictionaries to a DataFrame
                shooting_stats_df = pd.DataFrame(data)

                # Sort the DataFrame by 'pr_count' in descending order
                shooting_stats_df = shooting_stats_df.sort_values(by='Total_Shots', ascending=False).reset_index(drop=True)
                def player_pass_receiving_stats(pname):
                    dfp = df[(df['type']=='Pass') & (df['outcomeType']=='Successful') & (df['name'].shift(-1)==pname)]
                    dfkp = dfp[dfp['qualifiers'].str.contains('KeyPass')]
                    dfas = dfp[dfp['qualifiers'].str.contains('IntentionalGoalAssist')]
                    dfnt = dfp[dfp['endX']>=70]
                    dfpen = dfp[(dfp['endX']>=87.5) & (dfp['endY']>=13.6) & (dfp['endY']<=54.6)]
                    dfpro = dfp[(dfp['x']>=35) & (dfp['prog_pass']>=9.11) & (~dfp['qualifiers'].str.contains('CornerTaken|Frerkick'))]
                    dfcros = dfp[(dfp['qualifiers'].str.contains('Cross')) & (~dfp['qualifiers'].str.contains('CornerTaken|Frerkick'))]
                    dfxT = dfp[dfp['xT']>=0]
                    dflb = dfp[(dfp['qualifiers'].str.contains('Longball'))]
                    cutback = dfp[((dfp['x'] >= 88.54) & (dfp['x'] <= 105) & 
                                    ((dfp['y'] >= 40.8) & (dfp['y'] <= 54.4) | (dfp['y'] >= 13.6) & (dfp['y'] <= 27.2)) & 
                                    (dfp['endY'] >= 27.2) & (dfp['endY'] <= 40.8) & (dfp['endX'] >= 81.67))]
                    next_act = df[(df['name']==pname) & (df['type'].shift(1)=='Pass') & (df['outcomeType'].shift(1)=='Successful')]
                    ball_retain = next_act[(next_act['outcomeType']=='Successful') & ((next_act['type']!='Foul') & (next_act['type']!='Dispossessed'))]
                    if len(next_act) != 0:
                        ball_retention = round((len(ball_retain)/len(next_act))*100, 2)
                    else:
                        ball_retention = 0

                    if len(dfp) != 0:
                        name_counts = dfp['shortName'].value_counts()
                        name_counts_df = name_counts.reset_index()
                        name_counts_df.columns = ['name', 'count']
                        name_counts_df = name_counts_df.sort_values(by='count', ascending=False)  
                        name_counts_df = name_counts_df.reset_index()
                        r_name = name_counts_df['name'][0]
                        r_count = name_counts_df['count'][0]
                    else:
                        r_name = 'None'
                        r_count = 0        

                    return {
                        'Name': pname,
                        'Total_Passes_Received': len(dfp),
                        'Key_Passes_Received': len(dfkp), 
                        'Assists_Received': len(dfas),
                        'Progressive_Passes_Received': len(dfpro),
                        'Passes_Received_In_Final_Third': len(dfnt),
                        'Passes_Received_In_Opponent_Penalty_Box': len(dfpen),
                        'Crosses_Received': len(dfcros),
                        'Longballs_Received': len(dflb),
                        'Cutbacks_Received': len(cutback),
                        'Ball_Retention': ball_retention,
                        'xT_Received': dfxT['xT'].sum().round(2),
                        'Avg._Distance_Of_Pass_Receiving_Form_Oppo._Goal_Line': round(105-dfp['endX'].median(),2),
                        'Most_Passes_Received_From_(Count)': f'{r_name}({r_count})'
                    }

                pnames = df['name'].unique()

                # Create a list of dictionaries to store the counts for each player
                data = []

                for pname in pnames:
                    counts = player_pass_receiving_stats(pname)
                    data.append(counts)

                # Convert the list of dictionaries to a DataFrame
                player_pass_receiving_stats_df = pd.DataFrame(data)

                # Sort the DataFrame by 'pr_count' in descending order
                player_pass_receiving_stats_df = player_pass_receiving_stats_df.sort_values(by='Total_Passes_Received', ascending=False).reset_index(drop=True)
                def player_defensive_stats(pname):
                    playerdf = df[(df['name']==pname)]
                    ball_wins = playerdf[(playerdf['type']=='Interception') | (playerdf['type']=='BallRecovery')]
                    f_third = ball_wins[ball_wins['x']>=70]
                    m_third = ball_wins[(ball_wins['x']>35) & (ball_wins['x']<70)]
                    d_third = ball_wins[ball_wins['x']<=35]

                    tk = playerdf[(playerdf['type']=='Tackle')]
                    tk_u = playerdf[(playerdf['type']=='Tackle') & (playerdf['outcomeType']=='Unsuccessful')]
                    intc = playerdf[(playerdf['type']=='Interception')]
                    br = playerdf[playerdf['type']=='BallRecovery']
                    cl = playerdf[playerdf['type']=='Clearance']
                    fl = playerdf[playerdf['type']=='Foul']
                    ar = playerdf[(playerdf['type']=='Aerial') & (playerdf['qualifiers'].str.contains('Defensive'))]
                    ar_u = playerdf[(playerdf['type']=='Aerial') & (playerdf['outcomeType']=='Unsuccessful') & (playerdf['qualifiers'].str.contains('Defensive'))]
                    pass_bl = playerdf[playerdf['type']=='BlockedPass']
                    shot_bl = playerdf[playerdf['type']=='SavedShot']
                    drb_pst = playerdf[playerdf['type']=='Challenge']
                    drb_tkl = df[(df['name']==pname) & (df['type']=='Tackle') & (df['type'].shift(1)=='TakeOn') & (df['outcomeType'].shift(1)=='Unsuccessful')]
                    err_lat = playerdf[playerdf['qualifiers'].str.contains('LeadingToAttempt')]
                    err_lgl = playerdf[playerdf['qualifiers'].str.contains('LeadingToGoal')]
                    dan_frk = playerdf[(playerdf['type']=='Foul') & (playerdf['x']>16.5) & (playerdf['x']<35) & (playerdf['y']>13.6) & (playerdf['y']<54.4)]
                    prbr = df[(df['name']==pname) & ((df['type']=='BallRecovery') | (df['type']=='Interception')) & (df['name'].shift(-1)==pname) & 
                            (df['outcomeType'].shift(-1)=='Successful') &
                            ((df['type'].shift(-1)!='Foul') | (df['type'].shift(-1)!='Dispossessed'))]
                    if (len(br)+len(intc)) != 0:
                        post_rec_ball_retention = round((len(prbr)/(len(br)+len(intc)))*100, 2)
                    else:
                        post_rec_ball_retention = 0

                    return {
                        'Name': pname,
                        'Total_Tackles': len(tk),
                        'Tackles_Won': len(tk_u),
                        'Dribblers_Tackled': len(drb_tkl),
                        'Dribble_Past': len(drb_pst),
                        'Interception': len(intc),
                        'Ball_Recoveries': len(br), 
                        'Post_Recovery_Ball_Retention': post_rec_ball_retention,
                        'Pass_Blocked': len(pass_bl),
                        'Ball_Clearances': len(cl),
                        'Shots_Blocked': len(shot_bl),
                        'Aerial_Duels': len(ar),
                        'Aerial_Duels_Won': len(ar) - len(ar_u),
                        'Fouls_Committed': len(fl),
                        'Fouls_Infront_Of_Own_Penalty_Box': len(dan_frk),
                        'Error_Led_to_Shot': len(err_lat),
                        'Error_Led_to_Goal': len(err_lgl),
                        'Possession Win in Final third': len(f_third),
                        'Possession Win in Mid third': len(m_third),
                        'Possession Win in Defensive third': len(d_third)
                    }

                pnames = df['name'].unique()

                # Create a list of dictionaries to store the counts for each player
                data = []

                for pname in pnames:
                    counts = player_defensive_stats(pname)
                    data.append(counts)

                # Convert the list of dictionaries to a DataFrame
                player_defensive_stats_df = pd.DataFrame(data)

                # Sort the DataFrame by 'pr_count' in descending order
                player_defensive_stats_df = player_defensive_stats_df.sort_values(by='Total_Tackles', ascending=False).reset_index(drop=True)
                def touches_stats(pname):
                    df_player = df[df['name']==pname]
                    df_player = df_player[~df_player['type'].str.contains('SubstitutionOff|SubstitutionOn|Card|Carry')]

                    touches = df_player[df_player['isTouch']==1]
                    final_third = touches[touches['x']>=70]
                    pen_box = touches[(touches['x']>=88.5) & (touches['y']>=13.6) & (touches['y']<=54.4)]

                    points = touches[['x', 'y']].values
                    hull = ConvexHull(points)
                    area_covered = round(hull.volume,2)
                    area_perc = round((area_covered/(105*68))*100, 2)

                    df_player = df[df['name']==pname]
                    df_player = df_player[~df_player['type'].str.contains('CornerTaken|FreekickTaken|Card|CornerAwarded|SubstitutionOff|SubstitutionOn')]
                    df_player = df_player[['x', 'y', 'period']]
                    dfp_fh = df_player[df_player['period']=='FirstHalf']
                    dfp_sh = df_player[df_player['period']=='SecondHalf']
                    dfp_fpet = df_player[df_player['period']=='FirstPeriodOfExtraTime']
                    dfp_spet = df_player[df_player['period']=='SecondPeriodOfExtraTime']
                    
                    dfp_fh['distance_covered'] = np.sqrt((dfp_fh['x'] - dfp_fh['x'].shift(-1))**2 + (dfp_fh['y'] - dfp_fh['y'].shift(-1))**2)
                    dist_cov_fh = (dfp_fh['distance_covered'].sum()/1000).round(2)
                    dfp_sh['distance_covered'] = np.sqrt((dfp_sh['x'] - dfp_sh['x'].shift(-1))**2 + (dfp_sh['y'] - dfp_sh['y'].shift(-1))**2)
                    dist_cov_sh = (dfp_sh['distance_covered'].sum()/1000).round(2)
                    dfp_fpet['distance_covered'] = np.sqrt((dfp_fpet['x'] - dfp_fpet['x'].shift(-1))**2 + (dfp_fpet['y'] - dfp_fpet['y'].shift(-1))**2)
                    dist_cov_fpet = (dfp_fpet['distance_covered'].sum()/1000).round(2)
                    dfp_spet['distance_covered'] = np.sqrt((dfp_spet['x'] - dfp_spet['x'].shift(-1))**2 + (dfp_spet['y'] - dfp_spet['y'].shift(-1))**2)
                    dist_cov_spet = (dfp_spet['distance_covered'].sum()/1000).round(2)
                    tot_dist_cov = dist_cov_fh + dist_cov_sh + dist_cov_fpet + dist_cov_spet

                    df_speed = df.copy()
                    df_speed['len_Km'] = np.where((df_speed['type']=='Carry'), 
                                        np.sqrt((df_speed['x'] - df_speed['endX'])**2 + (df_speed['y'] - df_speed['endY'])**2)/1000, 0)
                    df_speed['speed'] = np.where((df_speed['type']=='Carry'), 
                                                (df_speed['len_Km']*60)/(df_speed['cumulative_mins'].shift(-1) - df_speed['cumulative_mins'].shift(1)) , 0)
                    speed_df = df_speed[(df_speed['name']==pname)]
                    avg_speed = round(speed_df['speed'].median(),2)
                    print(df_speed)

                    return {
                        'Name': pname,
                        'Total_Touches': len(touches),
                        'Touches_In_Final_Third': len(final_third),
                        'Touches_In_Opponent_Penalty_Box': len(pen_box),
                        'Total_Distance_Covered(Km)': tot_dist_cov,
                        'Total_Area_Covered': area_covered,

                    }
                    

                starters = df[df['isFirstEleven']==1]
                pnames = starters['name'].unique()[1:]

                # Create a list of dictionaries to store the counts for each player
                data = []

                for pname in pnames:
                    counts = touches_stats(pname)
                    data.append(counts)

                # Convert the list of dictionaries to a DataFrame
                touches_stats_df = pd.DataFrame(data)

                # Sort the DataFrame by 'pr_count' in descending order
                touches_stats_df
                # Streamlit app header
                    # Shooting Stats
                shooting_stats = player_shooting_stats(selected_player)
                shooting_stats_df = pd.DataFrame([shooting_stats])
                st.subheader("Shooting Statistics")
                st.dataframe(shooting_stats_df)

                    # Passing Stats
                passing_stats = players_passing_stats(selected_player)
                passing_stats_df = pd.DataFrame([passing_stats])
                st.subheader("Passing Statistics")
                st.dataframe(passing_stats_df)

                    # Carrying Stats
                carrying_stats = player_carry_stats(selected_player)
                carrying_stats_df = pd.DataFrame([carrying_stats])
                st.subheader("Carrying Statistics")
                st.dataframe(carrying_stats_df)

                    # Pass Receiving Stats
                pass_receiving_stats = player_pass_receiving_stats(selected_player)
                pass_receiving_stats_df = pd.DataFrame([pass_receiving_stats])
                st.subheader("Pass Receiving Statistics")
                st.dataframe(pass_receiving_stats_df)

                    # Defensive Stats
                defensive_stats = player_defensive_stats(selected_player)
                defensive_stats_df = pd.DataFrame([defensive_stats])
                st.subheader("Defensive Statistics")
                st.dataframe(defensive_stats_df)

                    # Touch Stats
                touch_stats = touches_stats(selected_player)
                touch_stats_df = pd.DataFrame([touch_stats])
                st.subheader("Touches Statistics")
                st.dataframe(touch_stats_df)

                    # Player Stats
