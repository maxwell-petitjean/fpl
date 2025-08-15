# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pulp

# ================== CONFIG ==================
GITHUB_BASE = "https://raw.githubusercontent.com/maxwell-petitjean/fpl/refs/heads/main/"
VAR_GW = 1
VAR_REL1, VAR_REL2, VAR_REL3 = 'IPS', 'LEI', 'SOU'
VAR_PRO1, VAR_PRO2, VAR_PRO3 = 'BUR', 'LEE', 'SUN'
URL1 = 'https://fantasy.premierleague.com/api/bootstrap-static/'
URL2 = 'https://fantasy.premierleague.com/api/fixtures?future=1'

# ================== STREAMLIT PAGE CONFIG ==================
st.set_page_config(page_title="FPL Optimiser", layout="wide")
st.title("⚽ FPL Optimiser")
st.subheader("Optimise your Fantasy Premier League team")
st.caption("Hit 'Run Model' to get started.")

# ================== MARKDOWN ================================
st.markdown("""
<style>
div.stElementContainer {
    width: 100%;
}
div.stButton {
    width: 100%;
}
div.stButton > button:first-child {
    background: linear-gradient(90deg, #4CAF50, #45a049);
    color: white;
    padding: 0.6em 2em;
    border-radius: 10px;
    border: none;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    transition: all 0.3s ease;
}
div.stButton > button:first-child:hover {
    background: linear-gradient(90deg, #45a049, #4CAF50);
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ================== INPUT FORM LOGIC ==================
def get_inputs():
    fpl_id = st.text_input("FPL ID (not live yet - only available after gw1 fixtures consolidated data)")
    exclude_names = st.text_area(
        "Exclude Names (comma separated)",
        value="Rayan Aït-Nouri, Bryan Mbeumo"
    ).split(",")
    exclude_teams = st.text_area(
        "Exclude Teams (comma separated)",
        value="BRE"
    ).split(",")
    include_names = st.text_area(
        "Include Names (comma separated)"
    ).split(",")
    budget = st.number_input("Budget", value=1000, step=1)
    return fpl_id, exclude_names, exclude_teams, include_names, budget

# ================== INPUT PARAMETERS ==================
with st.expander("⚙️ Input Parameters", expanded=True):
    fpl_id_input, exclude_names_input, exclude_teams_input, include_names_input, budget_input = get_inputs()

# ================== HELPERS ==================
@st.cache_data
def load_csv(filename):
    url = GITHUB_BASE + filename
    return pd.read_csv(url)

# ============= MODEL FUNCTION =============
def run_model(fpl_id, exclude_names, exclude_teams, include_names, budget):
    # ---- Load API data ----
    json1 = requests.get(URL1).json()
    json2 = requests.get(URL2).json()

    teams = pd.DataFrame(json1['teams'])[['id','name','short_name','strength_attack_home','strength_defence_home','strength_attack_away','strength_defence_away']]
    teams.columns = ['team_id','team_name','team_code','str_o_h','str_d_h','str_o_a','str_d_a']

    positions = pd.DataFrame(json1['element_types'])[['id','singular_name_short']]
    positions.columns = ['id','pos']

    # ---- Load CSVs from GitHub ----
    players_prev_1 = load_csv("players_24.csv")
    players_prev_2 = load_csv("players_23.csv")
    fixtures_prev = load_csv("fixtures_24.csv")
    fixtures_prev0 = load_csv("gws_24.csv")
    teams3 = load_csv("teams_24.csv")
    xm_manual = load_csv("xm_manual.csv")

    # ---- Players - Previous Seasons ----
    players_prev_11 = players_prev_1[['first_name','second_name','element_type','total_points','minutes']]
    players_prev_21 = players_prev_2[['first_name','second_name','element_type','total_points','minutes']]

    players_prev0 = pd.concat([players_prev_11,players_prev_21])
    players_prev1 = players_prev0.reset_index()
    players_prev1['name'] = players_prev1['first_name']+' '+players_prev1['second_name']
    players_prev1 = players_prev1[players_prev1['element_type'] != 'AM']

    players_prev2 = pd.pivot_table(players_prev1, values=['total_points','minutes'], index=['name'], aggfunc=[np.sum], fill_value=0)
    players_prev3 = players_prev2.reset_index()
    players_prev3.columns = players_prev3.columns.droplevel(0)
    players_prev4 = players_prev3.rename({players_prev3.columns[0]:'name',players_prev3.columns[1]:'minutes',players_prev3.columns[2]:'points'}, axis='columns')
    players_prev5 = players_prev4.copy()
    players_prev5['pp90'] = players_prev5['points'] / (players_prev5['minutes']/90)
    players_prev6 = players_prev5.fillna(0)
    players_prev6['pp90'] = round(players_prev6['pp90'],2)
    players_prev7 = players_prev6.sort_values('points',ascending=False)

    # ---- Fixtures - Previous Season ----
    fixtures_prev1 = fixtures_prev0.drop(fixtures_prev0[fixtures_prev0['GW']< 29].index)
    fixtures_prev2 = fixtures_prev1[['GW','name','position','total_points','starts','minutes','expected_goals','expected_assists','clean_sheets']]
    fixtures_prev3 = fixtures_prev2[fixtures_prev2['position'] != 'AM']
    fixtures_prev4 = pd.pivot_table(fixtures_prev3, values=['total_points','starts','minutes','expected_goals','expected_assists','clean_sheets'], index=['name','position'], aggfunc=[np.sum], fill_value=0)
    fixtures_prev5 = fixtures_prev4.reset_index()
    fixtures_prev5.columns = fixtures_prev5.columns.droplevel(0)
    fixtures_prev5.columns = ['name','pos','cs','xa','xg','mins','starts','points']
    fixtures_prev6 = fixtures_prev5.copy()
    fixtures_prev6['xm'] = round(fixtures_prev6['mins']/10,2)
    fixtures_prev7 = fixtures_prev6.copy()
    fixtures_prev7['pp90'] = round(fixtures_prev7['points'] / (fixtures_prev7['mins']/90),2)
    fixtures_prev8 = fixtures_prev7.copy()
    fixtures_prev8['cspp90'] = round((fixtures_prev8['cs']*4) / (fixtures_prev8['mins']/90),2)
    fixtures_prev9 = fixtures_prev8.copy()
    fixtures_prev9['xap'] = fixtures_prev9['xa']*3
    fixtures_prev9['xgp'] = np.where((fixtures_prev9['pos'] == 'FWD'), fixtures_prev9['xg'] * 4,
                            np.where((fixtures_prev9['pos'] == 'MID'), fixtures_prev9['xg'] * 5,
                            np.where((fixtures_prev9['pos'] == 'DEF') | (fixtures_prev9['pos'] == 'GKP'), fixtures_prev9['xg'] * 6,0 )))
    fixtures_prev9['xop'] = fixtures_prev9['xap']+fixtures_prev9['xgp']
    fixtures_prev10 = fixtures_prev9.copy()
    fixtures_prev10['xopp90'] = round((fixtures_prev10['xop']) / (fixtures_prev10['mins']/90),2)
    fixtures_prev10 = fixtures_prev10.fillna(0)

    # ---- Merge player stats ----
    players_prev8 = players_prev7.copy()
    players_prev8.columns = ['name_l2y','mins_l2y','points_l2y','pp90_l2y']
    fixtures_prev11 = fixtures_prev10.copy()
    fixtures_prev11.columns = ['name_lyf','pos_lyf','cs_lyf','xa_lyf','xg_lyf','mins_lyf','starts_lyf','points_lyf','xm_lyf','pp90_lyf','cspp90_lyf','xap_lyf','xgp_lyf','xop_lyf','xopp90_lyf']

    players = pd.DataFrame(json1['elements'])
    players1 = players[['id','first_name','second_name','team','element_type','now_cost','selected_by_percent','clearances_blocks_interceptions','recoveries','tackles','clean_sheets','expected_assists','expected_goals','total_points','minutes']]
    players2 = players1.merge(teams[['team_id', 'team_code']], left_on='team', right_on='team_id').merge(positions[['id', 'pos']], left_on='element_type', right_on='id')
    players3 = players2.copy()
    players3['name'] = players3['first_name']+' '+players3['second_name']
    players3['cbrit'] = players3['clearances_blocks_interceptions']+players3['recoveries']+players3['tackles']
    players4 = players3[['id_x','name','team_code','pos','now_cost','selected_by_percent','cbrit','clean_sheets','expected_assists','expected_goals','total_points','minutes']]
    players4.columns = ['id','name','team','pos','cost','ownership','cbrit','cs','xa','xg','points','mins']

    players5 = players4.merge(players_prev8, left_on='name', right_on='name_l2y').merge(fixtures_prev11, left_on='name', right_on='name_lyf')

    # ---- Build fixture difficulty ----
    team_gw_prev1 = fixtures_prev1[['GW','position','team','opponent_team','was_home','total_points','minutes']]
    team_gw_prev1['played60'] = np.where((team_gw_prev1['minutes'] > 60),1,0)
    team_gw_prev2 = team_gw_prev1.drop(team_gw_prev1[team_gw_prev1['played60'] == 0].index)
    team_gw_prev2['pos'] = np.where((team_gw_prev2['position'] == 'MID')|(team_gw_prev2['position'] == 'FWD'),'ATT', team_gw_prev2['position'])
    team_gw_prev3 = pd.pivot_table(team_gw_prev2, values=['total_points','played60'], index=['opponent_team','was_home','pos'], aggfunc=[np.sum], fill_value=0)
    team_gw_prev3 = team_gw_prev3.reset_index()
    team_gw_prev3.columns = team_gw_prev3.columns.droplevel(0)
    team_gw_prev3.columns = ['team','home','pos','players','points']
    team_gw_prev4 = team_gw_prev3.copy()
    team_gw_prev4['ppg'] = team_gw_prev4['points'] / team_gw_prev4['players']
    team_gw_prev5 = team_gw_prev4.copy()
    team_gw_prev5['ppgh'] = np.where((team_gw_prev5['home'] == True),team_gw_prev5['ppg'],0)
    team_gw_prev5['ppga'] = np.where((team_gw_prev5['home'] == False),team_gw_prev5['ppg'],0)
    team_gw_prev6 = pd.pivot_table(team_gw_prev5, values=['ppgh','ppga'], index=['team','pos'], aggfunc=[np.sum], fill_value=0).reset_index()
    team_gw_prev6.columns = team_gw_prev6.columns.droplevel(0)
    team_gw_prev6.columns = ['team','pos','ppgh','ppga']
    team_gw_prev7 = team_gw_prev6.merge(teams3[['id', 'short_name']], left_on='team', right_on='id')
    team_gw_prev8 = team_gw_prev7[['short_name','pos','ppgh','ppga']]
    team_gw_prev8.columns = ['team_code','pos','ppgh','ppga']
    team_gw_prev8['team_new'] = np.where(team_gw_prev8['team_code'] == VAR_REL1, VAR_PRO1,
                                 np.where(team_gw_prev8['team_code'] == VAR_REL2, VAR_PRO2,
                                 np.where(team_gw_prev8['team_code'] == VAR_REL3, VAR_PRO3, team_gw_prev8['team_code'])))
    team_gw_prev9 = team_gw_prev8.merge(teams[['team_code','team_id']], left_on='team_new', right_on='team_code')

    # ---- Upcoming fixtures ----
    fixtures = pd.DataFrame(json2)[['event','team_h','team_a']]
    fixtures2 = fixtures.drop(fixtures[fixtures['event'] > (VAR_GW + 5)].index)
    fixtures2 = fixtures2.drop(fixtures2[fixtures2['event'] < VAR_GW].index)
    fixtures_h1 = fixtures2.pivot(index='team_h', columns='event', values='team_a').reset_index().fillna(0).astype(int)
    fixtures_a1 = fixtures2.pivot(index='team_a', columns='event', values='team_h').reset_index().fillna(0).astype(int)
    fixtures_h1.columns = ['team_h','a1','a2','a3','a4','a5','a6']
    fixtures_a1.columns = ['team_a','h1','h2','h3','h4','h5','h6']

    lookup_team1 = dict(zip(teams[['team_id','team_code']].team_id, teams[['team_id','team_code']].team_code))
    fixtures_h4 = fixtures_h1.replace(lookup_team1)
    fixtures_a4 = fixtures_a1.replace(lookup_team1)

    # ---- Merge fixtures into players ----
    fix_att1 = team_gw_prev9[team_gw_prev9['pos'] == 'ATT']
    fix_def1 = team_gw_prev9[team_gw_prev9['pos'] == 'DEF']
    fix_gk1 = team_gw_prev9[team_gw_prev9['pos'] == 'GK']

    lookup_home_att = dict(zip(fix_att1['team_new'], fix_att1['ppgh']))
    lookup_home_def = dict(zip(fix_def1['team_new'], fix_def1['ppgh']))
    lookup_home_gk = dict(zip(fix_gk1['team_new'], fix_gk1['ppgh']))
    lookup_away_att = dict(zip(fix_att1['team_new'], fix_att1['ppga']))
    lookup_away_def = dict(zip(fix_def1['team_new'], fix_def1['ppga']))
    lookup_away_gk = dict(zip(fix_gk1['team_new'], fix_gk1['ppga']))

    # turn home fixtures into future points
    fixtures_home_att = fixtures_h4.copy()
    fixtures_home_def = fixtures_h4.copy()
    fixtures_home_gk = fixtures_h4.copy()

    cols_to_replace1 = fixtures_home_att.columns[1:]
    cols_to_replace2 = fixtures_home_def.columns[1:]
    cols_to_replace3 = fixtures_home_gk.columns[1:]

    fixtures_home_att[cols_to_replace1] = fixtures_home_att[cols_to_replace1].replace(lookup_away_att)
    fixtures_home_def[cols_to_replace2] = fixtures_home_def[cols_to_replace2].replace(lookup_away_def)
    fixtures_home_gk[cols_to_replace3] = fixtures_home_gk[cols_to_replace3].replace(lookup_away_gk)

    # turn away fixtures into future points
    fixtures_away_att = fixtures_a4.copy()
    fixtures_away_def = fixtures_a4.copy()
    fixtures_away_gk = fixtures_a4.copy()

    cols_to_replace4 = fixtures_away_att.columns[1:]
    cols_to_replace5 = fixtures_away_def.columns[1:]
    cols_to_replace6 = fixtures_away_gk.columns[1:]

    fixtures_away_att[cols_to_replace4] = fixtures_away_att[cols_to_replace4].replace(lookup_home_att)
    fixtures_away_def[cols_to_replace5] = fixtures_away_def[cols_to_replace5].replace(lookup_home_def)
    fixtures_away_gk[cols_to_replace6] = fixtures_away_gk[cols_to_replace6].replace(lookup_home_gk)

    new_cols = ['team','op1','op2','op3','op4','op5','op6']
    for df in [fixtures_home_att,fixtures_home_def,fixtures_home_gk,fixtures_away_att,fixtures_away_def,fixtures_away_gk]:
        df.columns = new_cols

    fixtures_att = pd.concat([fixtures_home_att,fixtures_away_att])
    fixtures_def = pd.concat([fixtures_home_def,fixtures_away_def])
    fixtures_gk = pd.concat([fixtures_home_gk,fixtures_away_gk])

    fixtures_att1 = fixtures_att.groupby('team', as_index=False).sum()
    fixtures_def1 = fixtures_def.groupby('team', as_index=False).sum()
    fixtures_gk1 = fixtures_gk.groupby('team', as_index=False).sum()

    fixtures_att1['fdr'] = fixtures_att1.iloc[:,1:].sum(axis=1)
    fixtures_def1['fdr'] = fixtures_def1.iloc[:,1:].sum(axis=1)
    fixtures_gk1['fdr'] = fixtures_gk1.iloc[:,1:].sum(axis=1)

    gkp_df = players5[players5['pos'] == 'GKP'].merge(fixtures_gk1, on='team', how='left')
    def_df = players5[players5['pos'] == 'DEF'].merge(fixtures_def1, on='team', how='left')
    att_df = players5[~players5['pos'].isin(['GKP', 'DEF'])].merge(fixtures_att1, on='team', how='left')
    players7 = pd.concat([gkp_df, def_df, att_df], ignore_index=True).sort_values(by='cost', ascending=False).reset_index(drop=True)

    # ---- Add extra player data ----
    player_data = players[['first_name','second_name','ep_next','chance_of_playing_next_round','clean_sheets_per_90','defensive_contribution_per_90','expected_goals_per_90','expected_assists_per_90']]
    player_data['name'] = player_data['first_name']+' '+player_data['second_name']
    player_data = player_data.drop(columns=['first_name', 'second_name'])
    player_data.columns = ['ep_fpl','flag_fpl','csp90_fpl','dcpp90_fpl','xgp90_fpl','xap90_fpl','name_fpl']
    players8 = players7.merge(player_data,left_on='name',right_on='name_fpl',how='left')

    #xmins
    players8 = players8.merge(xm_manual,left_on='name_fpl',right_on='xm_name',how='left')
    players8['xm_l2y'] = players8['mins']/38
    players8['xm_max'] = np.where(players8['flag_fpl'].isna(), players8['xm_lyf'], (players8['xm_lyf']/90) * players8['flag_fpl'])
    players8['xm'] = round((players8['xm_max'] + players8['xm_l2y']) / 2 ,2)
    players8['xm'] = np.where(players8['xm_manual'].isna(), players8['xm'], players8['xm_manual'])

    players9 = players8.copy()
    players9['xapp90_fpl'] = players9['xap90_fpl']*3
    players9['cspp90_fpl'] = players9['csp90_fpl']*4
    players9['xgpp90_fpl'] = np.where((players9['pos'] == 'FWD'), players9['xgp90_fpl'] * 4,
                             np.where((players9['pos'] == 'MID'), players9['xgp90_fpl'] * 5,
                             np.where((players9['pos'] == 'DEF') | (players9['pos'] == 'GKP'), players9['xgp90_fpl'] * 6,0 )))
    players9['xopp90_fpl'] = players9['xapp90_fpl']+players9['xgpp90_fpl']
    players9['pred_lyf'] = players9['cspp90_lyf']+players9['xopp90_lyf']+2
    players9['pred_fpl'] = players9['cspp90_fpl']+players9['xopp90_fpl']+2
    players9['ep_fpl'] = pd.to_numeric(players9['ep_fpl'], errors='coerce')
    players9['mean_value'] = players9[['pp90_l2y','pred_lyf','pred_fpl','pp90_lyf','ep_fpl']].mean(axis=1)
    players9['base_points'] = round((players9['mean_value']/85) * players9['xm'] ,2)
    for w in ['gw1','gw2','gw3','gw4','gw5','gw6']:
        players9[w] = round(players9[[f'op{w[-1]}','base_points']].mean(axis=1),2)
    players9['net_points'] = players9[[f'gw{i}' for i in range(1,7)]].sum(axis=1)
    players9['net_points'] = np.where(players9['xm'].isna() , 0 , players9['net_points'] )
    players9['fdr'] = round(players9['fdr'],2)

    players10 = players9.sort_values(by='points', ascending=False)
    players10 = players10[np.isfinite(players10['net_points'])]
    players10 = players10[np.isfinite(players10['xm_lyf'])]

    final_pos = positions[['pos','id']]
    final_pos.columns = ['pos_code','pos_id']
    players10 = players10.merge(final_pos,left_on='pos',right_on='pos_code',how='left')
    output = players10[['name','team','pos','pos_id','cost','ownership','points','mins','xm','fdr','pp90_l2y','pp90_lyf','pred_lyf','pred_fpl','ep_fpl','base_points','gw1','gw2','gw3','gw4','gw5','gw6','net_points']]
    output = output[['name','team','pos','pos_id','cost','ownership','net_points','xm','fdr','base_points','gw1','gw2','gw3','gw4','gw5','gw6']]
    output = output.head(220)

    # ---- FPL ID Squad Fetch ----
    if fpl_id:
        picks_url = f"https://fantasy.premierleague.com/api/entry/{fpl_id}/event/{VAR_GW}/picks/"
        picks_data = requests.get(picks_url).json()
        player_ids = [p['element'] for p in picks_data['picks']]
        players_data = requests.get(URL1).json()['elements']
        id_to_name = {p['id']: p['web_name'] for p in players_data}
        include_names = [id_to_name[pid] for pid in player_ids if pid in id_to_name]

    # ---- LP Optimisation ----
    prob = pulp.LpProblem("FPL_Team_Selection", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("Player", output.index, 0, 1, pulp.LpBinary)

    prob += pulp.lpSum(player_vars[i] for i in output.index) == 15
    prob += pulp.lpSum(player_vars[i] for i in output.index if output.loc[i, 'pos'] == 'GKP') == 2
    prob += pulp.lpSum(player_vars[i] for i in output.index if output.loc[i, 'pos'] == 'DEF') == 5
    prob += pulp.lpSum(player_vars[i] for i in output.index if output.loc[i, 'pos'] == 'MID') == 5
    prob += pulp.lpSum(player_vars[i] for i in output.index if output.loc[i, 'pos'] == 'FWD') == 3
    prob += pulp.lpSum(output.loc[i, 'cost'] * player_vars[i] for i in output.index) <= budget

    for team in output['team'].unique():
        prob += pulp.lpSum(player_vars[i] for i in output.index if output.loc[i, 'team'] == team) <= 3

    for i in output.index:
        if output.loc[i, 'name'] in exclude_names or output.loc[i, 'team'] in exclude_teams:
            prob += player_vars[i] == 0
        if output.loc[i, 'name'] in include_names:
            prob += player_vars[i] == 1

    weeks = ['gw1', 'gw2', 'gw3', 'gw4', 'gw5', 'gw6']
    week_vars = {w: pulp.LpVariable.dicts(f"Week_{w}", output.index, 0, 1, pulp.LpBinary) for w in weeks}

    for w in weeks:
        for i in output.index:
            prob += week_vars[w][i] <= player_vars[i]  # Only selected players can start

        # Exactly 11 starters
        prob += pulp.lpSum(week_vars[w][i] for i in output.index) == 11

        # Weekly position constraints
        prob += pulp.lpSum(week_vars[w][i] for i in output.index if output.loc[i, 'pos'] == 'GKP') == 1
        prob += pulp.lpSum(week_vars[w][i] for i in output.index if output.loc[i, 'pos'] == 'DEF') >= 3
        prob += pulp.lpSum(week_vars[w][i] for i in output.index if output.loc[i, 'pos'] == 'DEF') <= 5
        prob += pulp.lpSum(week_vars[w][i] for i in output.index if output.loc[i, 'pos'] == 'MID') >= 2
        prob += pulp.lpSum(week_vars[w][i] for i in output.index if output.loc[i, 'pos'] == 'MID') <= 5
        prob += pulp.lpSum(week_vars[w][i] for i in output.index if output.loc[i, 'pos'] == 'FWD') >= 1
        prob += pulp.lpSum(week_vars[w][i] for i in output.index if output.loc[i, 'pos'] == 'FWD') <= 3

    prob += pulp.lpSum(output.loc[i, w] * week_vars[w][i] for w in weeks for i in output.index)
    prob.solve()

    selected_team = output[[player_vars[i].value() == 1 for i in output.index]].copy()
    selected_team['starting_weeks'] = selected_team.index.map(lambda i: ', '.join([w for w in weeks if week_vars[w][i].value() == 1]))
    selected_team = selected_team.sort_values(by=['pos_id', 'cost', 'net_points'], ascending=[True, False, False])
    selected_team = selected_team[['name','team','pos','cost','ownership','net_points','xm','fdr','base_points','gw1','gw2','gw3','gw4','gw5','gw6','starting_weeks']]

    return selected_team, output

# ===== SESSION STATE SETUP =====
if "final_team" not in st.session_state:
    st.session_state.final_team = None
if "raw_output" not in st.session_state:
    st.session_state.raw_output = None

# ================== RUN BUTTON ==================
if st.button("🚀 Run Model"):
    exclude_names_clean = [n.strip() for n in exclude_names_input if n.strip()]
    exclude_teams_clean = [t.strip() for t in exclude_teams_input if t.strip()]
    include_names_clean = [n.strip() for n in include_names_input if n.strip()]

    with st.spinner("Optimising your squad... please wait ⏳"):
        final_team, raw_output = run_model(
            fpl_id_input if fpl_id_input else None,
            exclude_names_clean,
            exclude_teams_clean,
            include_names_clean,
            budget_input
        )

    st.success("✅ Model run complete!")

    # Round numeric values
    for df in [final_team, raw_output]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(2)

    # Save to session state
    st.session_state.final_team = final_team
    st.session_state.raw_output = raw_output

# ======= If model has results, show tabs =======
if st.session_state.final_team is not None and st.session_state.raw_output is not None:

    # ======= Position color map =======
    def highlight_pos(val):
        color_map = {
            "GKP": "#FFD700",
            "DEF": "#90EE90",
            "MID": "#ADD8E6",
            "FWD": "#FFB6C1"
        }
        return f"background-color: {color_map.get(val, 'white')}"

    tab1, tab2, tab3 = st.tabs(["📋 Full Squad", "📊 Summary", "📄 Research Players"])

    # --- Tab 1 — Final Squad
    with tab1:
        numeric_cols = st.session_state.final_team.select_dtypes(include=[np.number]).columns
        styled_df = st.session_state.final_team.style.applymap(highlight_pos, subset=["pos"]) \
                                                    .background_gradient(subset=numeric_cols, cmap="YlGnBu") \
                                                    .format(precision=2)
        st.dataframe(styled_df, use_container_width=True, height=600)
        csv = st.session_state.final_team.to_csv(index=False)
        st.download_button("⬇️ Download squad as CSV", csv, "squad.csv", "text/csv")

    # --- Tab 2 — Summary
    with tab2:
        st.metric("💰 Total Cost", f"{st.session_state.final_team['cost'].sum():.2f}")
        st.metric("📈 Total Points", f"{st.session_state.final_team['net_points'].sum():.2f}")

    # --- Tab 3 — Raw Output with Position Filter ---
    with tab3:
        raw_output = st.session_state.raw_output.sort_values(by="net_points", ascending=False)

        positions = raw_output['pos'].unique().tolist()
        pos_filter = st.multiselect("Filter by position", options=positions, default=positions)
        filtered_df = raw_output[raw_output['pos'].isin(pos_filter)]
        numeric_cols_raw = filtered_df.select_dtypes(include=[np.number]).columns
        styled_raw = filtered_df.style.background_gradient(subset=numeric_cols_raw, cmap="YlGnBu") \
                                        .format(precision=2)
        st.dataframe(styled_raw, use_container_width=True, height=800)

