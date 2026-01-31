import pandas as pd
import numpy as np
import requests
import pulp

# ============= CONFIG =============
GITHUB_BASE = "https://raw.githubusercontent.com/maxwell-petitjean/fpl/refs/heads/main/"

#auto upcoming gw
URL0 = 'https://fantasy.premierleague.com/api/fixtures?future=1'
json0 = requests.get(URL0).json()
nxtGw = pd.DataFrame(json0)[['event']]
VAR_GW = pd.to_numeric(nxtGw["event"], errors="coerce").min()

#manual upcoming gw
#VAR_GW = 24

VAR_GW0,VAR_GW1,VAR_GW2,VAR_GW3,VAR_GW4 = VAR_GW-1,VAR_GW-2,VAR_GW-3,VAR_GW-4,VAR_GW-5
VAR_GW5,VAR_GW6,VAR_GW7,VAR_GW8,VAR_GW9 = VAR_GW-6,VAR_GW-7,VAR_GW-8,VAR_GW-9,VAR_GW-10
VAR_GW_LY = 38-(1)

VGW_NAME_1 = 'gw'+str(VAR_GW)
VGW_NAME_2 = 'gw'+str(VAR_GW+1)
VGW_NAME_3 = 'gw'+str(VAR_GW+2)
VGW_NAME_4 = 'gw'+str(VAR_GW+3)
VGW_NAME_5 = 'gw'+str(VAR_GW+4)
VGW_NAME_6 = 'gw'+str(VAR_GW+5)

VAR_REL1, VAR_REL2, VAR_REL3 = 'IPS', 'LEI', 'SOU'
VAR_PRO1, VAR_PRO2, VAR_PRO3 = 'BUR', 'LEE', 'SUN'
URL1 = 'https://fantasy.premierleague.com/api/bootstrap-static/'
URL2 = 'https://fantasy.premierleague.com/api/fixtures?future=1'

URL30 = 'https://fantasy.premierleague.com/api/event/'+str(VAR_GW0)+'/live/'
URL31 = 'https://fantasy.premierleague.com/api/event/'+str(VAR_GW1)+'/live/'
URL32 = 'https://fantasy.premierleague.com/api/event/'+str(VAR_GW2)+'/live/'
URL33 = 'https://fantasy.premierleague.com/api/event/'+str(VAR_GW3)+'/live/'
URL34 = 'https://fantasy.premierleague.com/api/event/'+str(VAR_GW4)+'/live/'
URL35 = 'https://fantasy.premierleague.com/api/event/'+str(VAR_GW5)+'/live/'
URL36 = 'https://fantasy.premierleague.com/api/event/'+str(VAR_GW6)+'/live/'
URL37 = 'https://fantasy.premierleague.com/api/event/'+str(VAR_GW7)+'/live/'
URL38 = 'https://fantasy.premierleague.com/api/event/'+str(VAR_GW8)+'/live/'
URL39 = 'https://fantasy.premierleague.com/api/event/'+str(VAR_GW9)+'/live/'


def load_csv(filename):
    url = GITHUB_BASE + filename
    try:
        return pd.read_csv(url, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(url, encoding="latin1")


def run_model(
    fpl_id,
    transfers,
    exclude_names,
    exclude_teams,
    include_names,
    budget,
    picks_data,
    optimisation_mode="next_6_gw",   # "next_6_gw" or "gw1_only" (Free Hit)
):

    # ---- Load API data ----
    json1 = requests.get(URL1).json()
    json2 = requests.get(URL2).json()

    # turn to json
    json30 = requests.get(URL30).json()
    json31 = requests.get(URL31).json()
    json32 = requests.get(URL32).json()
    json33 = requests.get(URL33).json()
    json34 = requests.get(URL34).json()
    json35 = requests.get(URL35).json()
    json36 = requests.get(URL36).json()
    json37 = requests.get(URL37).json()
    json38 = requests.get(URL38).json()
    json39 = requests.get(URL39).json()

    # turn to df
    gw_df0 = pd.DataFrame(json30)
    gw_df1 = pd.DataFrame(json31)
    gw_df2 = pd.DataFrame(json32)
    gw_df3 = pd.DataFrame(json33)
    gw_df4 = pd.DataFrame(json34)
    gw_df5 = pd.DataFrame(json35)
    gw_df6 = pd.DataFrame(json36)
    gw_df7 = pd.DataFrame(json37)
    gw_df8 = pd.DataFrame(json38)
    gw_df9 = pd.DataFrame(json39)

    gw_df = pd.concat([gw_df0,gw_df1,gw_df2,gw_df3,gw_df4,gw_df5,gw_df6,gw_df7,gw_df8,gw_df9])

    # historical csvs
    players_prev0 = load_csv("players_24.csv")
    fixtures_prev0 = load_csv("gws_24.csv")
    teams3 = load_csv("teams_24.csv")
    xm_manual = load_csv("xm_manual.csv")

    # ------------------------------------------------------------------------------------------
    # ---- TEAMS data ----
    teams = pd.DataFrame(json1['teams'])[['id','name','short_name','strength_attack_home','strength_defence_home','strength_attack_away','strength_defence_away']]
    teams.columns = ['team_id','team_name','team_code','str_o_h','str_d_h','str_o_a','str_d_a']

    # ---- POSITIONS data ----
    positions = pd.DataFrame(json1['element_types'])[['id','singular_name_short']]
    positions.columns = ['id','pos']

    # ------------------------------------------------------------------------------------------
    # ---- Players - Previous Seasons ----
    players_prev1 = players_prev0[['first_name','second_name','element_type','total_points','minutes']]
    players_prev1['name'] = players_prev1['first_name']+' '+players_prev1['second_name']
    players_prev2 = players_prev1[['name','minutes','total_points']]
    players_prev3 = players_prev2.copy()
    players_prev3.columns = ['name','minutes','points']
    players_prev5 = players_prev3.copy()
    players_prev5['pp90'] = players_prev5['points'] / (players_prev5['minutes']/90)
    players_prev6 = players_prev5.fillna(0)
    players_prev6['pp90'] = round(players_prev6['pp90'],2).clip(upper=10)
    players_prev7 = players_prev6.sort_values('points',ascending=False)

    # ------------------------------------------------------------------------------------------
    # ---- FORM -- THIS SEASON
    players = pd.DataFrame(json1['elements'])
    players1 = players[['id','first_name','second_name','team','element_type','now_cost','selected_by_percent','clearances_blocks_interceptions','recoveries','tackles','clean_sheets','expected_assists','expected_goals','total_points','minutes']]
    players2 = players1.merge(teams[['team_id', 'team_code']], left_on='team', right_on='team_id')\
                       .merge(positions[['id', 'pos']], left_on='element_type', right_on='id')
    players3 = players2.copy()
    players3['name'] = players3['first_name']+' '+players3['second_name']
    players3['cbrit'] = players3['clearances_blocks_interceptions']+players3['recoveries']+players3['tackles']
    players4 = players3[['id_x','name','team_code','pos','now_cost','selected_by_percent','cbrit','clean_sheets','expected_assists','expected_goals','total_points','minutes']]
    players4.columns = ['id','name','team','pos','cost','ownership','dc','cs','xa','xg','points','mins']

    # ------------------------------------------------------------------------------------------
    # ---- FORM -- LAST 10 (THIS YEAR)
    form0 = pd.json_normalize(gw_df['elements'])
    form1 = form0[['id','stats.defensive_contribution','stats.clean_sheets','stats.expected_assists','stats.expected_goals','stats.total_points','stats.minutes']]
    form1.columns = ['id_form','dc_form','cs_form','xa_form','xg_form','points_form','mins_form']
    form2 = form1.merge(players4[['id', 'name']], left_on='id_form', right_on='id')
    form3 = form2[['name','dc_form','cs_form','xa_form','xg_form','points_form','mins_form']]
    form3.columns = ['name_form','dc_form','cs_form','xa_form','xg_form','points_form','mins_form']
    form3['xa_form'] = form3['xa_form'].astype(float)
    form3['xg_form'] = form3['xg_form'].astype(float)

    # ---- FORM -- LAST 10 (LAST YEAR)
    fixtures_prev1 = fixtures_prev0.drop(fixtures_prev0[fixtures_prev0['GW']< VAR_GW_LY].index)
    fixtures_prev2 = fixtures_prev1[fixtures_prev1['position'] != 'AM']
    fixtures_prev3 = fixtures_prev2[['name','cbrit','clean_sheets','expected_assists','expected_goals','total_points','minutes']]
    fixtures_prev3.columns = ['name_form','dc_form','cs_form','xa_form','xg_form','points_form','mins_form']
    fixtures_prev3['dc_form'] = (fixtures_prev3['mins_form'] / 90 ) * 5

    # ---- FORM -- CONCATENATE LAST 10
    form4 = pd.concat([form3,fixtures_prev3])
    form40 = form4.merge(players4[['name','pos']], left_on='name_form', right_on='name')

    # ---- FORM -- CLEAN DATA
    form5 = pd.pivot_table(form40, values=['dc_form','cs_form','xa_form','xg_form','points_form','mins_form'], index=['name_form','pos'], aggfunc=[np.sum], fill_value=0)
    form6 = form5.reset_index()
    form6.columns = form6.columns.droplevel(0)
    form6.columns = ['name_form','pos_form','cs_form','dc_form','mins_form','points_form','xa_form','xg_form']

    form7 = form6.copy()
    form7['xm_form'] = round(form7['mins_form']/10,2)
    form8 = form7.copy()
    form8['pp90_form'] = round(form8['points_form'] / (form8['mins_form']/90),2)
    form9 = form8.copy()
    form9['csp_form'] = round((form9['cs_form']*4),2)
    form9['cspp90_form'] = round(form9['csp_form'] / (form9['mins_form']/90),2)
    form10 = form9.copy()
    form10['xap_form'] = form10['xa_form']*3
    form10['xgp_form'] = np.where((form10['pos_form'] == 'FWD'), form10['xg_form'] * 4,
                            np.where((form10['pos_form'] == 'MID'), form10['xg_form'] * 5,
                            np.where((form10['pos_form'] == 'DEF') | (form10['pos_form'] == 'GKP'), form10['xg_form'] * 6,0 )))
    form10['xop_form'] = form10['xap_form']+form10['xgp_form']
    form11 = form10.copy()
    form11['xopp90_form'] = round((form11['xop_form']) / (form11['mins_form']/90),2)
    form12 = form11.copy()
    form12['dcp_form'] = np.where((form12['pos_form'] == 'MID'), form12['dc_form'] / 6,
                            np.where((form12['pos_form'] == 'DEF'), form12['dc_form'] / 5,0 ))
    form12['dcp_form'] = round(form12['dcp_form'],2)
    form12['dcpp90_form'] = round((form12['dcp_form']) / (form12['mins_form']/90),2).clip(upper=2)
    form12['pred_pp90_form'] = form12['xopp90_form']+form12['cspp90_form']+form12['dcpp90_form']+2
    form12 = form12.fillna(0)

    # ------------------------------------------------------------------------------------------
    # ---- Merge player stats ----
    players_prev8 = players_prev7.copy()
    players_prev8.columns = ['name_ly','mins_ly','points_ly','pp90_ly']
    form13 = form12.copy()
    players5 = players4.merge(players_prev8, how="left", left_on='name', right_on='name_ly')\
                       .merge(form13,    how="left", left_on='name', right_on='name_form')

    # ------------------------------------------------------------------------------------------
    # ---- Build fixture difficulty ----
    team_gw_prev1 = fixtures_prev1[['GW','position','team','opponent_team','was_home','total_points','minutes']]
    team_gw_prev1['played60'] = np.where((team_gw_prev1['minutes'] > 60),1,0)
    team_gw_prev2 = team_gw_prev1.drop(team_gw_prev1[team_gw_prev1['played60'] == 0].index)
    team_gw_prev2['pos'] = np.where((team_gw_prev2['position'] == 'MID') | (team_gw_prev2['position'] == 'FWD'),'ATT', team_gw_prev2['position'])
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

    # ------------------------------------------------------------------------------------------
    # ---- Upcoming fixtures ----
    fixtures = pd.DataFrame(json2)[['event','team_h','team_a']]
    fixtures2 = fixtures.drop(fixtures[fixtures['event'] > (VAR_GW + 5)].index)
    fixtures2 = fixtures2.drop(fixtures2[fixtures2['event'] < VAR_GW].index)
    fixtures_h1 = fixtures2.pivot(index='team_h', columns='event', values='team_a').reset_index().fillna(0).astype(int)
    fixtures_a1 = fixtures2.pivot(index='team_a', columns='event', values='team_h').reset_index().fillna(0).astype(int)
    fixtures_h1.columns = ['team_h','a1','a2','a3','a4','a5','a6']
    fixtures_a1.columns = ['team_a','h1','h2','h3','h4','h5','h6']

    lookup_team1 = dict(zip(teams[['team_id','team_code']].team_id,teams[['team_id','team_code']].team_code))
    fixtures_h4 = fixtures_h1.replace(lookup_team1)
    fixtures_a4 = fixtures_a1.replace(lookup_team1)

    # ------------------------------------------------------------------------------------------
    # ---- Merge fixtures into players ----
    fix_att1 = team_gw_prev9[team_gw_prev9['pos'] == 'ATT']
    fix_def1 = team_gw_prev9[team_gw_prev9['pos'] == 'DEF']
    fix_gk1  = team_gw_prev9[team_gw_prev9['pos'] == 'GK']

    lookup_home_att = dict(zip(fix_att1['team_new'], fix_att1['ppgh']))
    lookup_home_def = dict(zip(fix_def1['team_new'], fix_def1['ppgh']))
    lookup_home_gk  = dict(zip(fix_gk1['team_new'],  fix_gk1['ppgh']))
    lookup_away_att = dict(zip(fix_att1['team_new'], fix_att1['ppga']))
    lookup_away_def = dict(zip(fix_def1['team_new'], fix_def1['ppga']))
    lookup_away_gk  = dict(zip(fix_gk1['team_new'],  fix_gk1['ppga']))

    # turn home fixtures into future points
    fixtures_home_att = fixtures_h4.copy()
    fixtures_home_def = fixtures_h4.copy()
    fixtures_home_gk  = fixtures_h4.copy()

    cols_to_replace1 = fixtures_home_att.columns[1:]
    cols_to_replace2 = fixtures_home_def.columns[1:]
    cols_to_replace3 = fixtures_home_gk.columns[1:]

    fixtures_home_att[cols_to_replace1] = fixtures_home_att[cols_to_replace1].replace(lookup_away_att)
    fixtures_home_def[cols_to_replace2] = fixtures_home_def[cols_to_replace2].replace(lookup_away_def)
    fixtures_home_gk[cols_to_replace3]  = fixtures_home_gk[cols_to_replace3].replace(lookup_away_gk)

    # turn away fixtures into future points
    fixtures_away_att = fixtures_a4.copy()
    fixtures_away_def = fixtures_a4.copy()
    fixtures_away_gk  = fixtures_a4.copy()

    cols_to_replace4 = fixtures_away_att.columns[1:]
    cols_to_replace5 = fixtures_away_def.columns[1:]
    cols_to_replace6 = fixtures_away_gk.columns[1:]

    fixtures_away_att[cols_to_replace4] = fixtures_away_att[cols_to_replace4].replace(lookup_home_att)
    fixtures_away_def[cols_to_replace5] = fixtures_away_def[cols_to_replace5].replace(lookup_home_def)
    fixtures_away_gk[cols_to_replace6]  = fixtures_away_gk[cols_to_replace6].replace(lookup_home_gk)

    new_cols = ['team','op1','op2','op3','op4','op5','op6']
    for df in [fixtures_home_att,fixtures_home_def,fixtures_home_gk,fixtures_away_att,fixtures_away_def,fixtures_away_gk]:
        df.columns = new_cols

    fixtures_att = pd.concat([fixtures_home_att,fixtures_away_att])
    fixtures_def = pd.concat([fixtures_home_def,fixtures_away_def])
    fixtures_gk  = pd.concat([fixtures_home_gk, fixtures_away_gk])

    fixtures_att1 = fixtures_att.groupby('team', as_index=False).sum()
    fixtures_def1 = fixtures_def.groupby('team', as_index=False).sum()
    fixtures_gk1  = fixtures_gk.groupby('team',  as_index=False).sum()

    fixtures_att1['fdr'] = fixtures_att1.iloc[:,1:].sum(axis=1)
    fixtures_def1['fdr'] = fixtures_def1.iloc[:,1:].sum(axis=1)
    fixtures_gk1['fdr']  = fixtures_gk1.iloc[:,1:].sum(axis=1)

    gkp_df = players5[players5['pos'] == 'GKP'].merge(fixtures_gk1, on='team', how='left')
    def_df = players5[players5['pos'] == 'DEF'].merge(fixtures_def1, on='team', how='left')
    att_df = players5[~players5['pos'].isin(['GKP', 'DEF'])].merge(fixtures_att1, on='team', how='left')
    players6 = pd.concat([gkp_df, def_df, att_df], ignore_index=True).sort_values(by='cost', ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------------------------------
    # ---- Add extra player data ----
    players7 = players6.copy()
    player_data = players[['first_name','second_name','ep_next','chance_of_playing_next_round']]
    player_data['name'] = player_data['first_name']+' '+player_data['second_name']
    player_data = player_data.drop(columns=['first_name', 'second_name'])
    player_data = player_data[['name','ep_next','chance_of_playing_next_round']]
    player_data.columns = ['name_fpl','ep_fpl','flag_fpl']
    players8 = players7.merge(player_data,left_on='name',right_on='name_fpl',how='left')

    # xmins
    players8 = players8.merge(xm_manual,left_on='name',right_on='xm_name',how='left')
    players8['xm_ty'] = players8['mins']/VAR_GW0
    players8['xm_ly'] = (players8['mins_ly']/38).clip(lower=40).fillna(0)
    players8['xm_max'] = np.where(players8['flag_fpl'].isna(),players8['xm_form'],players8['xm_form'] * (players8['flag_fpl']/100))
    players8['xm'] = round((players8['xm_max'] + players8['xm_ty']) / 2 ,2)
    players8['xm'] = np.where(players8['xm_manual'].isna(), players8['xm'], players8['xm_manual'])

    # get predicted points
    players9 = players8.copy()
    players9 = players9.fillna(0)
    players9['ep_fpl'] = pd.to_numeric(players9['ep_fpl'], errors='coerce')
    players9['mean_value'] = round(players9[['pred_pp90_form','pp90_ly','ep_fpl']].mean(axis=1),2)
    players9['mean_value'] = np.where(players9['pp90_ly']==0,players9[['pred_pp90_form','ep_fpl']].mean(axis=1),players9['mean_value'])
    players9['base_points'] = round( (players9['mean_value']/90) * players9['xm'] ,2)
    players9['bp_copy'] = players9['base_points']

    players9['gw1'] = round(players9[['op1','base_points','bp_copy']].mean(axis=1),2)
    players9['gw2'] = round(players9[['op2','base_points','bp_copy']].mean(axis=1),2)
    players9['gw3'] = round(players9[['op3','base_points','bp_copy']].mean(axis=1),2)
    players9['gw4'] = round(players9[['op4','base_points','bp_copy']].mean(axis=1),2)
    players9['gw5'] = round(players9[['op5','base_points','bp_copy']].mean(axis=1),2)
    players9['gw6'] = round(players9[['op6','base_points','bp_copy']].mean(axis=1),2)

    players9['predicted_points'] = players9[['gw1','gw2','gw3','gw4','gw5','gw6']].sum(axis=1)
    players9[['op1','op2','op3','op4','op5','op6','fdr']] = players9[['op1','op2','op3','op4','op5','op6','fdr']].round(2)

    players10 = players9.sort_values(by='points', ascending=False)
    players10 = players10[np.isfinite(players10['predicted_points'])]

    final_pos = positions[['pos','id']]
    final_pos.columns = ['pos_code','pos_id']
    players10 = players10.merge(final_pos,left_on='pos',right_on='pos_code',how='left')

    player_output = players10[['name','team','pos','cost','ownership','points','mins','xm','points_ly','xg','xa','cs','dc','predicted_points','base_points','mean_value','xopp90_form','pred_pp90_form','pp90_ly','ep_fpl']]

    player_output['xg'] = pd.to_numeric(player_output['xg'], errors='coerce')
    player_output['xa'] = pd.to_numeric(player_output['xa'], errors='coerce')

    player_output['xg_p90'] = ( player_output['xg'] / ( player_output['mins'] / 90 ) ).round(2)
    player_output['xa_p90'] = ( player_output['xa'] / ( player_output['mins'] / 90 ) ).round(2)
    player_output['cs_p90'] = ( player_output['cs'] / ( player_output['mins'] / 90 ) ).round(2)
    player_output['dc_p90'] = ( player_output['dc'] / ( player_output['mins'] / 90 ) ).round(2)

    player_output = player_output[['name','team','pos','cost','ownership','points','mins','xg_p90','xa_p90','cs','dc','dc_p90','xopp90_form','pred_pp90_form','base_points','predicted_points']]
    player_output = player_output.fillna(0)
    player_output = player_output.sort_values(by='predicted_points', ascending=False)

    output = players10[['id','name','team','pos','pos_id','cost','ownership','predicted_points','xm','fdr','gw1','gw2','gw3','gw4','gw5','gw6']]
    output.columns = ['name','team','pos','pos_id','cost','ownership','predicted_points','xm','fdr',VGW_NAME_1,VGW_NAME_2,VGW_NAME_3,VGW_NAME_4,VGW_NAME_5,VGW_NAME_6]
    output = output.sort_values(by='predicted_points', ascending=False)

    ----------------------------------------------------------------------------------------------------------
    # ---- FPL ID Squad Fetch (from picks_data) ----

    # ---- Current squad + derived sell prices (tenths) ----
    current_ids = [p["element"] for p in picks_data] if picks_data else []
    current_id_set = set(current_ids)
    
    # transfers_data should be passed into run_model; fallback to empty list
    transfers_data = transfers_data or []
    
    # Map: element -> (latest_time, "in"/"out", cost_in_tenths)
    latest_move = {}
    for t in transfers_data:
        ts = t.get("time", "")
        ein, eout = t.get("element_in"), t.get("element_out")
        ein_cost, eout_cost = t.get("element_in_cost"), t.get("element_out_cost")
    
        if ein is not None:
            prev = latest_move.get(ein)
            if prev is None or ts > prev[0]:
                latest_move[ein] = (ts, "in", ein_cost)
    
        if eout is not None:
            prev = latest_move.get(eout)
            if prev is None or ts > prev[0]:
                latest_move[eout] = (ts, "out", eout_cost)
    
    # Current now_cost map from bootstrap-static output (tenths)
    # Requires output["element"] to exist (bootstrap id renamed to element)
    now_cost_by_element = dict(zip(output["element"], output["cost"]))
    
    # Purchase price for current squad (tenths)
    # - If we have a transfer-in record, use that cost
    # - Otherwise assume purchase price == now_cost (conservative)
    purchase_price_by_element = {}
    for pid in current_ids:
        now = now_cost_by_element.get(pid, 0)
        mv = latest_move.get(pid)
    
        if mv and mv[1] == "in" and mv[2] is not None:
            purchase_price_by_element[pid] = int(mv[2])
        else:
            purchase_price_by_element[pid] = int(now)
    
    # Selling price rule (FPL):
    # - If price fell or unchanged: sell == now
    # - If price rose: sell == purchase + floor((now - purchase) / 2)
    def selling_price(now_cost: int, purchase_price: int) -> int:
        if now_cost <= purchase_price:
            return now_cost
        return purchase_price + (now_cost - purchase_price) // 2
    
    sell_price_by_element = {}
    for pid in current_ids:
        now = int(now_cost_by_element.get(pid, 0))
        pp = int(purchase_price_by_element.get(pid, now))
        sell_price_by_element[pid] = selling_price(now, pp)
    
    # Attach current/sell info to output
    output["is_current"] = output["element"].isin(current_id_set).astype(int)
    output["sell_price"] = output["element"].map(sell_price_by_element).fillna(0).astype(int)
    
    # Derive bank (ITB) in tenths:
    # bank = total_budget_cap - sum(purchase prices of current squad)
    # Note: conservative for players never transferred in (we assume pp == now)
    current_purchase_total = sum(purchase_price_by_element.values()) if purchase_price_by_element else budget
    bank = max(0, int(budget - current_purchase_total))

