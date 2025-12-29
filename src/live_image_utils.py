import os
import re
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd

def read_value(t, path, suffix=''):
    
    df = pd.read_csv(os.path.join(path, f"Img_t{(4-len(str(t)))*'0'+str(t)}-nuc_quant{suffix}.csv")).set_index('nucleus_id')
    #cols = [col for col in df.columns if col_phrase in col]
    #cols = cols+['time_point_index']
    df['time_point_index'] = t
    return df#[cols]

def read_img(t, path, suffix=''):
    df = pd.read_csv(os.path.join(path, f"Img_t{(4-len(str(t)))*'0'+str(t)}-img_quant{suffix}.csv"))#.set_index('nucleus_id')
    #cols = [col for col in df.columns if col_phrase in col]
    #cols = cols+['time_point_index']
    df['time_point_index'] = t
    return df#[cols]

def read_perinuclei(t, path, suffix=''):
    df = pd.read_csv(os.path.join(path, f"Img_t{(4-len(str(t)))*'0'+str(t)}-per_quant{suffix}.csv"))#.set_index('nucleus_id')
    #cols = [col for col in df.columns if col_phrase in col]
    #cols = cols+['time_point_index']
    df['time_point_index'] = t
    return df#[cols]



def rename_cols(df):
    rename_map = {}
    for col in df.columns:
        if "dna_" in col:
            rename_map[col] = col.replace("dna_", "h2b_")
        if "nuc_" in col:
            rename_map[col] = col.replace("nuc_", "h2b_")
        if "hoechst_" in col:
            rename_map[col] = col.replace("hoechst_", "h2b_")
        if "cellevent_" in col:
            rename_map[col] = col.replace("cellevent_", "ce_")
    return df.rename(rename_map, axis=1)

def extract_tracks(path):
    pi_path = os.path.join(path, "tracks_pi.csv")
    if os.path.exists(pi_path):
        tracks = pd.read_csv(pi_path, sep=",")
    else:
        tracks = pd.read_csv(os.path.join(path, "tracks.csv"), sep=",")
    try:
        dfs = [read_value(t, path, '_pi') for t in tracks["time_point_index"].unique()]
        img_df = pd.concat(
        [read_img(t, path, '_pi') for t in tracks["time_point_index"].unique()]
    ).set_index("time_point_index")
    except:
        dfs = [read_value(t, path) for t in tracks["time_point_index"].unique()]
        img_df = pd.concat(
        [read_img(t, path) for t in tracks["time_point_index"].unique()]
    ).set_index("time_point_index")
    #perinucs = [read_perinuclei(t, path) for t in tracks["time_point_index"].unique()]
    tracks = tracks.rename({"nucleus_index": "nucleus_id"}, axis=1).merge(
        pd.concat(dfs).reset_index(), on=["nucleus_id", "time_point_index"], how="left"
    )

    #tracks = tracks.merge(
    #    pd.concat(perinucs).reset_index().rename({"corresp_nucleus_id": "nucleus_id"}, axis=1), on=["nucleus_id", "time_point_index"], how="left", suffixes=('', '_per')   )
    
    
    img_df = rename_cols(img_df)
    tracks = rename_cols(tracks)
    tracks = tracks.merge(img_df, on="time_point_index", how="left", suffixes=("", "_img"))
    tracks["track_length"] = tracks["track_index"].map(
        tracks.groupby("track_index")["time_point_index"].count()
    )
    tracks["label"] = None
    tracks["experiment"] = "/".join(path.split("/")[:-1])
    tracks["well"] = path.split("/")[-1].split("-")[0]
    return tracks

track_columns = ['h2b_intensity_mean', 'ce_intensity_mean', 'pi_intensity_mean']#, 'pi_intensity_q4_mean']
def calculate_track_stats(df):
    for col in track_columns:
        df[col] = df[col]/df[col+'_img']
    stats = df[track_columns].agg(['mean', 'std', 'min', 'max'])  # Średnia, odchylenie, min, max
    
    first_values = df[track_columns].iloc[0]  # Wartość na początku tracku
    last_values = df[track_columns].iloc[-1]  # Wartość na końcu tracku
    
    # Średnia pochodna (diff) - różnica między kolejnymi punktami, uśredniona
    diff = df[track_columns].diff()
    
    stats.loc['first'] = first_values
    stats.loc['last'] = last_values
    stats.loc['mean_diff'] = diff.mean()
    stats.loc['max_diff'] = diff.max()
    stats.loc['min_diff'] = diff.min()
    stats['length'] = len(df)
    
    #stats['virus'] = df['virus'].unique()[0]
    #stats['NK'] = df['NK'].unique()[0]
    stats['beginning'] = df['time_point_index'].min()
    stats['end'] = df['time_point_index'].max()
    stats.loc['difference'] = stats.loc['max'] - stats.loc['min']
    df.set_index('time_point_index', inplace=True)
    stats.loc['max_t'] = df[track_columns].idxmax()
    return stats

def clean_data(track_stats):
    track_stats = track_stats.reset_index()
    track_stats = track_stats.pivot(index=['track_index', 'experiment', 'well'], columns='level_3')
    # Flatten the MultiIndex columns
    new_cols = []
    col_list = []
    for col in track_stats.columns:
        if col[0] in ['length', 'virus', 'NK', 'beginning', 'end'] and col[1]=='first':
            new_cols.append(col[0])
            col_list.append(col)
        elif col[0] not in ['length', 'virus', 'NK', 'beginning', 'end']:
            new_cols.append('_'.join(col[0].split('_')[:-1])+'_'+col[1])
            col_list.append(col)

    track_stats = track_stats[col_list]
    track_stats.columns = new_cols
    return track_stats

def extract_wells_info(path: str) -> pd.DataFrame:
    # Load the HTML content
    with open(os.path.join(path, "WellsInfo.html"), "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Find the table
    table = soup.find("table", {"class": "dataframe"})
    rows = table.find("tbody").find_all("tr")

    data = []
    current_row_header = None

    for row in rows:
        cols = row.find_all(["th", "td"])
        if len(cols) == 6:  # First row of a rowspan block
            current_row_header = cols[0].text.strip()
            col_vals = [current_row_header] + [col.text.strip() for col in cols[1:]]
        else:  # Next rows in the rowspan block
            col_vals = [current_row_header] + [col.text.strip() for col in cols]

        data.append(col_vals)

    # Define column names manually since header is complex
    df = pd.DataFrame(data, columns=["Row", "Column", "well", "Virus", "CellType1", "CellType2"])
    df["experiment"] = path
    return df

def extract_well(rootdir, model, n=20):
    print(rootdir)
    tracks = pd.DataFrame()
    for name in os.listdir(rootdir):
        subdir = os.path.join(rootdir, name)
        if os.path.isdir(subdir) and re.match(r"\d{4}-ST", subdir.split('/')[-1]):#fullmatch
            
            well_tracks = extract_tracks(subdir)
            well_tracks['All tracks'] = len(well_tracks)
            well_tracks['Eligible tracks'] = len(well_tracks[well_tracks['track_length']>n])
            tracks = pd.concat((tracks, well_tracks), axis=0)
    #print('All tracks:', len(tracks.groupby(['track_index', 'experiment', 'well'])))
    tracks = tracks[tracks['track_length']>n]
    #print('Eligible tracks:', tracks.groupby(['track_index', 'experiment', 'well']).count().drop_duplicates())
    
    track_stats = tracks.groupby(['track_index', 'experiment', 'well']).apply(calculate_track_stats)
    print('Cleaning data')
    track_stats = clean_data(track_stats)
    print('Predicting')
    if model is not None:
        track_stats['label'] = model.predict(track_stats.drop(['h2b_intensity_max_t', 'ce_intensity_max_t', 'pi_intensity_max_t'], axis=1))
    else:
        track_stats['label'] = None
    print('merging')
    track_stats = pd.merge(track_stats.reset_index(), extract_wells_info(rootdir), on=['well', 'experiment'])
    #track_stats = pd.merge(track_stats.reset_index(), tracks[['well', 'experiment', 'All tracks', 'Eligible tracks']], on=['well', 'experiment'], how='left')
    return track_stats


# Cache the contour file parsing results
def get_nuclei_number(path):
    """Gets number of nuclei from shuttletracker list of nuclei"""
    counter = 0
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "nucleus" in line.lower():
            counter+=1
    return counter

def collect_nuclei_data(root_path):
    data = []

    for dirpath in os.listdir(root_path):
        h2b=False
        dirpath = os.path.join(root_path, dirpath)
        if os.path.isfile(dirpath):
            continue
        
        # Updated pattern: allows optional "_h2b"
        pattern = r'Img_t\d{4}-nuclei_h2b\.txt$'
        
        for filename in os.listdir(dirpath):

            if re.match(pattern, filename):
                file_path = os.path.join(dirpath, filename)
                match = re.search(r't(\d{4})', filename)
                if not match:
                    continue
                time_point_index = int(match.group(1))

                well = re.sub(r'\D', '', os.path.basename(dirpath))
                experiment = root_path

                count = get_nuclei_number(file_path)

                data.append({
                    'well': well,
                    'experiment': experiment,
                    'time_point_index': time_point_index,
                    'nuclei_number': count,
                })
                h2b=True
        if not h2b:
            for filename in os.listdir(dirpath):
    
                if re.match(r'Img_t\d{4}-nuclei\.txt$', filename):
                    file_path = os.path.join(dirpath, filename)
                    match = re.search(r't(\d{4})', filename)
                    if not match:
                        continue
                    time_point_index = int(match.group(1))

                    well = re.sub(r'\D', '', os.path.basename(dirpath))
                    experiment = root_path

                    count = get_nuclei_number(file_path)

                    data.append({
                        'well': well,
                        'experiment': experiment,
                        'time_point_index': time_point_index,
                        'nuclei_number': count,
                    })
            
    return pd.DataFrame(data)
