from pathlib import Path
import pandas as pd
from tqdm.autonotebook import tqdm
from sklearn.model_selection import StratifiedKFold

ROOT = Path.cwd()
INPUT_ROOT = ROOT / 'data'
TRAIN_RESAMPLED_DIR = INPUT_ROOT / "train_audio_resampled"
N_FOLDS = 5
SEED = 42
train = pd.read_csv(INPUT_ROOT / "train_mod.csv")

tmp_list = []
for audio_d in tqdm(TRAIN_RESAMPLED_DIR.iterdir()):
    if not audio_d.exists():
        continue
    if audio_d.is_file():
            continue
    for wav_f in (audio_d.iterdir()):
        tmp_list.append([audio_d.name, wav_f.name, wav_f.as_posix()])
            
train_wav_path_exist = pd.DataFrame(
    tmp_list, columns=["ebird_code", "resampled_filename", "file_path"])

del tmp_list

train_all = pd.merge(
    train, train_wav_path_exist, on=["ebird_code", "resampled_filename"], how="inner")
    
skf = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
train_all["fold"] = -1
for fold_id, (train_index, val_index) in enumerate(skf.split(train_all, train_all["ebird_code"])):
    train_all.iloc[val_index, -1] = fold_id
    
print(train.shape)
print(train_wav_path_exist.shape)
print(train_all.shape)

train_all.to_csv(INPUT_ROOT / "train_all.csv", index=False)
print('load csv')
print(pd.read_csv(INPUT_ROOT / "train_all.csv").shape)
