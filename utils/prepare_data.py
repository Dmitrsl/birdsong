import numba
from pathlib import Path
import pandas as pd

HEIGHT = 137
WIDTH = 236
ROOT = Path("./data/")
SEED = 2020

@numba.jit()
def prepare_image(datadir, featherdir, data_type='train',
                  submission=False, indices=[0, 1, 2, 3]):
    assert data_type in ['train', 'test']
    if submission:
        image_df_list = [pd.read_parquet(datadir / f'{ROOT}{data_type}_image_data_{i}.parquet')
                         for i in indices]
    else:
        image_df_list = [pd.read_feather(featherdir / f'{ROOT}{data_type}_image_data_{i}.feather')
                         for i in indices]

    print('image_df_list', len(image_df_list))
    HEIGHT = 137
    WIDTH = 236
    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    del image_df_list
    gc.collect()
    images = np.concatenate(images, axis=0)
    return images

def prepare_image_128(data_type='train',
                  submission=False, indices=[0, 1, 2, 3]):
    assert data_type in ['train', 'test']
    if submission:
        image_df_list = [pd.read_parquet(f'{ROOT}/{data_type}_image_data_{i}.parquet')
                         for i in indices]
    else:
        image_df_list = [pd.read_parquet(f'{ROOT}/{data_type}_image_data_{i}.parquet')
                         for i in indices]

    print('image_df_list', len(image_df_list))
    HEIGHT = 137
    WIDTH = 236 
    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    del image_df_list
    gc.collect()

    images = np.concatenate(images, axis=0)
    images = np.array([crop_resize(i) for i in images])

    return images


@numba.jit()
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

@numba.jit()
def crop_resize(img0, size=128, pad=16):
    img0 = 255 - img0
    img0[img0 < 80] = 0
    img0[img0 > 100] = 255
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))

def prepere_data(FOLD: int, SEED: int, ROOT: Path, indices=[0, 1, 2, 3]):
    
    train = pd.read_csv(f'{ROOT}/train.csv')
    train_labels_ = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

    train_images_ = prepare_image_128(indices=indices)

    mskf = MultilabelStratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)

    train_index, test_index  = next(mskf.split(train_images_, train_labels_))

    train_images = train_images_[train_index]
    test_images = train_images_[test_index]

    train_labels = train_labels_[train_index]
    test_labels = train_labels_[test_index]

    return train_images, test_images, train_labels, test_labels

def main():
    print('ok')

if __name__ == '__main__':
    main()