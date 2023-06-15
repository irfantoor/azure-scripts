# update_data.py
# Author : Irfan TOOR <email@irfantoor.com>
#

# imports
import os
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------------------
# data paths -- adjust these paths and PCA parameters

globo_source = "../data/source/news-portal-user-interactions-by-globocom/"
data_source = "../data/source/"
data_cleaned = "../data/cleaned/"
data_target = "../irfantoor-recommend/recommend/data/"

# if you want to apply PCA for dimentional reduction
apply_pca = True
pca_components = 100 # originally in the embedding : 250

# -----------------------------------------------------------------------------------------
# item_metadata_nunique
print('preparing : item_metadata', end=' ')
item_metadata = pd.read_csv(os.path.join(globo_source, 'articles_metadata.csv'))
item_metadata.rename(columns={"article_id": "item", "category_id": "group"}, inplace=True)
print ('ok')

print('preparing : item_metadata_unique', end=' ')
item_metadat_nunique = item_metadata.nunique()
item_metadat_nunique.to_csv(os.path.join(data_cleaned, 'item_metadata_nunique.csv'))
print ('ok')

# -----------------------------------------------------------------------------------------
# users_clicks : ∑ clicks_hour_{i}.csv, where i=000 to 384

# initialize
print('preparing : user_clicks', end=' ')
data_path = os.path.join(globo_source, 'clicks')
file_ids = range(385)
clicks = []

# read all files
for i in file_ids:
    file = f'clicks_hour_%03d.csv'%(i)
    file_path = os.path.join(data_path, file)
    ds = pd.read_csv(file_path)

    for r in ds.to_numpy():
        clicks.append(r)

# convert to DataFrame
clicks = np.array(clicks)
users_clicks = pd.DataFrame(clicks, columns=ds.columns)
users_clicks = users_clicks[['user_id', 'click_article_id', 'click_timestamp']].rename(columns={"user_id":"user", "click_article_id":"item", "click_timestamp":"timestamp"})
print('ok')

# -----------------------------------------------------------------------------------------
# item_clicks.csv

print('preparing : item_clicks', end=' ')
item_clicks = users_clicks.groupby('item').count().sort_values(by='user', ascending=False)['user']
item_clicks = pd.DataFrame(
    {
        'items': item_clicks.keys(),
        'clicks': item_clicks.values,
    }
)
item_clicks.to_csv(os.path.join(data_cleaned, 'item_clicks.csv'), index=False)
print('ok')

# -----------------------------------------------------------------------------------------
# user_interactions_with_groups.csv

print('preparing : user_interactions_with_groups', end=' ')
user_interactions_with_groups = users_clicks.join(item_metadata.set_index('item'), on='item')[['user', 'item', 'timestamp', 'group']]
user_interactions_with_groups.to_csv(
    os.path.join(data_cleaned, 'user_interactions_with_groups.csv'),
    index=False
)
print('ok')

# -----------------------------------------------------------------------------------------
# user_interactions_nunique.csv

print('preparing : user_interactions_nunique', end=' ')
user_interactions_nunique = user_interactions_with_groups.nunique()
user_interactions_nunique.to_csv(os.path.join(data_cleaned, 'user_interactions_nunique.csv'))
user_interactions_nunique
print('ok')

# -----------------------------------------------------------------------------------------
# group_clicks.csv

print('preparing : group_clicks', end=' ')
group_clicks = user_interactions_with_groups[['group', 'item']].groupby(by='group').count().sort_values(by='item', ascending=False)['item']
group_clicks = pd.DataFrame(
    {
        'group': group_clicks.keys(),
        'clicks': group_clicks.values
    }
)
group_clicks.to_csv(os.path.join(data_cleaned, 'group_clicks.csv'), index=False)
print('ok')

# -----------------------------------------------------------------------------------------
# group_items.csv

print('preparing : group_items', end=' ')
item_metadata[['group', 'item']].to_csv(os.path.join(data_cleaned, 'group_items.csv'), index=False)
print('ok')

# -----------------------------------------------------------------------------------------
# item_features.csv

print('preparing : item_features', end=' ')
item_features = pd.DataFrame(
    pd.read_pickle(os.path.join(globo_source, 'articles_embeddings.pickle'))
)

# from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if apply_pca:
    print(' applying PCA ...', end=' ')
    # intialize pca and logistic regression model
    pca = PCA(n_components=pca_components)
    # lr = LogisticRegression(multi_class='auto', solver='liblinear')

    # fit and transform data
    sc = StandardScaler()
    X, y = item_features.iloc[:, 1:].values, item_features.iloc[:, 0].values

    X_std = sc.fit_transform(X)

    X_pca = pca.fit_transform(X_std)
    X_pca.shape

    item_features = pd.DataFrame(X_pca)
    # lr.fit(X_pca, y)

item_features.to_csv(os.path.join(data_cleaned, 'item_features.csv'), index=False)
print('ok')
# -----------------------------------------------------------------------------------------
# item_features_shape.csv

print('preparing : item_features_shape', end=' ')
item_features_shape = pd.DataFrame(
    {
        'shape':item_features.shape
    }
)
item_features_shape.to_csv(os.path.join(data_cleaned, 'item_features_shape.csv'), index=False)
print('ok')

# -----------------------------------------------------------------------------------------
# top_group_clicks.csv

print('preparing : top_group_clicks', end=' ')
top_group_clicks = group_clicks[group_clicks['clicks']>=100000]
top_group_clicks.to_csv(
    os.path.join(data_cleaned, 'top_group_clicks.csv'),
    index=False
)
print('ok')

# -----------------------------------------------------------------------------------------
# top_item_clicks.csv

print('preparing : top_item_clicks', end=' ')
top_item_clicks = item_clicks[item_clicks['clicks']>=10000]
top_item_clicks.to_csv(
    os.path.join(data_cleaned, 'top_item_clicks.csv'),
    index=False
)
print('ok')

# -----------------------------------------------------------------------------------------
# copier les fichier

import shutil

print('')
print('copying files: ')
source_list = [
    [data_source, '100k', '100k.txt'],
    
    [globo_source, '', 'articles_embeddings.pickle'],
    [globo_source, '', 'articles_metadata.csv'],

    [data_cleaned, '', 'group_clicks.csv'],
    [data_cleaned, '', 'group_items.csv'],
    [data_cleaned, '', 'item_clicks.csv'],
    [data_cleaned, '', 'item_features_shape.csv'],
    [data_cleaned, '', 'item_features.csv'],
    [data_cleaned, '', 'item_metadata_nunique.csv'],
    [data_cleaned, '', 'top_group_clicks.csv'],
    [data_cleaned, '', 'top_item_clicks.csv'],
    [data_cleaned, '', 'user_interactions_nunique.csv'],
    [data_cleaned, '', 'user_interactions_with_groups.csv'],
]

for item in source_list:
    src = os.path.join(item[0], item[1], item[2])
    dst = os.path.join(data_target, item[2])
    if not os.path.exists(dst):
        print(f"creating file: {dst}")
    else:
        print(f"file: {dst}, already exists, overwriting")

# -----------------------------------------------------------------------------------------
# summary.json

import json

print('')
print('preparing : summary', end=' ')
n_items = item_metadat_nunique['item']
n_clicked_items = user_interactions_nunique['item']
n_groups = item_metadat_nunique['group']
n_clicked_groups = user_interactions_nunique['group']

summary = {
    "n_items": int(n_items),
    "n_groups": int(n_groups),
    "n_features": int(item_features.shape[1]),
    "n_users": int(user_interactions_nunique['user']),
    "n_clicked_items": int(n_clicked_items),
    "n_clicked_items_percent": f"%2.2f%%"%(n_clicked_items/n_items*100),
    "n_clicked_groups": int(n_clicked_groups),
    "n_clicked_groups_percent": f"%2.2f%%"%(n_clicked_groups/n_groups*100),
}

with open(os.path.join(data_target, "summary.json"), "w+") as fp:
    json.dump(summary, fp)
print('ok')

print(json.dumps(summary, indent=4))
print("Azure-function deployment trigger ...")

