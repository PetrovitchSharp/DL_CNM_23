import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split


def main():
    data = pd.read_csv(
        'data/processed/X_train_companies_with_clusters.csv'
        )
    valid_share = 0.1

    # Разбиваем часть датасета, где известны кластеры
    data_with_clusters = data[data['cluster'] != -1]
    splitter = GroupShuffleSplit(test_size=valid_share,
                                 n_splits=1,
                                 random_state=2205)
    split = splitter.split(data_with_clusters,
                           groups=data_with_clusters['cluster'])
    train_inds, valid_inds = next(split)

    train_with_clusters = data_with_clusters.iloc[train_inds]
    valid_with_clusters = data_with_clusters.iloc[valid_inds]

    # Разделяем часть датасета, где нет известных кластеров
    data_with_clusters = data[data['cluster'] == -1]
    train_no_clusters, valid_no_clusters = train_test_split(
        data_with_clusters,
        random_state=2205,
        test_size=valid_share
    )

    train_data = pd.concat(
        [train_with_clusters, train_no_clusters]
        ).reset_index(drop=True)
    valid_data = pd.concat(
        [valid_with_clusters, valid_no_clusters]
        ).reset_index(drop=True)

    train_data.to_csv(
        'data/processed/train_companies_for_metric_learning.csv',
        index=False)
    valid_data.to_csv(
        'data/processed/valid_companies_for_metric_learning.csv',
        index=False)
    print('Train and valid data saved!')


if __name__ == '__main__':
    main()
