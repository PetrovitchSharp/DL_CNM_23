import random
from collections import defaultdict
from typing import Any, List, Tuple


class CompanyNameDataLoader:
    ''' Класс с функционалом DataLoader'а - проходим по датасету с заданным
        размером батча, в каждом батче:
        - 1 опорный элемент
        - 1 элемент из того же кластера - positive
        - batch_size - 2 элемента из других классов - negative
    '''
    def __init__(self, data, shuffle: bool = True,
                 batch_size: int = 16, preprocessing: Any = None):
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        # Опорные items - будем проходить по всем таким названиям
        self.basis_items = data[data['cluster'] != -1]['name'].values
        if self.shuffle:
            random.shuffle(self.basis_items)
        self.size = len(self.basis_items)

        # Подготовим кластеры для выбора positive items
        data_with_clusters = self.data[self.data['cluster'] != -1].reset_index(
            drop=True)
        clusters = defaultdict(list)
        for i, row in data_with_clusters.iterrows():
            clusters[row['cluster']].append(row['name'])
        self.clusters = clusters

        # Список негативных элементов
        random.seed(2205)
        self.negative = list(data[data['cluster'] == -1]['name'].values)
        random.shuffle(self.negative)
        random.seed()

    def __iter__(self):
        self.idx = 0
        self.negative_idx = 0
        self.other_clusters_idx = 0
        return self

    def __next__(self) -> Tuple[List[str], List[int]]:
        if self.idx == self.size:
            raise StopIteration
        # опорный элемент
        item = self.basis_items[self.idx]
        cluster_id = self.data[self.data['name'] == item]['cluster'].iloc[0]
        cluster = self.clusters[cluster_id]

        # выбираем для него positive элемент
        positive_candidates = set(cluster) - set([item])
        if self.shuffle:
            positive_item = random.choice(tuple(positive_candidates))
        else:
            positive_item = sorted(positive_candidates)[-1]

        # выбираем negative элементы из других кластеров
        other_clusters = list(self.data[self.data['cluster'] != cluster_id]['name'].values) # noqa E501
        other_items_n = (self.batch_size - 2) // 2
        if self.shuffle:
            other_clusters_items = random.sample(other_clusters, other_items_n)
        else:
            if self.other_clusters_idx + other_items_n >= len(other_clusters):
                self.other_clusters_idx = 0
            other_clusters_items = other_clusters[
                self.other_clusters_idx: self.other_clusters_idx + other_items_n # noqa E501
                ]
            self.other_clusters_idx += other_items_n

        # выбираем negative элементы из элементов без кластера
        negative_n = self.batch_size - other_items_n - 2
        if self.shuffle:
            negative_items = random.sample(self.negative, negative_n)
        else:
            negative_items = self.negative[self.negative_idx: self.negative_idx + negative_n] # noqa E501
            self.negative_idx += negative_n
            if self.negative_idx >= len(self.negative):
                self.negative_idx = 0

        self.idx += 1

        if self.preprocessing is not None:
            item = self.preprocessing(item)
            positive_item = self.preprocessing(positive_item)
            other_clusters_items = [self.preprocessing(s)
                                    for s in other_clusters_items]
            negative_items = [self.preprocessing(s) for s in negative_items]

        labels = list(range(self.batch_size))
        labels[0] = 1
        labels[1] = 1

        items = [item, positive_item] + other_clusters_items + negative_items
        return items, labels
