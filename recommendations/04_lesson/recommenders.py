import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

from utils import prefilter_items


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, item_features, weighting=True):
        # your_code
        self.item_features = item_features
        self.data, self.user_item_matrix = self.prepare_matrix(data, item_features)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
    
    @staticmethod
    def prepare_matrix(data: pd.DataFrame, item_features: pd.DataFrame):
        data = prefilter_items(data, take_n_popular=5000, item_features=item_features)
        
        user_item_matrix = pd.pivot_table(data, 
                                          index='user_id', columns='item_id', 
                                          values='quantity',
                                          aggfunc='count', 
                                          fill_value=0
                                         ).astype(float)
        
        return data, user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                        regularization=regularization,
                                        iterations=iterations,  
                                        num_threads=num_threads,
                                        random_state=24)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        # your_code
#         Найдем все товары купленные пользователем, отсортируем,
#         оставим только популярные, из них извлечем первые N
#         Для каждого товара найдем похожий.
        top = self.data.loc[self.data['user_id'] == user, ['item_id', 'quantity']].\
                    groupby(['item_id'])['quantity'].count().reset_index().\
                    sort_values(by='quantity', ascending=False)
        top_n_items = np.array(top['item_id'])[:N]
#         Здесь столкнулся с такой проблемой, что у пользователя покупоку может быть меньше чем N.
#         Поэтому для каждого товара искать по 1 похожему не подойдет
#         Мы будем искать для каждого товара N // len(top_n_items) + 1 + 1 (доп +1 для самого товара)
#         Таким образом, у нас всегда будет нужное количество похожих товаров с минимальным количеством вычислений
#         Затем отсортируем эти товары по вероятности похожести, выбросим фиктивный товар и возьмем первые N
        find_N_similar = N // len(top_n_items) + 2
        sim_items = [similar_items for top_item in top_n_items
                                   for similar_items in self.model.similar_items(self.itemid_to_id[top_item],
                                                                                 N=find_N_similar)[1:]]
        sim_items.sort(key=lambda x: x[1], reverse=True)
        res = np.array([self.id_to_itemid[item[0]] for item in sim_items])
        res = res[res != 999999][:N]      
        
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        # your_code
        # Найдем 5 похожих пользователей, возьмем все товары которые они купили
        # Отбросим фиктивный товар и среди них извлечем N самых популярных по количеству покупок
        similar_users = self.model.similar_users(self.userid_to_id[user], N=6)[1:]
        similar_users = np.array(similar_users, dtype='int')[:, 0]
        
        top_items = self.data.loc[self.data['user_id'].isin(similar_users), ['item_id', 'quantity']].\
                                    groupby(['item_id'])['quantity'].count().reset_index().\
                                    sort_values(by='quantity', ascending=False)
        top_items = np.array(top_items['item_id'])
        res = top_items[top_items != 999999][:N]

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def user_recommendation(self, user, N=5):
        res = [self.id_to_itemid[rec[0]] for rec in 
                        self.model.recommend(userid=self.userid_to_id[user], 
                                             user_items=csr_matrix(self.user_item_matrix),
                                             N=N, 
                                             filter_already_liked_items=False, 
                                             filter_items=[self.itemid_to_id[999999]],
                                             recalculate_user=False)]
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
