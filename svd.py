import numpy as np

import dataset as dp


class SVD:
    def __init__(self, matrix, f=20):
        # необходимо проинициализировать pu и qi, а также rui из train_data
        self.matrix = np.array(matrix)
        self.f = f
        self.bi = {}
        self.bu = {}
        self.qi = {}
        self.pu = {}
        self.avg = np.mean(self.matrix[:, 2])
        for i in range(self.matrix.shape[0]):
            user_id = self.matrix[i, 0]
            item_id = self.matrix[i, 1]
            self.bi.setdefault(item_id, 0)
            self.bu.setdefault(user_id, 0)
            # каждому фильму q c идентификатором item_id соответствует вектор из f элементов т.е. набор параметров фильма
            self.qi.setdefault(item_id, np.random.random((self.f, 1)) / 10 * np.sqrt(self.f))
            # каждому пользователю p c идентификатором user_id соответствует вектор из f элементов т.е. набор параметров пользователя
            self.pu.setdefault(user_id, np.random.random((self.f, 1)) / 10 * np.sqrt(self.f))

    def predict(self, user_id, item_id):
        # функция минимизации
        self.bi.setdefault(item_id, 0)
        self.bu.setdefault(user_id, 0)
        self.qi.setdefault(item_id, np.zeros((self.f, 1)))
        self.pu.setdefault(user_id, np.zeros((self.f, 1)))
        rating = self.avg + self.bi[item_id] + self.bu[user_id] + np.sum(
            self.qi[item_id] * self.pu[user_id])  # скалярное произведение

        if rating > 5:
            rating = 5
        if rating < 1:
            rating = 1
        return rating

    def train(self, steps=60, gamma=0.005, Lambda=0.02):
        # обучаем, применяя градиентный спуск
        print('train data size', self.matrix.shape)
        for step in range(steps):
            KK = np.random.permutation(
                self.matrix.shape[0])  # перестановка 80 000 оценок. Для 5 приблизительно так [1 4 3 0 2]
            for i in range(self.matrix.shape[0]):
                j = KK[i]
                user_id = self.matrix[j, 0]
                item_id = self.matrix[j, 1]
                rating = self.matrix[j, 2]
                eui = rating - self.predict(user_id, item_id)
                self.bu[user_id] += gamma * (eui - Lambda * self.bu[user_id])
                self.bi[item_id] += gamma * (eui - Lambda * self.bi[item_id])
                tmp = self.qi[item_id]
                self.qi[item_id] += gamma * (eui * self.pu[user_id] - Lambda * self.qi[item_id])
                self.pu[user_id] += gamma * (eui * tmp - Lambda * self.pu[user_id])
            gamma = 0.93 * gamma  # why?

    def test(self, test_data):
        # rmse
        test_data = np.array(test_data)
        print('test data size', test_data.shape)
        rmse = 0.0
        for i in range(test_data.shape[0]):
            user_id = test_data[i, 0]
            item_id = test_data[i, 1]
            rating = test_data[i, 2]
            eui = rating - self.predict(user_id, item_id)
            rmse += eui ** 2
        print('rmse of test data is', np.sqrt(rmse / test_data.shape[0]))
        return np.sqrt(rmse / test_data.shape[0])


train_data, test_data, data = dp.load_dataset()

f = 20
model = SVD(train_data, f)
model.train()
model.test(test_data)
