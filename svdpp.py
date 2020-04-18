import numpy as np


class SVDpp:
    def __init__(self, mat, f=20):
        self.mat = np.array(mat)
        self.f = f
        self.bi = {}
        self.bu = {}
        self.qi = {}
        self.pu = {}
        self.avg = np.mean(self.mat[:, 2])
        self.y = {}
        self.u_dict = {}
        for i in range(self.mat.shape[0]):
            user_id = self.mat[i, 0]
            item_id = self.mat[i, 1]
            self.u_dict.setdefault(user_id, [])
            self.u_dict[user_id].append(item_id)
            self.bi.setdefault(item_id, 0)
            self.bu.setdefault(user_id, 0)
            self.qi.setdefault(item_id, np.random.random((self.f, 1)) / 10 * np.sqrt(self.f))
            self.pu.setdefault(user_id, np.random.random((self.f, 1)) / 10 * np.sqrt(self.f))
            self.y.setdefault(item_id, np.zeros((self.f, 1)) + .1)

    def predict(self, user_id, item_id):  # Функция оценки прогноза
        # Роль #setdefault состоит в том, чтобы создавать новые элементы bi, bu, qi, pu и пользовательские рейтинги u_dict, когда пользователь или элемент не появился, и устанавливать начальное значение равным 0.
        self.bi.setdefault(item_id, 0)
        self.bu.setdefault(user_id, 0)
        self.qi.setdefault(item_id, np.zeros((self.f, 1)))
        self.pu.setdefault(user_id, np.zeros((self.f, 1)))
        self.y.setdefault(user_id, np.zeros((self.f, 1)))
        self.u_dict.setdefault(user_id, [])
        user_impl_prf, sqrt_Ru = self.getY(user_id, item_id)
        rating = self.avg + self.bi[item_id] + self.bu[user_id] + np.sum(
            self.qi[item_id] * (self.pu[user_id] + user_impl_prf))  # Формула оценки прогноза
        # Поскольку оценка варьируется от 1 до 5, когда оценка больше 5 или меньше 5, возвращается 5,1.
        if rating > 5:
            rating = 5
        if rating < 1:
            rating = 1
        return rating

    def getY(self, user_id, item_id):  # Рассчитать sqrt_Ru и Σyj
        Ru = self.u_dict[user_id]
        I_Ru = len(Ru)
        sqrt_Ru = np.sqrt(I_Ru)
        y_u = np.zeros((self.f, 1))
        if I_Ru == 0:
            user_impl_prf = y_u
        else:
            for i in Ru:
                y_u += self.y[i]
            user_impl_prf = y_u / sqrt_Ru

        return user_impl_prf, sqrt_Ru

    def train(self, steps=30, gamma=0.04, Lambda=0.15):  # Функция обучения, шаг - это количество итераций SGD.
        print('train data size', self.mat.shape)
        for step in range(steps):
            print('step', step + 1, 'is running')
            KK = np.random.permutation(self.mat.shape[0])
            # Алгоритм стохастического градиентного спуска
            rmse = 0.0
            for i in range(self.mat.shape[0]):
                j = KK[i]
                user_id = self.mat[j, 0]
                item_id = self.mat[j, 1]
                rating = self.mat[j, 2]
                predict = self.predict(user_id, item_id)
                user_impl_prf, sqrt_Ru = self.getY(user_id, item_id)
                eui = rating - predict
                rmse += eui ** 2
                self.bu[user_id] += gamma * (eui - Lambda * self.bu[user_id])
                self.bi[item_id] += gamma * (eui - Lambda * self.bi[item_id])
                self.pu[user_id] += gamma * (eui * self.qi[item_id] - Lambda * self.pu[user_id])
                self.qi[item_id] += gamma * (eui * (self.pu[user_id] + user_impl_prf) - Lambda * self.qi[item_id])
                for j in self.u_dict[user_id]:
                    self.y[j] += gamma * (eui * self.qi[j] / sqrt_Ru - Lambda * self.y[j])

            gamma = 0.93 * gamma
            print('rmse is', np.sqrt(rmse / self.mat.shape[0]))

    def test(self, test_data):  # Гамма уменьшается со скоростью обучения 0,93

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
