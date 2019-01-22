import numpy as np
import scipy as sp
import pandas as pd


class recommendationsystem:

    def __init__(self, df, similarity_func, target, closer_count):
        self.df = df
        self.target = target
        self.closer_count = closer_count
        self.similarity_func = similarity_func



    def euclidean_similarity(self, vector_1, vector_2):
        import numpy as np
        idx = vector_1.nonzero()[0]
        vector_1 = vector_1[idx]
        vector_2 = vector_2[idx]

        idx = vector_2.nonzero()[0]
        vector_1 = vector_1[idx]
        vector_2 = vector_2[idx]

        return np.linalg.norm(vector_1 - vector_2)


    def cosine_similarity(self, vector_1, vector_2):
        import scipy as sp
        idx = vector_1.nonzero()[0]
        vector_1 = vector_1[idx]
        vector_2 = vector_2[idx]

        idx = vector_2.nonzero()[0]
        vector_1 = vector_1[idx]
        vector_2 = vector_2[idx]

        return 1 - sp.spatial.distance.cosine(vector_1, vector_2)

    def similarity_matrix(self):
        import numpy as np
        import scipy as sp
        import pandas as pd
        # 인덱스 데이터 저장
        index = self.df.index

        matrix = []
        for idx_1, value_1 in self.df.iterrows():
            row = []
            for idx_2, value_2 in self.df.iterrows():
                if self.similarity_func =='cosine':
                    row.append(self.cosine_similarity(value_1, value_2))
                elif self.similarity_func == 'euclidean':
                    row.append(self.euclidean_similarity(value_1, value_2))
                else:
                    raise FunctionError
            matrix.append(row)
        sim_df = pd.DataFrame(matrix, columns=index, index=index)

        return sim_df


    def mean_score(self):
        import numpy as np
        import scipy as sp
        import pandas as pd

        sim_df = self.similarity_matrix()
        ms_df = sim_df.drop(self.target)
        ms_df = ms_df.sort_values(self.target, ascending=False)[:self.closer_count]
        ms_df = self.df.loc[ms_df.index]

        pred_df = pd.DataFrame(columns=self.df.columns)
        pred_df.loc['user'] = self.df.loc[self.target]
        pred_df.loc['mean'] = ms_df.mean()

        return pred_df


    def run(self):
        import numpy as np
        import scipy as sp
        import pandas as pd

        sim_df = self.similarity_matrix()
        pred_df = self.mean_score()
        recommend_df = pred_df.T
        recommend_df = recommend_df[recommend_df['user']==0].sort_values('mean',ascending=False)

        return list(recommend_df.index)


    def mse(self):
        import numpy as np
        import scipy as sp
        import pandas as pd

        pred_df = self.mean_score()
        value = pred_df.loc['user']
        pred = pred_df.loc['mean']
        #평가할 때 없는데이터는 평가할 수 없으므로
        # value에서 0인 데이터 제거
        idx = value.nonzero()[0]
        value, pred = np.array(value)[idx], np.array(pred)[idx]

        # pred에서 0인 데이터 제거
        idx = pred.nonzero()[0]
        value, pred = np.array(value)[idx], np.array(pred)[idx]

        #print(value, pred.astype(int))
        return sum((value - pred) ** 2) / len(idx)

    def rmse(self):
        import numpy as np
        import scipy as sp
        import pandas as pd

        pred_df = self.mean_score()
        value = pred_df.loc['user']
        pred = pred_df.loc['mean']
        #평가할 때 없는데이터는 평가할 수 없으므로
        # value에서 0인 데이터 제거
        idx = value.nonzero()[0]
        value, pred = np.array(value)[idx], np.array(pred)[idx]

        # pred에서 0인 데이터 제거
        idx = pred.nonzero()[0]
        value, pred = np.array(value)[idx], np.array(pred)[idx]

        #print(value, pred.astype(int))
        return np.sqrt(sum((value - pred) ** 2) / len(idx))


    def mae(self):

        pred_df = self.mean_score()
        value = pred_df.loc['user']
        pred = pred_df.loc['mean']
        #평가할 때 없는데이터는 평가할 수 없으므로
        # value에서 0인 데이터 제거
        idx = value.nonzero()[0]
        value, pred = np.array(value)[idx], np.array(pred)[idx]

        # pred에서 0인 데이터 제거
        idx = pred.nonzero()[0]
        value, pred = np.array(value)[idx], np.array(pred)[idx]

        #print(value, pred.astype(int))
        return sum(np.absolute(value - pred)) / len(idx)

    def evaluate(self, algorithm):
                #df: sample_df, algorithm: 성능평가함수

        # user 리스트
        users = self.df.index
        sim_df = self.similarity_matrix()
        # user별 평가값의 모음
        evaluate_list = []

        for target in users:
            pred_df = self.mean_score() # 한 유저에 대한 pred_df
            evaluate_var = self.algorithm()
            evaluate_list.append(evaluate_var)

        return np.average(evaluate_list)
