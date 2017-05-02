import os
import numpy as np
import pandas as pd
import config as cfg
from copy import deepcopy

class Data(object):
    def __init__(self, category, feature_scale=False, feature_select=False, problem_type='C'):
        self.category = category
        self.problem_type = problem_type
        self.feature_scale = feature_scale
        self.feature_select = feature_select

    def _scale_features(self):
        raise NotImplementedError("Not implemented yet!")

    def _select_data(self, modality, q_category, split, size='all'):
        scale = 'normalize' if self.feature_scale else 'regular'
        p_type = 'classify' if self.problem_type == 'C' else 'estimate'

        if self.feature_select:
            file_ = os.path.join(cfg.SEL_FEAT, scale, p_type, split,
                                 '{}_{}_{}.csv'.format(q_category, modality, split))
        else:
            file_ = os.path.join(cfg.ALL_FEAT, scale, p_type, split,
                                 '{}_{}_{}.csv'.format(q_category, modality, split))
        
        data = pd.read_csv(file_)
        if split == "train" and size != "all":
            split_file = cfg.TRAIN_SPLIT_FILE
            split_df = pd.read_csv(split_file, usecols=['Participant_ID'])
            split_df = split_df.loc[:int(size) - 1]
            data = data[data['video'].isin(split_df['Participant_ID'])]
        return self._group_features(data)

    def _group_features(self, data):
        y_label = 'label' if self.problem_type == 'C' else 'score'
        # print max(data[y_label])
        grouped = data.groupby('video')
        X = []
        y = []
        for video, group in grouped:
            X_person = []
            y_person = []
            for i in range(len(group)):
                X_person.append(group.iloc[i].tolist()[1:-2])
                y_person.append(group.iloc[i][y_label])
            X.append(X_person)
            y.append(y_person)
        return X, y

    def get_full_train(self, modality):
        x_train, y_train, x_val, y_val = self.get_data(modality)

        x_train[0].extend(x_val[0])
        x_train[1].extend(x_val[1])

        y_train[0].extend(y_val[0])
        y_train[1].extend(y_val[1])

        return x_train, y_train

    def get_test_data(self, modality):
        if self.category == 'PN':
            cat_1 = "positive"
            cat_2 = "negative"
        else:
            cat_1 = "discriminative"
            cat_2 = "nondiscriminative"

        print "Reading test data for {}".format(modality)

        x_test = [map(np.asarray, self._select_data(modality, cat_1, "test")[0]),
                 map(np.asarray, self._select_data(modality, cat_2, "test")[0])]
        y_test = [map(np.asarray, self._select_data(modality, cat_1, "test")[1]),
                 map(np.asarray, self._select_data(modality, cat_2, "test")[1])]
        return x_test, y_test

    def get_test_data_multi(self):
        x_a_test, y_a_test = self.get_test_data('acoustic')
        x_v_test, y_v_test = self.get_test_data('visual')
        x_l_test, y_l_test = self.get_test_data('linguistic')

        return [x_a_test, x_v_test, x_l_test], [y_a_test, y_v_test, y_l_test]

    def get_full_train_multi(self):
        x_a_train, y_a_train = self.get_full_train('acoustic')
        x_v_train, y_v_train = self.get_full_train('visual')
        x_l_train, y_l_train = self.get_full_train('linguistic')

        return [x_a_train, x_v_train, x_l_train], [y_a_train, y_v_train, y_l_train]

    def get_data(self, modality, size='all'):
        if self.category == 'PN':
            cat_1 = "positive"
            cat_2 = "negative"
        else:
            cat_1 = "discriminative"
            cat_2 = "nondiscriminative"

        print "Reading data for {}".format(modality)

        x_train = [map(np.asarray, self._select_data(modality, cat_1, "train", size=size)[0]),
                   map(np.asarray, self._select_data(modality, cat_2, "train", size=size)[0])]
        y_train = [map(np.asarray, self._select_data(modality, cat_1, "train", size=size)[1]),
                   map(np.asarray, self._select_data(modality, cat_2, "train", size=size)[1])]
        x_val = [map(np.asarray, self._select_data(modality, cat_1, "val")[0]),
                 map(np.asarray, self._select_data(modality, cat_2, "val")[0])]
        y_val = [map(np.asarray, self._select_data(modality, cat_1, "val")[1]),
                 map(np.asarray, self._select_data(modality, cat_2, "val")[1])]

        return x_train, y_train, x_val, y_val

    def get_multi_data(self, size='all'):
        X_A_train, y_A_train, X_A_val, y_A_val = self.get_data('acoustic', size)
        X_V_train, y_V_train, X_V_val, y_V_val = self.get_data('visual', size)
        X_L_train, y_L_train, X_L_val, y_L_val = self.get_data('linguistic', size)

        Xs = [X_A_train, X_V_train, X_L_train]
        ys = [y_A_train, y_V_train, y_L_train]
        Xs_val = [X_A_val, X_V_val, X_L_val]
        ys_val = [y_A_val, y_V_val, y_L_val]

        return Xs, ys, Xs_val, ys_val

    @staticmethod
    def concat_features(x1, x2, x3, y):
        if not len(x1) == len(x2) == len(x3) == 2:
            raise RuntimeError('Data sizes are not equal')
        elif not len(x1[0]) == len(x2[0]) == len(x3[0]):
            raise RuntimeError('Number of samples not equal')

        num_samples = len(x1[0])
        x = [[], []]
        y = deepcopy(y)
        for cat_idx in range(len(x)):
            for idx in xrange(num_samples):
                try:
                    stacked_data = np.hstack((x1[cat_idx][idx], x2[cat_idx][idx], x3[cat_idx][idx]))
                    x[cat_idx].append(stacked_data)
                except ValueError:
                    num_min_samples = min([data[cat_idx][idx].shape[0] for data in (x1, x2, x3)])
                    stacked_data = np.hstack((x1[cat_idx][idx][:num_min_samples, :],
                                              x2[cat_idx][idx][:num_min_samples, :],
                                              x3[cat_idx][idx][:num_min_samples, :]))
                    x[cat_idx].append(stacked_data)
                    y[cat_idx][idx] = y[cat_idx][idx][:num_min_samples]
        return x, y


if __name__ == '__main__':
    feat_data = Data('PN', feature_select=True, feature_scale=False, problem_type='C')
    # x_train, y_train, x_val, y_val = feat_data.get_data(modality='acoustic')
    X, Y = feat_data.get_test_data(modality='acoustic')
    # X, y = feat_data.get_full_train(modality='acoustic')
    print X[1][0].shape
    exit()
    X_A_train, y_A_train, X_A_val, y_A_val = feat_data.get_data('acoustic')
    X_V_train, y_V_train, X_V_val, y_V_val = feat_data.get_data('visual')
    X_L_train, y_L_train, X_L_val, y_L_val = feat_data.get_data('linguistic')

    feat_data.concat_features(X_A_train, X_V_train, X_L_train)
