import os

import numpy as np
import torch
from models.HyperG.utils.meter import CIndexMeter
from sklearn.linear_model import Ridge, LinearRegression, Lasso


def lasso_regression(X_train, X_test, y_train, y_test, C_idx_evaluate=True):
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # print(type(X_train), type(y_train))
    # print(X_train.shape, y_train.shape)

    lasso = Lasso()
    lasso.fit(X_train, y_train)

    lasso001 = Lasso(alpha=0.01, max_iter=10e5)
    lasso001.fit(X_train, y_train)

    lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
    lasso00001.fit(X_train, y_train)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # cox = CoxPHFitter()
    # cox.fit(X_train, y_train)

    if C_idx_evaluate:
        for model in [lasso, lasso001, lasso00001, lr]:
            c_index = CIndexMeter()
            pred = torch.from_numpy(model.predict(X_test))
            st = torch.from_numpy(y_test)
            for item in zip(pred, st):
                c_index.add(*item)
            c_index_v = c_index.value()
            print(c_index_v)
    else:
        train_score = lasso.score(X_train, y_train)
        test_score = lasso.score(X_test, y_test)
        coeff_used = np.sum(lasso.coef_ != 0)
        print("training score:", train_score)
        print("test score: ", test_score)
        print("number of features used: ", coeff_used)

        train_score001 = lasso001.score(X_train, y_train)
        test_score001 = lasso001.score(X_test, y_test)
        coeff_used001 = np.sum(lasso001.coef_ != 0)

        print("training score for alpha=0.01:", train_score001)
        print("test score for alpha =0.01: ", test_score001)
        print("number of features used: for alpha =0.01:", coeff_used001)

        train_score00001 = lasso00001.score(X_train, y_train)
        test_score00001 = lasso00001.score(X_test, y_test)
        coeff_used00001 = np.sum(lasso00001.coef_ != 0)
        print("training score for alpha=0.0001:", train_score00001)
        print("test score for alpha =0.0001: ", test_score00001)
        print("number of features used: for alpha =0.0001:", coeff_used00001)

        lr_train_score = lr.score(X_train, y_train)
        lr_test_score = lr.score(X_test, y_test)
        print("LR training score:", lr_train_score)
        print("LR test score: ", lr_test_score)

        # plt.subplot(1, 2, 1)
        # plt.plot(lasso.coef_, alpha=0.7, linestyle='none', marker='*', markersize=5, color='red',
        #          label=r'Lasso; $\alpha = 1$', zorder=7)  # alpha here is for transparency
        # plt.plot(lasso001.coef_, alpha=0.5, linestyle='none', marker='d', markersize=6, color='blue',
        #          label=r'Lasso; $\alpha = 0.01$')  # alpha here is for transparency
        #
        # plt.xlabel('Coefficient Index', fontsize=16)
        # plt.ylabel('Coefficient Magnitude', fontsize=16)
        # plt.legend(fontsize=13, loc=4)
        # plt.subplot(1, 2, 2)
        # plt.plot(lasso.coef_, alpha=0.7, linestyle='none', marker='*', markersize=5, color='red',
        #          label=r'Lasso; $\alpha = 1$', zorder=7)  # alpha here is for transparency
        # plt.plot(lasso001.coef_, alpha=0.5, linestyle='none', marker='d', markersize=6, color='blue',
        #          label=r'Lasso; $\alpha = 0.01$')  # alpha here is for transparency
        # plt.plot(lasso00001.coef_, alpha=0.8, linestyle='none', marker='v', markersize=6, color='black',
        #          label=r'Lasso; $\alpha = 0.00001$')  # alpha here is for transparency
        # plt.plot(lr.coef_, alpha=0.7, linestyle='none', marker='o', markersize=5, color='green', label='Linear Regression',
        #          zorder=2)
        # plt.xlabel('Coefficient Index', fontsize=16)
        # plt.ylabel('Coefficient Magnitude', fontsize=16)
        # plt.legend(fontsize=13, loc=4)
        # plt.tight_layout()
        # plt.show()


def ridge_regression(X_train, X_test, y_train, y_test, C_idx_evaluate=True):
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    rr = Ridge(alpha=0.01)
    rr.fit(X_train, y_train)
    rr100 = Ridge(alpha=100)  # comparison with alpha value
    rr100.fit(X_train, y_train)
    if C_idx_evaluate:
        for model in [lr, rr, rr100]:
            c_index = CIndexMeter()
            pred = torch.from_numpy(model.predict(X_test))
            st = torch.from_numpy(y_test)
            for item in zip(pred, st):
                c_index.add(*item)
            c_index_v = c_index.value()
            print(c_index_v)
    else:
        train_score = lr.score(X_train, y_train)
        test_score = lr.score(X_test, y_test)
        Ridge_train_score = rr.score(X_train, y_train)
        Ridge_test_score = rr.score(X_test, y_test)
        Ridge_train_score100 = rr100.score(X_train, y_train)
        Ridge_test_score100 = rr100.score(X_test, y_test)
        print("linear regression train score:", train_score)
        print("linear regression test score:", test_score)
        print("ridge regression train score low alpha:", Ridge_train_score)
        print("ridge regression test score low alpha:", Ridge_test_score)
        print("ridge regression train score high alpha:", Ridge_train_score100)
        print("ridge regression test score high alpha:", Ridge_test_score100)

        # plt.plot(rr.coef_, alpha=0.7, linestyle='none', marker='*', markersize=5, color='red',
        #          label=r'Ridge; $\alpha = 0.01$', zorder=7)  # zorder for ordering the markers
        # plt.plot(rr100.coef_, alpha=0.5, linestyle='none', marker='d', markersize=6, color='blue',
        #          label=r'Ridge; $\alpha = 100$')  # alpha here is for transparency
        # plt.plot(lr.coef_, alpha=0.4, linestyle='none', marker='o', markersize=7, color='green', label='Linear Regression')
        # plt.xlabel('Coefficient Index', fontsize=16)
        # plt.ylabel('Coefficient Magnitude', fontsize=16)
        # plt.legend(fontsize=13, loc=4)
        # plt.show()


if __name__ == '__main__':
    features_dir = '/ddlrepo/TCGA-LUSC/tmp/features'

    X_train = np.load(os.path.join(features_dir, 'train_extracted_features.npy'))
    X_test = np.load(os.path.join(features_dir, 'val_extracted_features.npy'))
    y_train = np.load(os.path.join(features_dir, 'train_survival_time.npy'))
    y_test = np.load(os.path.join(features_dir, 'val_survival_time.npy'))

    lasso_regression(X_train, X_test, y_train, y_test)
    ridge_regression(X_train, X_test, y_train, y_test)
