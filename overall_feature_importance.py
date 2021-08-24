"""
reference: https://towardsdatascience.com/the-5-feature-selection-algorithms-every-data-scientist-need-to-know-3a6b566efd2


Filter based: We specify some metric and based on that filter features.
     An example of such a metric could be correlation/chi-square.
Wrapper-based: Wrapper methods consider the selection of a set of features as a search problem.
     Example: Recursive Feature Elimination
Embedded: Embedded methods use algorithms that have built-in feature selection methods.
    For instance, Lasso and RF have their own feature selection methods.
"""
import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
#from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

num_feats = 10
survey_df = pd.read_csv('C:/Users/AnanyaStitipragyan/mergedData_fwb_ces.csv')# load the dataset
X = survey_df[survey_df.columns.difference(['ENDSMEET', 'index'])]
print("type of X",type(X))
print("shape of X",X.shape)
y = survey_df['ENDSMEET']

feature_name = X.columns.tolist()

""" Pearson Correlation filter-based"""
def cor_selector(X, y,num_feats):
    cor_list = []
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature
cor_support, cor_feature = cor_selector(X, y,num_feats)
#print(str(len(cor_feature)), 'selected features')

"""chi-squared filter-based"""
X_norm = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k=num_feats)
chi_fit = chi_selector.fit(X_norm, y)
chi_dfscores = pd.DataFrame(chi_fit.scores_)
chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns.tolist()
#print(str(len(chi_feature)), 'selected features')

"""Univariate f_classif filter based"""
uv_selector = SelectKBest(score_func=f_classif, k=10)
f_classif_fit = uv_selector.fit(X,y)
f_classif_dfscores = pd.DataFrame(f_classif_fit.scores_)
uv_support = uv_selector.get_support()
uv_feature = X.loc[:,uv_support].columns.tolist()

"""Recursive Feature Elimination - wrapper based"""
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
rfe_fit = rfe_selector.fit(X_norm, y)
rfe_scores = rfe_selector.score(X, y)
#rfe_dfscores = pd.DataFrame(rfe_fit.scores_)
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
#print(str(len(rfe_feature)), 'selected features')

""" Tree-based: SelectFromModel - Random Forest - Embedded method"""
embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
rf_fit = embeded_rf_selector.fit(X, y)
#rf_dfscores = pd.DataFrame(rf_fit.scores_)
embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
#print(str(len(embeded_rf_feature)), 'selected features')

"""Lasso: embedded method"""
embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1", solver='liblinear'), max_features=num_feats)
lr_fit = embeded_lr_selector.fit(X_norm, y)
#lr_dfscores = pd.DataFrame(lr_fit.scores_)
embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
#print(str(len(embeded_lr_feature)), 'selected features')


"""put all selection together"""
feature_selection_df = pd.DataFrame({'Feature':feature_name,'F_classif':uv_support, 'Pearson':cor_support, 'Chi-2':chi_support,
                                     'RFE':rfe_support,
                                    'Random Forest':embeded_rf_support,'Lasso':embeded_lr_support})
# feature_selection_score_df = pd.DataFrame({'Feature':feature_name,'F_classif':f_classif_dfscores, 'Chi-2':chi_dfscores,
                                     #'RFE':rfe_dfscores,
                                    #'Random Forest':rf_dfscores,'Lasso':lr_dfscores
                                        # })

# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# feature_selection_score_df['Total_Score'] = np.sum(feature_selection_score_df, axis=1)
# merged['Total'] = np.sum(merged, axis=1)
# display the top 10 features
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'], ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)

# feature_selection_score_df = feature_selection_score_df.sort_values(['Total_Score','Feature'] , ascending=False)
# feature_selection_score_df.index = range(1, len(feature_selection_score_df)+1)
# merged = merged.sort_values(['Total','Feature'], ascending=False)
# merged.index = range(1, len(merged)+1)

print(feature_selection_df.head(num_feats))
# print(feature_selection_score_df.head(num_feats))
# print(merged.head(num_feats))
