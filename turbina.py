# turbofan jet engines using classification algorithm
# imports python libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# utilizando a biblioteca pandas para importarmos o dataset
df_raw_train = pd.read_csv('data/CMaps/train_FD001.txt',
                           parse_dates=False, delimiter=" ", decimal=".", header=None)
df_raw_test = pd.read_csv('data/CMaps/test_FD001.txt',
                          parse_dates=False, delimiter=" ", decimal=".", header=None)
df_raw_results = pd.read_csv('data/CMaps/RUL_FD001.txt',
                             parse_dates=False, delimiter=" ", decimal=".", header=None)

# vamos nomear as columas
cols = [
    'unit', 'cycles', 'setting1', 'setting2', 'setting3',
    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
    's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19',
    's20', 's21', 's22', 's23']

# mudando nome das colunas
df_raw_train.columns = cols
df_raw_test.columns = cols

# fazendo uma cópia

df4 = df_raw_train.copy()

# Filtrar Colunas
df4 = df4.drop(columns=[
    'setting3', 's1', 's5', 's10', 's16', 's18', 's19', 's22', 's23'])

# Feature Engineering
# vamos criar uma coluna com variáveis os números máximos de ciclo de cada engine
max_col = pd.DataFrame(df4.groupby('unit')['cycles'].max().reset_index())
df4 = df4.merge(max_col, on=['unit'], how='left')
# RUL
df4['rul'] = df4['cycles_y'] - df4['cycles_x']
# vamos dropar as colunas com cycles_y e max
df4 = df4.drop(columns=['cycles_y'])
# Vamos criar mais duas variáveis de tipo classificatória, quando a máquina estiver a 30 dias para a falha e outra para quando a máquina estiver a 10 dias.

m = 30
m2 = 10

# coluna com 30 dias para falha
df4['30_days'] = df4['rul'].apply(lambda x: 1 if x <= m else 0)
# coluna com 10 dias para falha
df4['10_days'] = df4['rul'].apply(lambda x: 1 if x <= m2 else 0)

# copia o dataset para nova etapa
df5 = df4.copy()
# Rescala
# instanciar robust scaler
rs = RobustScaler()

# sensor 6
df5['s6'] = rs.fit_transform(df5[['s6']].values)
# sensor 9
df5['s9'] = rs.fit_transform(df5[['s9']].values)
# sensor 14
df5['s14'] = rs.fit_transform(df5[['s14']].values)


# Dividir o Dataframe em Treino e Teste
X = df5.drop('30_days', axis=1)
Y = df5['30_days']

tss = TimeSeriesSplit(n_splits=3)
for train_index, test_index in tss.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

# colunas selecionadas pelo boruta
cols_selected_boruta = ['unit',
                        'cycles_x',
                        's2',
                        's3',
                        's4',
                        's7',
                        's8',
                        's9',
                        's11',
                        's12',
                        's13',
                        's14',
                        's15',
                        's20',
                        's21',
                        '10_days']


# Modelos de Machine Learning
# boruta columns selection
x_train = X_train[cols_selected_boruta]
x_test = X_test[cols_selected_boruta]

# #for cross validation
X_train_full = X[cols_selected_boruta]
y_train_full = Y

cv = TimeSeriesSplit(n_splits=10)


# Model's performance
def ml_scores(model_name, y_test, y_pred):

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)

    return pd.DataFrame({'Accuracy': accuracy,
                         'Precision': precision,
                         'Recall': recall,
                         'F1': f1,
                         'ROC': roc},
                        index=[model_name])


# Baseline: Dummy Classifier
# Model and fit
dummy = DummyClassifier().fit(x_train, y_train)
# Predicting
y_pred = dummy.predict(x_test)
Dummy_performance = ml_scores('Baseline', y_test, y_pred)
print(Dummy_performance)

#%%
# Logistic Regression model fit
# Model and fit
lr = LogisticRegression().fit(x_train, y_train)
# Predicting
y_pred_lr = lr.predict(x_test)
performance_lr = ml_scores('Logistic Regression', y_test, y_pred_lr)
print(performance_lr)

# Random Forest Classifier model and fit
# Model and fit
rfc = RandomForestClassifier().fit(x_train, y_train)
# Predicting
y_pred_rfc = rfc.predict(x_test)
rfc_performance = ml_scores('Random Forest', y_test, y_pred_rfc)
print(rfc_performance)

# Confusion Matrix function
def conf_matrix(y_train, y_pred):
    cm = confusion_matrix(y_train, y_pred)
    cm_data = pd.DataFrame(cm, columns = ['Positive', 'Negative'], index=['Positive', 'Negative'])
    sns.heatmap(cm_data, annot=True, cmap='Blues', fmt='d', annot_kws={'size': 24}).set_title('Confusion Matrix')

    return plt.show()

conf_matrix(y_test, y_pred_lr)
plt.show()
