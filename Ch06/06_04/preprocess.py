# %%
from sklearn.decomposition import PCA
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

cal_housing = fetch_california_housing(as_frame=True)
df = cal_housing['data']
df.describe()

# %%

X, y = cal_housing['data'], cal_housing['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
)

# %%

clf = SVR()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# %%

X_scaled = preprocessing.scale(X)

# %%

df = pd.DataFrame(
    X_scaled,
    columns=cal_housing['feature_names']
)
df.describe()

# %%

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.3,
)
clf = SVR()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# %%

pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)
X_pca.shape
# %%
