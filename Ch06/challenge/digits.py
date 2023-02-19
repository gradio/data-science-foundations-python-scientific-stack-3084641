# %%
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X, y = digits['data'], digits['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
)
# %%

i = 353
print(y[i])
img = digits['images'][i]
plt.imshow(img, cmap='gray')

# %%
img.shape

# %%
X.shape

# %%

pipe = Pipeline([
    ('pca', PCA(n_components=10)),
    ('Knay', KNeighborsClassifier()),
])

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
)

pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

# %%
pipe.steps

# %%
kb = 2**10

data = pickle.dumps(pipe)
len(data)/kb
# %%
