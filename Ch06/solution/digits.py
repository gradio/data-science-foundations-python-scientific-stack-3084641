# %%
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits['data'], digits['target']
# %%
import matplotlib.pyplot as plt

i = 353
print(y[i])
img = digits['images'][i]
plt.imshow(img, cmap='gray')

# %%
img.shape

# %%
X.shape

# %%

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

pipe = Pipeline([
    ('pca', PCA(n_components=10)),
    ('clf', KNeighborsClassifier()),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

# %%
import pickle

kb = 2**10

data = pickle.dumps(pipe)
len(data)/kb