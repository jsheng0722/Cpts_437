import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines
from mpl_toolkits.mplot3d import Axes3D
from sklearn import neighbors, datasets, tree

bc = datasets.load_breast_cancer()
X = bc.data[:, :3]
y = bc.target

# Decision tree
clf1 = tree.DecisionTreeClassifier(max_depth = 4)
clf1.fit(X,y)

# K nearest neighbors with 5 neighbors
clf2 = neighbors.KNeighborsClassifier(n_neighbors=5)
clf2.fit(X,y)
# Majority classifier

# Own implementation of a KNN classifier

# you need to define the min and max values from the data
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

step_size = 0.05
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size),
                         np.arange(z_min, z_max, step_size))

# the colors of the plot (parameter c)
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
# cmap_mid = ListedColormap(['#FF5555', '#55FF55', '#5555FF'])

# should represent the predicted class value
Z = clf2.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

Z = Z.reshape(xx.shape)

# we found this linewidth to work well

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
c_pred = y
ax.scatter(xx, yy, zz, c=c_pred, marker='s', edgecolors='k', linewidth=0.2)
ax.xlim(xx.min(), yy.max())
ax.ylim(yy.min(), yy.max())
ax.zlim(zz.min(), zz.max())

# you will want to enhance the plot with a legend and axes titles
plt.title("Binary classification (k = %i" % (n_neighbors))
plt.show()