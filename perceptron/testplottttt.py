import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

def loadDataSet(filename):
    a=np.load(filename)
    dataset=[]
    labels=[]
    for i in a:
        x,y=i[5],i[7]
        dataset.append([x,y])
        labels.append(i[14])
    return np.array(dataset), np.array(labels)

def make_meshgrid(x, y, h=.02):
    x_min, x_max = 0,1
    y_min, y_max = 0,1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def main():
    data = 'nomalized_samples.npy'
    feature, lable = loadDataSet(data)
    dataArr_test = feature[:3000]
    labelArr_test = lable[:3000]
    x1_samples = []
    x2_samples = []

    for i in range(3000):
        q = dataArr_test[i]

        if labelArr_test[i] == 1:
            x1_samples.append(q)
        else:
            x2_samples.append(q)
    x1 = np.array(x1_samples)
    x2 = np.array(x2_samples)
    # import some data to play with

    # Take the first two features. We could avoid this by using a two-dim dataset
    X = np.concatenate((x1, x2), axis=0)
    y = np.array(labelArr_test)

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 150.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='poly', degree=3, C=C))
    models = (clf.fit(X, y) for clf in models)

    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        # 画出预测结果
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        # 把原始点画上去
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
    plt.show()

if __name__=='__main__':
    main()
