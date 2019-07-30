import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy
from sklearn import svm

def loadDataSet(filename):
    a=np.load(filename)
    dataset=[]
    labels=[]
    for i in a:
        x,y=i[2],i[7]
        dataset.append([x,y])
        labels.append(i[14])
    return np.array(dataset), np.array(labels)

def main():
    data='nomalized_samples.npy'
    feature, lable = loadDataSet(data)
    dataArr_test = feature[:1000]
    labelArr_test = lable[:1000]
    x1_samples = []
    x2_samples = []

    for i in range(900):
        q = dataArr_test[i]

        if labelArr_test[i] == 1:
            x1_samples.append(q)
        else:
            x2_samples.append(q)
    x1 = np.array(x1_samples)
    x2 = np.array(x2_samples)

    fig = plt.figure()

    plt.scatter(x1[:,0], x1[:,1], marker='+')
    plt.scatter(x2[:,0], x2[:,1], c='green', marker='o')

    X = np.concatenate((x1, x2), axis=0)
    Y = np.array(labelArr_test)

    C = 200.0  # SVM regularization parameter
    clf = svm.SVC(kernel='rbf', gamma=0.7, C=C)
    clf.fit(X, Y)

    h = .02  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = 0,1
    y_min, y_max = 0,1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

    plt.show()

if __name__=='__main__':
    main()