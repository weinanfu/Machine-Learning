import csv
import math
import random
import matplotlib.ticker as ticker

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import explained_variance_score


class Data_Proccess:
    def __init__(self, mode):
        self.mode = mode
        self.heads = []

    def read_data(self, path):
        data = []
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i in reader:
                if '"?"' not in i:
                    data.append(i)
        return data

    def preprocess_data(self, data):
        l, ll = len(data), len(data[0])
        heads = data[0]
        categories = {}
        nomalize = {}
        for i in range(l):
            if i == 1:
                for j in range(ll):
                    if self.isnumber(data[i][j]):
                        data[i][j] = float(data[i][j])
                        nomalize[j] = data[i][j]
                    else:
                        if j in categories:
                            categories[j].add(data[i][j])
                        else:
                            categories[j] = {data[i][j]}
            if i > 1:
                for j in range(len(data[i])):
                    if j in categories:
                        if data[i][j] not in categories[j]:
                            categories[j].add(data[i][j])
                    else:
                        data[i][j] = float(data[i][j])
                        if j in nomalize:
                            nomalize[j] = max(data[i][j], nomalize[j])
                        else:
                            nomalize[j] = data[i][j]
        self.heads = heads
        return heads, categories, data, nomalize

    def isnumber(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def quantize(self, categories, label):
        dic = {}
        for i in categories:
            if i != label:
                l = len(categories[i])
                for j, k in enumerate(categories[i], start=1):
                    dic[k] = j / l
            else:
                q = -1
                for j, k in enumerate(categories[i], start=1):
                    dic[k] = pow(q, j)
        return dic

    def deal_data(self, dic, data, categories):
        l, ll = len(data), len(data[0])
        for i in range(1, l):
            for j in categories:
                data[i][j] = dic[data[i][j]]
        return np.array(data[1:])

    def nomalize(self, nomalize, data):
        l, ll = len(data), len(data[0])
        for i in range(l):
            for j in nomalize:
                data[i][j] = data[i][j] / float(nomalize[j])

    def cross_validation(self, data, number):
        div = len(data) // number
        np.random.shuffle(data)
        test_data = data[: div]
        training_data = data[div:]
        return test_data, training_data

    def find_negative(self, trainset):
        po, ne = [], []
        for i in trainset:
            if i[-1] == 1.0:
                po.append(i)
            else:
                ne.append(i)
        return np.array(po), np.array(ne)

    def information_gain(self, col, data):
        categories = {}
        for i in data[1:]:
            q = i[col] // 0.1
            if q in categories:
                if i[-1] in categories[q]:
                    categories[q][i[-1]] += 1
                else:
                    categories[q][i[-1]] = 1
            else:
                categories[q] = {i[-1]: 1}
        ecol = self.entropy(categories, len(data) - 1)
        return ecol
        #self.plot(col, categories)

    def entropy(self, dic, s):
        e = 0
        for i in dic:
            se, su = self.e(dic[i])
            e += se * su / s
        return e

    def e(self, dic):
        # dic: -1:x, 1:y
        s = sum(dic.values())
        e = 0
        for i in dic:
            e += self.one_entropy(dic[i], s)
        return e, s

    def plot(self, col, cat):
        head = self.heads[col]
        x = list(cat.keys())
        if type(x[0]) is float:
            x.sort()
        po = []
        na = []
        for i in x:
            if '"<=50K"' in cat[i]:
                na.append(cat[i]['"<=50K"'])
            else:
                na.append(0)
            if '">50K"' in cat[i]:
                po.append(cat[i]['">50K"'])
            else:
                po.append(0)
        width = 0.5
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ind = np.arange(1, len(x)+1)
        if type(x[0]) is float:
            plt.plot([i - width / 2 for i in x], po, color='blue', linestyle='-')
            plt.plot([i + width / 2 for i in x], na, color='orange', linestyle=':')
        else:
            plt.bar([i - width / 2 for i in ind], po, facecolor='red', width=width, edgecolor='white', label='>50', lw=1)
            plt.bar([i + width / 2 for i in ind], na, alpha=1, width=width, facecolor='yellowgreen', edgecolor='white',
                    label='<=50', lw=1)
            plt.xticks(ind, x)
            ax.set_xticklabels(x, rotation=50)
        plt.xlabel(head)
        plt.ylabel('amount')
        plt.legend(loc='upper right')
        plt.show()
        fig.savefig(f'{head}.png')

    def one_entropy(self, a, sums):
        return -a / sums * math.log(a / sums, 2)

    def ent(self, x):
        dic = {}
        for i in x:
            if i in dic:
                dic[i] += 1
            else:
                dic[i] = 1
        sums = sum(dic.values())
        e = 0
        for i in dic:
            e += self.one_entropy(dic[i], sums)
        return e

    def visualize(self, positive, negative, q, data, label, dual, _gamma):

        plt.xlabel('X1')  # 横坐标

        plt.ylabel('X2')  # 纵坐标

        plt.scatter(positive[:, 0], positive[:, 1], c='b', marker='+', alpha=0.5)  # +1样本红色标出

        plt.scatter(negative[:, 0], negative[:, 1], c='g', marker='o', alpha=0.5)  # -1样本绿色标出
        nonZeroAlpha = dual[0]

        supportVector = data  # 支持向量

        y = label[q]  # 支持向量对应的标签

        plt.scatter(supportVector[:, 0], supportVector[:, 1], s=80, c='y', alpha=0.5, marker='o')  # 标出支持向量

        print("支持向量个数:", len(q))

        X1 = np.arange(0, 1, 0.05)

        X2 = np.arange(0, 1, 0.05)

        x1, x2 = np.meshgrid(X1, X2)

        g = _gamma

        for i in range(len(q)):
            # g+=nonZeroAlpha[i]*y[i]*(x1*supportVector[i][0]+x2*supportVector[i][1])

            g += nonZeroAlpha[i] * y[i] * np.exp(
                -0.5 * ((x1 - supportVector[i][0]) ** 2 + (x2 - supportVector[i][1]) ** 2) / (g ** 2))

        plt.contour(x1, x2, g, 0, cmap=plt.cm.coolwarm)  # 画出超平面

        plt.title("decision boundary")

        plt.show()


if __name__ == '__main__':
    s = Data_Proccess('SVM')
    data = s.read_data('adult.csv')
    heads, categories, data, nomalize = s.preprocess_data(data)
    en = []

    lo = len(data[0]) - 1
    dic = s.quantize(categories, lo)
    data = s.deal_data(dic, data, categories)
    s.nomalize(nomalize, data)
    print('Nomalized')
    print('Sample number: ', len(data))
    basic_e = s.ent(data[:, -1])
    print('Basic entropy is: ', basic_e)
    pass
    for i in range(len(heads) - 1):
        en.append((i, heads[i], s.information_gain(i, data)))
    en.sort(key=lambda x: x[2])
    print('Information gain:')
    for i in en:
        print(i)
    e = np.array(en)


    name_list = np.array(e[:,1])
    num_list = np.array(e[:,2])

    test_data, train_data = s.cross_validation(data, 10)

    print('cross validation')

    #np.save('tests', test_data)
    #np.save('trains', train_data)

    trainset = train_data[:, :-1]
    label = train_data[:, -1]

    test_set = test_data[:, :-1]
    real_label = test_data[:, -1]
    print('SVM working:')
    #SVMClassifier = SMO(trainset, label, 1, 0.001, 40)
    #SVMClassifier.visualize(po, ne)

    pp = []
    nn = []
    for i in range(len(label)):
        if label[i] == 1:
            pp.append(trainset[i])
        else:
            nn.append(trainset[i])
    pp = np.array(pp)
    nn = np.array(nn)

    #clf = svm.SVC(kernel='poly', degree=3, C=150)
    #clf = svm.SVC(kernel='linear')
    clf = svm.SVC(kernel='rbf',gamma= 0.7, C =150)
    clf.fit(trainset, label)
    ans = clf.predict(test_set)
    #plt.scatter(pp[:, 0], pp[:, 1], marker='1', c='g')
    #plt.scatter(nn[:, 0], nn[:, 1], marker='4', c='k')
    count,tp,tn,fp, fn= 0,0,0,0,0

    for i in range(len(ans)):
        if ans[i] == real_label[i] and real_label[i] == 1:
            tp += 1
            count += 1

        if ans[i] == real_label[i] and real_label[i] == -1:
            tn += 1
            count += 1

        if ans[i] != real_label[i] and real_label[i] == -1:
            fp += 1

        if ans[i] != real_label[i] and real_label[i] == 1:
            fn += 1

    sup = clf.support_
    dual = clf.dual_coef_
    s.visualize(pp, nn, sup, clf.support_vectors_, label, dual, clf._gamma)
    tru  = np.array(real_label)
    pre = np.array(ans)
    print(count)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1score = 2*precision*recall/(precision+recall)
    print('precision: ',precision,'    recall: ',recall,'    f1score: ',f1score)
    V = explained_variance_score(tru, pre)
    print('variance:  ',V)
    #plt.scatter(sup[:, 0], sup[:, 1], c='b', marker='>')
    #plt.show()
    print('The training accuracy:', count / len(ans))



