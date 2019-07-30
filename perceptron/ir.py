from numpy import *
import numpy as np
def Caculate_IR(retrieved,relevant):
    recall = []
    precision = []
    f1score = []
    interprecision = []
    for i in range(10):
        precision.append(100*relevant[i]/retrieved[i])
        recall.append(100*relevant[i] / 200)
        f1score.append(2*precision[i]*recall[i]/(precision[i]+recall[i]))
    interprecision.append(precision[9])
    for i in range(9, 0, -1):
        temp = max(precision[i-1],precision[i])
        interprecision.append(temp)
    interprecision.reverse()
    return recall,precision, f1score, interprecision
if __name__ == '__main__':
    retrieved = [10,20,30,40,50,60,70,80,90,100]
    relevant =[7,14,20,27,30,35,37,40,40,40]
    recall, precision, f1score, interprecision = Caculate_IR(retrieved, relevant)
    print('precision: ', precision, '    recall: ', recall, '    f1score: ', f1score,    'interprecision: ', interprecision)




