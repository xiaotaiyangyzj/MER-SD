import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
from scipy import spatial

pkm = pd.read_csv("data\frcsub\pkm_frcsub.csv" ,header = None)
pkc = pd.read_csv("data\frcsub\pkc_Frcsub.csv", index_col=0)
#读取q矩阵
EB = pd.read_csv('data\frcsub\Frcsub_q.csv' ,index_col=0)

#定义余弦相似度函数
def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos

def dei(pkmi , ei ):
    global Re
    Re = 1
    for i in range(len(ei)):
        if ei[i] is 1 :
            Re = pkmi * Re
    return 1 - Re

delta = 0.7

global all_student_Tei_list
all_student_Tei_list = []
for student in range(len(pkc)): #对每个学生的 习题过滤
    pkci = pkc.iloc[student]
    pkmi = pkm.iloc[student]
    student_Tei_list = []
    for i in range(len(EB)):    #针对第student个学生 进行 习题过滤  从第一道习题开始
        ei = EB.iloc[i]
        sim = cos_sim(ei,pkci)
        de = dei(pkmi , ei)
        dis = delta - de 
        Tei = math.sqrt(sim**2 + dis**2)
        student_Tei_list.append(Tei)
    all_student_Tei_list.append(student_Tei_list)
print(np.array(all_student_Tei_list).shape)

df = pd.DataFrame(all_student_Tei_list)

df.to_csv("data\frcsub\all_student_Tei_list1.csv")

tmp0 =  tmp.iloc[0]

def get_n_max(Te , TopN):
    global ES
    ES = []
    for i in range(len(Te)):
        Tei = Te.iloc[i]
        c = Tei.sort_values(ascending=False).index[:TopN]
        c = list(c)
        ES.append(c)
    ES = pd.DataFrame(ES)
    return ES

TopN = 300
ES = get_n_max(df , TopN)
ES = ES+1



