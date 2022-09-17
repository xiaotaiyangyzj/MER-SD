import  pandas as pd
import numpy as np
import json


#获取学生答题记录并处理
stus_record = []
with open('ASSISTment0910.txt' ,encoding='utf8') as i_f:
    stus_record = json.load(i_f)

stus_records = []
for stu_record in stus_record.values():
    stu_record = str(stu_record.replace('[', ''))
    stu_record = stu_record.replace(']', '')
    stu_record = stu_record.replace('(', '')
    stu_record = stu_record.replace(')', '')
    stu_record = stu_record.replace(' ', '')
    stu_record = stu_record.split(',')
    stus_records.append(stu_record)
print(stus_records[0])

#读取q矩阵
stus_q = pd.read_csv('ASSISTment2009-q.csv' , index_col=0)
stus_q = stus_q.values
stus_q = stus_q.tolist()
#
#计算学生的w(k)
all_wk = []
for stu_records in  stus_records:
    # 每个人答对 和答错
    q_len = len(stus_q[0])
    ci = [0.] * q_len #单个学生 所有回答的问题
    ri = [0.] * q_len #答对的总和
    for i in range(len(stu_records)):
        if (stu_records[2*i + 1]) == '1':
            ri = np.array(ri) + np.array(stus_q[i])
        ci = np.array(ci) + np.array(stus_q[i])
        # print(all_point)
        if i == (len(stu_records)/2-1) :
            break
    wk = 1- np.divide(ri, ci, out=np.zeros_like(ci), where=ci != 0)
    all_wk.append(wk)
print(len(all_wk))
all_wk = pd.DataFrame(all_wk)
# print(all_wk)
all_wk.to_csv('assist2009_wk.csv' ,index=None)

