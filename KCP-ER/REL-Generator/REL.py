import pandas as pd
import numpy as np
import random as r
import ast as a
import math
import copy

df_Es=pd.read_csv(".\\data\\frcsub\\ES_frcsub.csv")
student_num=len(df_Es)  #Get the number of students
df_knowledge=pd.read_csv(".\\data\\frcsub\\stu_weak_knowledge_frcsub.csv")

df_Es['exercise_code'] = df_Es['exercise_code'].apply(a.literal_eval) 
#Exercise_code the content of code is converted to list again
df_knowledge['weak_knowledge'] = df_knowledge['weak_knowledge'].apply(a.literal_eval) 
#Re convert the content of knowledge into list

df_q=pd.read_csv(".\\data\\frcsub\\q_frcsub.csv")   #Exercise knowledge relation matrix

#Obtain the knowledge concepts contained in each exercise
qu_kn={}     #用于存放每道习题包含的知识点
j=0
for i in range(1,len(df_q)+1):
    row=df_q.iloc[j,:]       #获取第j行的所有列
    relate=list(row[row.values==1].index)    #如果第j行的某列数据为1，表示这道题包含这个知识点
    a=[]
    for k in relate:
        a.append(df_q.columns.get_loc(k)+1)
    qu_kn[i]=a                                     #输出每道题包含的知识点
    j=j+1
print(qu_kn)
copy_qu_kn=copy.deepcopy(qu_kn)

def mean(q1,q2): 
    d_sum=0
    for i in q1:
        if i in q2:
            q2.remove(i)
        else:
            d_sum+=1
    d_sum+=len(q2)           
    return d_sum  

def mean_dis(e_l):      
    length=len(e_l)
    distance=1
    for i in range(length):
        for j in range(i+1,length):
            dist=mean(qu_kn[e_l[i]],qu_kn[e_l[j]])
            if dist>distance:
                distance=dist
    return distance

#生成初始习题列表
every_stu=[]
all_stu=[]
stu_can={}
for i in range(1,student_num+1):
    length=len(df_Es["exercise_code"][i-1])  #每个学生的习题列表长度
    M=r.randint(1,2*length) 
    max_dis=0
    can_l=[]
    for j in range(M):
        subset=r.sample(df_Es["exercise_code"][i-1],r.randint(1,length))  #随机生成随机长度的子集
        if mean_dis(subset)>max_dis:
            max_dis=mean_dis(subset)
            can_l=list(set(subset))
    stu_can[i]=can_l
print(stu_can)

#生成指定长度的随机数
def get_sub_set(nums):
    sub_sets = [[]]
    for x in nums:
        sub_sets.extend([item + [x] for item in sub_sets])
    return sub_sets
def uni_set(uni_set,length):
    sub=get_sub_set(uni_set)
    sub_set_len=[]
    for i in sub:
        if len(i)==length:
            sub_set_len.append(i)
    return sub_set_len

#用候选题库中的题随机替换候选列表中的习题
def replace_res(es_list,sub_list):  #候选习题，初始子集
    es_list_copy=copy.deepcopy(es_list)
    for i in es_list:
        if i in sub_list:
            es_list_copy.remove(i)
    if es_list_copy:
        for index, x in enumerate(sub_list):
            if r.randint(0,1):
                sub_list[index] = r.choice(es_list_copy)
                break
    return sub_list

def REL_generator(eslist,canlist,T,iteration):  #传入候选习题库，候选习题列表  温度  迭代次数
    k=1.380649e-23
    c=0.99    
    e=0.000001
    rel=can_l=canlist
    max_dis=mean_dis(canlist)
    while T>e:
        for i in range(iteration):
            rel_r=replace_res(eslist,canlist)
            if mean_dis(rel_r)>max_dis:
                max_dis=mean_dis(rel_r)
                can_l=list(set(rel_r))
            else:
                p=r.random()
#                 p=math.exp((-(mean_dis(rel_r)-max_dis))/(k*T))
                suiji=r.random()#随机生成一个0-1之间的数字
                if suiji>=p:
                    max_dis=mean_dis(rel_r)
                    can_l=list(set(rel_r))
        rel=list(set(can_l))            
        T*=c
    return rel

re={}            #用于存储对学生的推荐习题列表
for i in range(1,student_num+1): #遍历每一位学生   
    iterate=r.randint(1,2*len(df_Es["exercise_code"][i-1]))+1 #迭代次数
    res=REL_generator(df_Es["exercise_code"][i-1],list(set(stu_can[i])),r.random(),iterate) 
    re[i]=res
    print("给第%s号学生推荐的习题是%s"%(i,res))

df_q_diff=pd.read_csv(".\\data\\frcsub\\exercise_diff_frcsub.csv")
print(df_q_diff.head())
df_s_state=pd.read_csv(".\\data\\frcsub\\bear_diff_frcsub.csv")
print(df_s_state.head())

q_diff=df_q_diff['exercise_diff']
s_state=df_s_state['bear_diff']

#难度和长度
a_d=0  #统计难度差
l_a=0  #统计总习题个数
flag1={} #难度标签
flag2={} #题量标签
flag3={} #覆盖质量标签
for i in re: 
    flag1[i]=0
    flag2[i]=0
    flag3[i]=0
for i in re:   #遍历推荐结果
    if len(re[i])<=4:
        flag2[i]=1
    if len(re[i])!=0:
        l_a=l_a+len(re[i])
        d_s=0
        p_d=0
        for j in re[i]:
            d_s=d_s+math.fabs(q_diff[j-1]-s_state[i-1])
        p_d=d_s/len(re[i])
        if p_d <=0.1:
            flag1[i]=1
        a_d=a_d+p_d
diff=a_d/student_num
length=l_a/student_num
print('%.4f'%diff)
print('%.4f'%length)

coverage=[] #Coverage of weak knowledge concepts

for k,v in re.items():
    cover=[]
    if len(df_knowledge['weak_knowledge'][k-1]):
        if len(v)!=0:
            for i in v:
                for j in copy_qu_kn[i]:
                    if j not in cover:
                        cover.append(j)
            jj=list(set(df_knowledge['weak_knowledge'][k-1]).intersection(set(cover)))
            cover_rate=len(jj)/len(df_knowledge['weak_knowledge'][k-1])
    else:
        cover_rate=1
    coverage.append(cover_rate)
    if cover_rate>=0.5:
        flag3[k]=1
    
coverage_rate=np.sum(coverage)/student_num
print('%.4f'%coverage_rate)

racc=0
for i in re:
    if  flag1[i] and flag2[i] and flag3[i]:
        racc+=1
racc/=student_num
print('%.4f'%racc)