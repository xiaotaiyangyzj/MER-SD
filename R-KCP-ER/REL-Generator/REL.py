import pandas as pd
import numpy as np
import random as r
import ast as a
import math
import copy

df_Es=pd.read_csv(".\\data\\frcsub\\ES_frcsub.csv")
# df_Es=df_Es.iloc[:,1:]
print(df_Es.head())
student_num=len(df_Es)  #Get the number of students
print(student_num)
df_knowledge=pd.read_csv(".\\data\\frcsub\\stu_weak_knowledge_frcsub.csv")
print(df_knowledge.head())

df_Es['exercise_code'] = df_Es['exercise_code'] .apply(a.literal_eval) 
#Exercise_code the content of code is converted to list again
df_knowledge['weak_knowledge'] = df_knowledge['weak_knowledge'].apply(a.literal_eval) 
#Re convert the content of knowledge into list

df_q_diff=pd.read_csv(".\\data\\frcsub\\exercise_diff_frcsub.csv")
print(df_q_diff.head())
df_s_state=pd.read_csv(".\\data\\frcsub\\bear_diff_frcsub.csv")
print(df_s_state.head())

q_diff=df_q_diff['exercise_diff']
s_state=df_s_state['bear_diff']

df_q=pd.read_csv(".\\data\\frcsub\\q_frcsub.csv")   #Exercise knowledge relation matrix
print(df_q.head())
print(len(df_q))

R={} #候选习题库
exer=[]
for stu in range(student_num):
    for each_exercise in df_Es['exercise_code'][stu]:
        if math.fabs(q_diff[each_exercise-1]-s_state[stu])<=0.10:
            exer.append(each_exercise)
    R[stu]=exer
    exer=[]
print(R)

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

def cost(wk,state_diff,sub_e,e): #Weak knowledge points, bearing difficulty, recommended exercise list, candidate exercise list
    not_covered_rate=candidate_rate=diff_state=0
    exercise_diffent=0    
    cover=[] #Count the knowledge points contained in the exercise list
    if len(e) and len(wk):      
        if len(sub_e):
            
            #The proportion of weak knowledge points not covered by statistics 
            for i in sub_e:
                for j in copy_qu_kn[i]:
                    if j not in cover:
                        cover.append(j)
            not_cover=list(set(wk)-set(cover))
            not_covered_rate=len(not_cover)/len(wk)
            
            #Count the difference between the difficulty of bearing and the difficulty of recommended exercises
            for i in sub_e:#Traverse the candidate exercise set
                exercise_diffent+=q_diff[i-1]
            age_exercise_diff=exercise_diffent/len(sub_e)
            diff_state=math.fabs(age_exercise_diff-state_diff)
        else:
             not_covered_rate=diff_state=0
            
        #Count the proportion of the number of recommended exercises in the number of candidate exercises
        candidate_rate=len(sub_e)/len(e)
        
    else:
        not_covered_rate=diff_state=0
        candidate_rate=1
    cost_mul=not_covered_rate*candidate_rate*diff_state
    return cost_mul

#生成初始习题列表
every_stu=[]
all_stu=[]
stu_can={}
for i in range(1,student_num+1):
    length=len(R[i-1])  #每个学生的习题列表长度
    #There are 20 exercises in the exercise bank in frcsub, so 20% is 4
    if length<=4:
        stu_can[i]=list(R[i-1])
    else:
        M=4
        min_cost=1
        can_l=[]
        for j in range(M):
            subset=r.sample(R[i-1],M)  #Randomly generate subsets with length M
            if cost(df_knowledge['weak_knowledge'][i-1],s_state[i-1],subset,R[i-1])<min_cost:
                min_cost=cost(df_knowledge['weak_knowledge'][i-1],s_state[i-1],subset,R[i-1])

                can_l=list(set(subset))
        stu_can[i]=can_l
print(stu_can)

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

def REL_generator(wk,st,eslist,canlist,T,iteration):  #传入薄弱知识点，承受难度，候选习题库，候选习题列表  温度  迭代次数
    k=1.380649e-23
    c=0.99    
    e=0.000001
    rel=can_l=canlist
    min_cost=cost(wk,st,canlist,eslist)
    while T>e:
        for i in range(iteration):
            rel_r=replace_res(eslist,canlist)
            if cost(wk,st,rel_r,eslist)<min_cost:
                min_cost=cost(wk,st,rel_r,eslist)
                can_l=list(set(rel_r))
            else:
                p=r.random()
#                 p=math.exp((-(mean_dis(rel_r)-min_cost))/(k*T))
                suiji=r.random()#随机生成一个0-1之间的数字
                if suiji>=p:
                    min_cost=cost(wk,st,rel_r,eslist)
                    can_l=list(set(rel_r))
        rel=list(set(can_l))            
        T*=c
    return rel

re={}            #用于存储对学生的推荐习题列表
for i in range(1,student_num+1): #遍历每一位学生 
    if len(R[i-1]):
        iterate=r.randint(1,2*len(R[i-1]))+1 #迭代次数
        res=REL_generator(df_knowledge['weak_knowledge'][i-1],s_state[i-1],R[i-1],list(set(stu_can[i])),r.random(),iterate) 
        re[i]=res
    else:
        res=[]
        re[i]=res
    print("给第%s号学生推荐的习题是%s"%(i,res))

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