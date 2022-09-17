import pandas as pd
import random
import functools
import copy
import math
import numpy as np

#Download learner answer record
answer_reword=pd.read_csv("Frcsub\\Frcsub_answer_reword.csv",header=None)

#Download learner cognitive status
cognitive_state=pd.read_csv("Frcsub\\Frcsub_cognitive_state.csv",header=None)

#Download the relation matrix between exercises and knowledge points
exercise_knowledge=pd.read_csv("Frcsub\\Frcsub_q.csv",header=None)

#Obtain the difficulty that learners bear
bear_diff=pd.read_csv("Frcsub\\bear_diff_frcsub.csv")

learner_bear={}
for i in range(len(bear_diff['bear_diff'])):
    learner_bear[i+1]=bear_diff['bear_diff'][i]

exercise_diff={} #Calculate the difficulty of each exercise
exercise_flag={} #Count the number of times each question is done and the number of wrong answers
exercise_number=len(set(answer_reword[1])) #Number of exercises
for i in set(answer_reword[1]):
    all_number=0
    wrong_number=0
    for j in range(len((answer_reword[1]))):
        if answer_reword[1][j]==i:            
            all_number+=1
            if answer_reword[2][j]==0:
                wrong_number+=1
    exercise_flag[i]=[all_number,wrong_number]
    exercise_diff[i]=round(wrong_number/all_number,4)

learner_number=len(bear_diff) #Number of learners
weak_knowledge={} #Acquiring learners' weak knowledge concepts
for i in range(learner_number):
    each_weak_knowledge=[]
    for j in cognitive_state.iloc[i][cognitive_state.iloc[i].values<=0.5].index:
        each_weak_knowledge.append(j+1)
    weak_knowledge[i+1]=each_weak_knowledge

Exercise_knowledge={} #Obtain the knowledge points contained in each exercise
for i in range(len(exercise_knowledge)):
    each_Exercise_knowledge=[]
    for j in exercise_knowledge.iloc[i][exercise_knowledge.iloc[i].values==1].index:
        each_Exercise_knowledge.append(j+1)
    Exercise_knowledge[i+1]=each_Exercise_knowledge

def compare(i,j):
    if learner_bear[i]<learner_bear[j]:
        return -1 # It means that in this case, row a is on the left of row B. in fact, it only needs to return negative value
    elif learner_bear[i]>learner_bear[j]:
        return 1 # It means that in this case, a is on the right of B. in fact, it only needs to return positive value
    else:
        return 0 # Indicates equality, and is arranged in the order of circular access

def recommend(ex_k,wk,le_b):#Relation of exercise knowledge points,Weak knowledge points of learner i,The bearing difficulty of learner i.
    can_ex=set() #Candidate exercise set
    while wk:
        e_cov=set() #Store weak knowledge points contained in the selected exercises
        d=2 #Assume the initial exercise difficulty is 2
        for exercise,knowledge in ex_k.items():
            cov=set(wk) & set(knowledge) #Number of weak knowledge points covered by exercises
            d_e=exercise_diff[exercise]   #Difficulty of exercises
            if d_e-le_b==0:
                if len(cov)!=0:
                    can_e=exercise #Candidate exercises
                    e_cov=cov
            else:
                if len(cov)/abs(d_e-le_b)>len(e_cov)/abs(d-le_b):
                    can_e=exercise
                    e_cov=cov
                    d=d_e
        can_ex.add(can_e)
        wk-=e_cov
        if (can_e in ex_k):
            del ex_k[can_e]
    return can_ex

recommend_candidate={} # Recommended candidate exercise set
for i in weak_knowledge:
    if len(weak_knowledge[i])==0:
        recommend_candidate[i]=[]
    else:
        Exercise_knowledge_copy=copy.deepcopy(Exercise_knowledge)
        recommend_candidate[i]=list(recommend(Exercise_knowledge_copy,set(weak_knowledge[i]),learner_bear[i]))
print(recommend_candidate)

    return cover_rate

def cover_quality(ex_list,wk): #Incoming exercise list and weak knowledge points of learners
    num=0
    length=len(wk)
    if len(ex_list):
        for i in ex_list:
            cover=set(wk) & set(Exercise_knowledge[i]) #Weak knowledge points covered by exercises
            num+=len(cover) # Number of weak knowledge points covered by exercises
            wk-=cover
        cover_rate=num/length
    else:
        cover_rate=0
    return cover_rate

def quality(ex,wk): #Incoming exercise and weak knowledge points of learners
    if isinstance(wk,list):
        cover=set(wk) & set(Exercise_knowledge[ex]) #Weak knowledge points covered by exercises
        num=len(cover) # Number of weak knowledge points covered by exercises
    else:
        if wk in Exercise_knowledge[ex]:
            num=1
        else:
            num=0
    return num

d_recommend={} #Select the set of exercises with appropriate difficulty
for learner,exercise in recommend_candidate.items():
    if len(exercise):
        for i in exercise:
            if math.fabs(exercise_diff[i]-learner_bear[learner])>0.1:
                exercise.remove(i)
        if len(exercise):
            d_recommend[learner]=exercise
        else:
            d_recommend[learner]=[]
    else:
        d_recommend[learner]=[]

recommend_set={}  #Generate recommended collection
for exercise_index,exercise_list in d_recommend.items(): #Traverse the list of candidate exercises of all students
    each_exercise_list=copy.deepcopy(exercise_list)
    weak_knowledge_copy=copy.deepcopy(weak_knowledge[exercise_index])
    recommend0=[]
    while cover_quality(recommend0,set(weak_knowledge[exercise_index]))<0.7 and len(each_exercise_list) and len(weak_knowledge_copy):
        e_need=each_exercise_list[0] #Suppose the first exercise is the one I want
        e_need_cov=quality(e_need,weak_knowledge_copy) # Number of weak knowledge points covered by exercises
        for i in each_exercise_list:
            i_cov=quality(i,weak_knowledge_copy)
            if i_cov>e_need_cov:
                e_need=i
                e_need_cov=i_cov
        recommend0.append(e_need)
        each_exercise_list.remove(e_need)
        for i in Exercise_knowledge[e_need]:
            if i in weak_knowledge_copy:
                weak_knowledge_copy.remove(i)
    recommend=recommend0
    while len(each_exercise_list) and len(weak_knowledge_copy):
        e_need=each_exercise_list[0] #Suppose the first exercise is the one I want
        e_need_cov=quality(e_need,weak_knowledge_copy) # Number of weak knowledge points covered by exercises
        min_cost=cost(weak_knowledge[exercise_index],learner_bear[exercise_index],recommend,exercise_list)
        e_need=each_exercise_list[0] #Suppose the first exercise is the one I want
        e_need_cov=quality(e_need,weak_knowledge_copy) # Number of weak knowledge points covered by exercises
        for i in each_exercise_list:
            i_cov=quality(i,weak_knowledge_copy)
            if i_cov>e_need_cov:
                e_need=i
                e_need_cov=i_cov
        recommend0.append(e_need)
        each_exercise_list.remove(e_need)
        for i in Exercise_knowledge[e_need]:
            if i in weak_knowledge_copy:
                weak_knowledge_copy.remove(i)
        new_cost=cost(weak_knowledge[exercise_index],learner_bear[exercise_index],recommend0,exercise_list)
        if new_cost<min_cost:
            recommend=recommend0
            min_cost=new_cost
    recommend_set[exercise_index]=recommend
print(recommend_set)

a_d=0  #Statistical difficulty difference
l_a=0  #Count the total number of exercises
n_p=0 #Number of people without weak knowledge points
for i in recommend_set:   #Traverse recommended results
    if len(recommend_set[i])!=0:
        l_a=l_a+len(recommend_set[i])
        d_s=0
        p_d=0
        for j in recommend_set[i]:
            d_s=d_s+math.fabs(exercise_diff[j]-learner_bear[i])
        p_d=d_s/len(recommend_set[i])
        a_d=a_d+p_d
    else:
        n_p+=1
diff=a_d/(len(recommend_set)-n_p)
length=l_a/(len(recommend_set)-n_p)
print('%.4f'%diff)
print('%.4f'%length)

coverage=[] #Coverage of weak knowledge concepts
Racc6=0
Racc65=0
Racc7=0
Racc75=0
Racc8=0
Racc85=0
Racc9=0
Racc95=0
for k,v in recommend_set.items():
    cover=[]
    if len(weak_knowledge[k]):
        if len(v)!=0:
            for i in v:
                for j in Exercise_knowledge[i]:
                    if j not in cover:
                        cover.append(j)
            jj=list(set(weak_knowledge[k]).intersection(set(cover)))
            cover_rate=len(jj)/len(weak_knowledge[k])
    else:
        cover_rate=1
    coverage.append(cover_rate)
    if cover_rate>0.6:
        Racc6+=cover_rate
    if cover_rate>0.65:
        Racc65+=cover_rate
    if cover_rate>0.7:
        Racc7+=cover_rate
    if cover_rate>0.75:
        Racc75+=cover_rate
    if cover_rate>0.8:
        Racc8+=cover_rate
    if cover_rate>0.85:
        Racc85+=cover_rate
    if cover_rate>0.9:
        Racc9+=cover_rate
    if cover_rate>0.95:
        Racc95+=cover_rate
coverage_rate=np.average(coverage)
Racc6=Racc6/len(coverage)
Racc65=Racc65/len(coverage)
Racc7=Racc7/len(coverage)
Racc75=Racc75/len(coverage)
Racc8=Racc8/len(coverage)
Racc85=Racc85/len(coverage)
Racc9=Racc9/len(coverage)
Racc95=Racc95/len(coverage)
print('%.4f'%coverage_rate)
print('%.4f'%Racc6)
print('%.4f'%Racc65)
print('%.4f'%Racc7)
print('%.4f'%Racc75)
print('%.4f'%Racc8)
print('%.4f'%Racc85)
print('%.4f'%Racc9)
print('%.4f'%Racc95)