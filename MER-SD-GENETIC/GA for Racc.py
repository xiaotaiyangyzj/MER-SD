import pandas as pd
import numpy as np
import random
import json
import time

time_start=time.time()

def get_R(WK, item2knowledge, item_n): # 获得学生的R集合和WK集合
    R = [0 for i in range(item_n)]
    for i in range(item_n):
        if i not in item2knowledge:
            continue
        for j in item2knowledge[i]: # 习题i包含这个知识点j
            if (j in WK):
                R[i] = 1 # 那么这个习题就是该学生需要可能做的习题
    return R

def encode(R, item_n): # 编码,必须是从候选集合中选择
    ls = [0 for i in range(item_n)]
    for i in range(item_n):
        if (R[i] == 1) :
            r = random.random()
            if (r <= 0.5) :
                ls[i] = 1
    return ls

def fitness(user_id, chromosome, WK, item2knowledge, item_n, knowledge_n, Item_diff, User_diff, mp): # 计算某个染色体的适应度函数
# 染色体(某个个体) Q矩阵 薄弱知识点向量 习题的总数目 知识点的总数目
    num = [0 for i in range(item_n)] # 每个习题如果被选择，包含的知识点的数目

    Anre = 0 # 推荐的习题数目
    # Aadd = 0 # 平均难度差（推荐的习题难度与学生能够承受的难度）


    res = 0 # 所有习题包含的所有知识点（没有考虑知识点厚度，可能某个知识点出现了很多次）
    for i in range(item_n):
        if chromosome[i] == 1:
            if abs(Item_diff[i] - User_diff[user_id]) <= 0.1:
                flag = 0
                for j in item2knowledge[i]: # j表示习题i对应的知识点集合
                    if ((res >> j) & 1):
                        flag = 1
                        break
                if flag == 1:
                    continue
                else:
                    for j in item2knowledge[i]:
                        res = res | (1 << j)
                    Anre += 1
        if Anre == 4:
             break


    intersection_num = 0 # 知识点交集的数目
    weak_concept_num = len(WK[user_id]) + 1 # 学生的薄弱知识点的数目
    # for i in range(item_n):
    #     res = res | num[i] # 按位或，把所有的知识点取交集

    for i in WK[user_id]:
        if ((res >> i) & 1):
            intersection_num += 1

    wccr = intersection_num / weak_concept_num  # 覆盖率
    print(user_id, wccr, Anre)
    if wccr < 0.5:
        return 0
    return 1
    # print(user_id, intersection_num, weak_concept_num)
    # return intersection_num / weak_concept_num

    # for i in range(knowledge_n):
    #     if (((res >> i) & 1) == WK[i] and WK[i] == 1): # 相同且是薄弱知识点
    #         intersection_num += 1
    #     if (WK[i] == 1): # 计算分母
    #         weak_concept_num += 1

def select(user_id, population, WK, item2knowledge, item_n, knowledge_n, Y, Item_diff, User_diff, mp): # 选择
    # 计算每个染色体的适应度函数，然后采用鲁道夫保留Y个染色体
    fits = [0 for i in range(len(population))]
    for i in range(len(population)):
        fits[i] = fitness(user_id, population[i], WK, item2knowledge, item_n, knowledge_n, Item_diff, User_diff, mp) # 计算每个染色体的适应度函数值
    # print(fits)
    # new_fits = [1 for i in range(len(population))]
    # if min(fits) != max(fits): # 归一化
    #     new_fits = [(fits[i] - min(fits)) / (max(fits) - min(fits)) for i in range(len(population))]

    all_fitness = [fits[i] for i in range(len(population))]
    mydict = dict(zip(all_fitness, population)) # 字典化，排序
    new_dict = sorted(mydict.items(), key=lambda x: x[0], reverse=True)  # 按照值类型降序排列

    new_p = list(dict(new_dict).values())
    population = new_p[0: Y - 1] # 只保留前Y个  可能存在不足 Y 个value的情况

    while len(population) < Y:
        population.append(population[0])

    return population, max(fits)

def cross(user_id, Y, gamma, item_n, population, p_cross): # 交叉

    for j in range(0,int(gamma * Y)): # 额外生成 gamma * Y个染色体
        x1 = random.random()
        i = random.randint(0, len(population)-2) # 第一步:确定染色体编号
        while (x1 > p_cross): # 不符合交叉条件，重新生成
            x1 = random.random()
            i = random.randint(0, len(population)-2)
        position = random.randrange(0, item_n - 1)  # [0,exercise_n-1] 第二步:确定染色体上的基因位置

        tmp11 = population[i][:position]
        tmp22 = population[i + 1][position:]
        tmp1 = tmp11 + tmp22 # 交换
        population.append(tmp1)
    return population

def mutation(user_id, Y, gamma, item_n, population, p_mutation): # 变异

    for j in range(0,int(gamma*Y)):
        i = random.randint(0, len(population)-1)# 第一步:确定染色体编号
        tmp = population[i]
        x2 = random.random()
        if (x2 <= p_mutation):
            position = random.randrange(0, item_n)  # [0,exercise_n-1]第二步:确定染色体上的基因变异的位置
            if tmp[position] == 0:
                tmp[position] = 1
            else:
                tmp[position] = 0
        population.append(tmp) # 大概率还是和原先的染色体相同，因为变异概率较低
    return population

def get_weak_concepts(path, epsilon = 0.5):
    data = pd.read_csv(path) # 学生对知识点的掌握矩阵Q
    knowledge_n = data.shape[1] # 列数，知识点的个数
    user_n = data.shape[0] # 行数，学生的个数

    item = pd.read_csv("case_study/item.csv")
    item_n = max(item['item_id']) # item.shape[0] # 习题的个数(1-3162)
    item2knowledge = {} # 习题对应的知识点
    for i, s in item.iterrows(): # 获取每个习题所对应的知识点列表
        item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
        item2knowledge[item_id - 1] = knowledge_codes # item_id习题编号从0开始

    knowledge = [data[str(i)] for i in range(1, knowledge_n + 1)] # 每个学生的认知状态
    weak_concepts = [[] for i in range(user_n)] # 每个学生的薄弱知识点

    for i in range(knowledge_n):  # i 表示知识点编号
        for j in range(len(knowledge[i])): # j 表示学生id
            if knowledge[i][j] <= epsilon:
                weak_concepts[j].append(i) # i+1使得知识点下标从1开始, 如果是i则会与item.csv对应 # 暂时用不到 j+1则使得学生的下标从1开始
    # print(item2knowledge)
    return weak_concepts, item2knowledge, user_n, knowledge_n, item_n

def get_recommend_item(weak_concept, item2knowledge, chromosome):
    num = 0 # 染色体包含的有效习题的数目
    concept = weak_concept # 某个学生的薄弱知识点编号
    for i in range(len(chromosome)):
        item = chromosome[i]
        if (item == 1):
            flag = 0
            for knowledge in item2knowledge[i]:
                if knowledge in concept:
                    flag = 1
                    concept.remove(knowledge)
            if flag == 1:
                num += 1
    # a = input()
    return num
# def print(): # 生成推荐习题列表并输出

def get_difficulty(path):
    f = open(path, 'r', encoding='utf-8')
    data = json.load(f)
    Item_user = dict()  # 对每一道习题(key键),做过的学生的集合(一个学生不会出现超过1次的情况)
    Item_user_error = dict()  # 对每一道习题，第一次做错的学生的集合
    Item_diff = dict()  # 每个习题的难度 = SF/S
    User_ok = dict()  # 对于每个学生，第一次做对的习题的集合
    User_diff = dict()  # 每个学生可以承受的平均习题难度
    user_n = 0
    Item = set()
    mp = dict()

    for key, values in data.items():
        ls = eval(values)  #  ls = eval(values) # math的txt格式与Ass的不一样，一个是二维列表 另一个是字符串
        for j in range(len(ls)):
            item = ls[j][0]  # 习题编号
            Item.add(item)
    Item = list(sorted(Item))
    for j in Item:
        mp[j] = Item.index(j)

    for key, values in data.items():
        ls = eval(values) # eval(values)
        User_ok[user_n] = set()
        for j in range(len(ls)):
            item = mp[ls[j][0]]  # 习题编号
            ok = ls[j][1]  # 是否做对

            if (item not in Item_user):  # 初始化
                Item_user[item] = set()
            if (item not in Item_user_error):
                Item_user_error[item] = set()

            Item_user[item].add(user_n)  # 新增
            if (ok == 0):  # item这个题目，user_n这个学生做错了！
                Item_user_error[item].add(user_n)
            if (ok == 1):
                User_ok[user_n].add(item)
        user_n += 1

    for item in Item_user.keys():  # 遍历所有的键：习题编号
        S = len(Item_user[item])  # 做过item题的总人数
        SF = len(Item_user_error[item])  # 第一次做错的人数
        Item_diff[item] = SF / S
        # print(item, Item_diff[item])

    for user in User_ok.keys():
        User_diff[user] = 0
        for item in User_ok[user]:
            User_diff[user] += Item_diff[item]
        if len(User_ok[user]) > 0:
            User_diff[user] /= len(User_ok[user])
    return Item_diff, User_diff, mp

def main(T = 10, Y = 200, gamma = 0.15, p_cross = 0.6, p_mutation = 0.01): # 遗传算法主函数 调用一系列其他函数

    path = 'case_study/case_study.csv' # 认知状态
    path2 = 'case_study/case_study.txt'
    WK, item2knowledge, user_n, knowledge_n, item_n = get_weak_concepts(path, epsilon=0.5) # 某个学生的薄弱知识点，由NCDM产生
    Item_diff, User_diff, mp = get_difficulty(path2)  # 习题难度，学生可以承受的平均难度，raw难度-->new离散化后的难度
    q = [] # 覆盖质量
    num = [] # 推荐题目的数量
    print('item_n == ', item_n, 'knowledge_n == ', knowledge_n, 'user_n == ', user_n)
    for user_id in range(user_n):
        all = []
        all_len = pow(10, 3)
        R = get_R(WK[user_id], item2knowledge, item_n)  # 当前第user_id学生的候选集合，由WK产生

        for i in range(all_len):
            all.append(encode(R, item_n))  # 生成all_len个随机01串,每个01串的长度为N，即判断N个习题是否选择

        population = random.sample(all, Y)  # 从all中随机选择M个01串

        fit = -1
        for i in range(T):
        # 进行交叉操作
            population = cross(user_id, Y, gamma, item_n, population, p_cross)
        # 进行变异操作
            population = mutation(user_id, Y, gamma, item_n, population, p_mutation)
        # 进行选择操作
            population, fit = select(user_id, population, WK, item2knowledge, item_n, knowledge_n, Y, Item_diff, User_diff, mp)
            # q[user_id] = max(fit, q[user_id])
        q.append(fit)
        num.append((float)(get_recommend_item(WK[user_id], item2knowledge, population[0])))
        print('第', user_id, '学生的精确度为:', fit)
        print('截至目前',user_id,'个学生的平均精确度为', np.average(q))
        print('第', user_id, '学生的推荐题目数量为:', np.average(num))
        # num[user_id] = get_recommend_item(population[0])
        # print('第',user_id,'个学生的推荐题量为:',get_recommend_item(population[0]))
        # print('第',user_id,'个学生的覆盖质量为:',q[user_id])
        # a = input()
    print('所有学生的平均精确度为:', np.average(q))
    print('所有学生的平均推荐题目数量为:', np.average(num))

if __name__ == '__main__':
    # get_weak_concepts("ASSISTments2017.csv", 101)
    main()

time_end=time.time()
time_sum=time_end-time_start
print(time_sum)

f=open("..\data\Running_time.txt","a+")

f.write('ASSISTment2017 running time in MER-SD-GENETIC is'+' '+str(time_sum)+'s'+' '+"\n")

f.close
