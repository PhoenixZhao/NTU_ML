#coding=utf8
'''
    ML作业2，解题要点:
        1,train数据一共100个，所以每一轮会算出100个u_t，然后根据不同的threshold和s计算Ein(u_t)，选择Ein最小时的(s,theta)；
        2,需要遍历不同的feature，然后求出最小的(s, theta, i)
        3,选择阈值theta的思路就是从负无穷到两个特征值的中点；这也就是positive negative rays，O(N)的过程
'''
import math

max_iterations = 300
train_data_file = '../assignment2/hw2_adaboost_train.dat'
train_data = []
NEG_INF = -10000
U = []#记录每一次迭代中的u_t list iter_num * N;

def load_data():
    '''
        读入训练集，训练集格式为:
            feature1    feature2  label
    '''
    global train_data
    lines = open(train_data_file, 'r').readlines()
    train_data = [(float(f1), float(f2), int(label)) for f1, f2, label in [l.strip().split() for l in lines]]

def generate_thetas(feature_index):
    '''
        指定选择的特征，根据训练数据中该维特征生成对应的theta list
         −∞ ,(x_n,i + x_n+1,i) / 2
         先得按照对应维度的值进行升序排列
    '''
    global train_data
    theta_values = [NEG_INF]
    sorted(train_data, key=lambda d:d[feature_index], reverse=False)
    for i in range(len(train_data) - 1):
        theta_values.append((train_data[i][feature_index] + train_data[i+1][feature_index]) * 0.5)
    return theta_values

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def is_lt(a, b):
    if (a-b) < 0:
        return True
    return False

def is_unequal(a, b):
    return 0 if a == b else 1

def cal_Ein(u_t, error_num):
    N = len(u_t)
    sum_v = 0.0
    for u, err in zip(u_t, error_num):
        sum_v += u * err
    return sum_v / N

def cal_Eps(u_t, min_err_num):
    '''
        计算每一轮的epsilon:ε
    '''
    sum_v = 0.0
    for u, err in zip(u_t, min_err_num):
        sum_v += u * err
    return sum_v / sum(u_t)

def update_ut(Eps, delta_t, u_t, error_num):
    for ind, err in enumerate(error_num):
        #err只有0, 1两种取值, 所以可以用它来表示index
        u_t[ind] = u_t[ind] * delta_t[err]

def adaboost():
    '''
        adaboost的迭代算法,要求迭代次数不小于300
    '''
    u_t = [0.01] * 100
    #每一轮都会生成一个decision_stump，四个参数:f_i, s, theta, alpha
    decision_stumps = []
    iter_errs = []
    for t in range(max_iterations):
        min_err, min_err_num = 10.0, []
        print '%d iterations...' % (t+1)
        #decision stump的三个参数
        min_feature_index, min_s, min_theta = 0, 1, NEG_INF
        for f_i in range(2):
        #两维特征
            for s in [-1, 1]:
            #s 取两个值
                theta_values = generate_thetas(f_i)
                for theta in theta_values:
                #theta的取值范围，根据题目要求生成对应的list, 然后求出最小的Ein
                    error_num = [0] * 100
                    for ind, row in enumerate(train_data):
                        y = s * sign(row[f_i] - theta)
                        error_num[ind] = is_unequal(y, row[2])#训练集中第三列为label
                    Ein = cal_Ein(u_t, error_num)
                    if is_lt(Ein, min_err):
                        #import pdb;pdb.set_trace()
                        min_err = Ein
                        min_feature_index = f_i
                        min_s = s
                        min_theta = theta
                        min_err_num = list(error_num) #要求值传递
        #import pdb;pdb.set_trace()
        Eps = cal_Eps(u_t, min_err_num)
        if not is_unequal(Eps, 1.0) or not is_unequal(Eps, 0.0):
            import pdb;pdb.set_trace()
            print 'Eps equals %s, terminate' % str(Eps)
            return

        delta_t = (math.sqrt(Eps/(1-Eps)), math.sqrt((1-Eps)/Eps))
        print 'Eps=%s, delta_t=%s,  u_t=%s\n' % (str(Eps), str(delta_t), str(u_t))

        update_ut(Eps, delta_t, u_t, min_err_num)
        alpha = math.log(delta_t[1])
        decision_stumps.append((min_feature_index, min_s, theta, alpha))
        iter_errs.append(min_err)
    print 'Ein of all iterations: %s' % (str(iter_errs))
    print 'decision_stumps is %s' % ('-'.join([str(r) for r in decision_stumps]))
    #return decision_stumps

if __name__ == '__main__':
    from datetime import datetime
    print '******************%s********************' % datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    load_data()
    print train_data
    adaboost()
    print '********************************'
