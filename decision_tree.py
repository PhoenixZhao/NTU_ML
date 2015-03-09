#coding=utf8
'''
    台大ML作业3的Q13-Q20,自己实现简单的决策树
    C&RT alogrithm and Gini index
'''
import copy

all_nodes = [] #记录所有的node，根据index来保存树结构
recursive_depth = 0
INF = 10000
LEFT_LABEL = -1
RIGHT_LABEL = 1


train_data_file = '../assignment3/hw3_train.dat'
test_data_file = '../assignment3/hw3_test.dat'


DEBUG = False

class Node(object):
    '''
        决策树的每一个node，每个节点的attribute如下
        index: id
        lchild，rchild，parent: 子女和父母的index
        f_i, theta: 该node上的决策信息，使用函数f = sign(x_i - theta)
        规定: "-1"往左走，"+1"往右走
    '''

    def __init__(self, index, parent=-1):
        self.index = index
        self.lchild = -1
        self.rchild = -1
        self.parent = parent
        self.f_i = -1
        self.theta = -1
        self.label= 0

def load_data():
    '''
        读入训练集和测试集，格式为:
            feature1    feature2  label
    '''
    lines = open(train_data_file, 'r').readlines()
    lines = lines[:5] if DEBUG else lines
    train_data = [(float(f1), float(f2), int(label)) for f1, f2, label in [l.strip().split() for l in lines]]
    lines = open(test_data_file, 'r').readlines()
    test_data = [(float(f1), float(f2), int(label)) for f1, f2, label in [l.strip().split() for l in lines]]
    return train_data, test_data

def generate_thetas(feature_index, train_data):
    '''
        指定选择的特征，根据训练数据中该维特征生成对应的theta list
         −∞ ,(x_n,i + x_n+1,i) / 2, +∞
         先得按照对应维度的值进行升序排列
    '''
    theta_values = [-INF]
    train_data = sorted(train_data, key=lambda d:d[feature_index], reverse=False)
    for i in range(len(train_data) - 1):
        theta_values.append((train_data[i][feature_index] + train_data[i+1][feature_index]) * 0.5)
    theta_values.append(INF)
    return theta_values

def sign(x):
    return 1 if x>= 0 else -1

def impurity(data):
    '''
        计算impurity，使用Gini index
        data为list
        使用公式impurity = 1 - sum_k{(1-sum_N[[y==k]] / N)}^2
    '''
    if not data:
        return 0.0
    N = len(data)
    pos, neg = 0, 0

    for _, _, y in data:
        if y == 1:
            pos += 1
        else:
            neg += 1
    return 1.0 - (pos**2 + neg**2) * 1.0 / N**2

def get_opt_parameters(train_data):
    '''
        train_data格式: (feature1, feature2, label)
        寻找使得err最小的划分参数组合,theta, f_i
        min_err = size(D1) * impurity(D1) + size(D2) * impurity(D2)
        使用一个trick的方法: 将数据按照对应的维度排序，然后从左往右切分，找出最好的impurity，而不用一个theta遍历所有的数据
    '''
    min_err, opt_f_i, opt_theta = INF, 0, -INF
    N = len(train_data)
    for f_i in range(2):
        train_data = sorted(train_data, key=lambda d:d[f_i], reverse=False)
        for ind in range(N):
            err = ind * impurity(train_data[:ind]) + (N - ind) * impurity(train_data[ind:])

            if min_err > err:
                min_err = err
                opt_f_i = f_i
                opt_theta = -INF if ind == 0 else (train_data[ind - 1][f_i] + train_data[ind][f_i]) * 0.5
    return opt_f_i, opt_theta

def create_children(node):
    '''
        为当前node生成两个孩子节点，并建立好父子关系
        同时更新all_nodes记录所有的节点
    '''
    global all_nodes
    pos = len(all_nodes)
    lchild = Node(pos, node.index)
    rchild = Node(pos+1, node.index)

    node.lchild = pos
    node.rchild = pos + 1

    all_nodes.append(lchild)
    all_nodes.append(rchild)
    #print 'create children(l=%d,r=%d) for node %d' % (pos, pos+1, node.index)
    return lchild, rchild

def decision_tree(node, train_data):
    '''
        递归调用，根据train_data的impurity判断是否需要terminate;
        如果没有terminate，则选择最好的feature_i，和theta，将数据进行划分；
        -1往左走，+1往右走，创建两个新node，继续重复上述过程直到terminate
    '''

    majority_of_y = lambda d: sign(sum(d))#由大多数y来决定最后的label

    global all_nodes, recursive_depth
    recursive_depth += 1
    print 'recursive_depth=%d, node_num=%d, impurity=%.3f, current_data_num=%d' %(recursive_depth, node.index, impurity(train_data), len(train_data))
    if not train_data:
        return
    if impurity(train_data) == 0.0:
        #all_nodes.append(node)
        #说明是叶子节点，需要生成gt(x)
        node.label = majority_of_y([y for _, _, y in train_data])
        return node

    #import pdb;pdb.set_trace()
    f_i, theta = get_opt_parameters(train_data)
    node.theta = theta
    node.f_i = f_i

    lchild, rchild = create_children(node)

    #import pdb;pdb.set_trace()
    left_part = [t for t in train_data if t[f_i] < theta]
    right_part = [t for t in train_data if t[f_i] >= theta]

    decision_tree(lchild, left_part)

    decision_tree(rchild, right_part)

    return node


def predict(node, x):
    '''
        predict with decision tree recursively
    '''
    if node.label:#1 or -1
        return node.label
    else:
        if x[node.f_i] < node.theta:
            lchild = all_nodes[node.lchild]
            return predict(lchild, x)
        else:
            rchild = all_nodes[node.rchild]
            return predict(rchild, x)


def main():
    train_data, test_data = load_data()
    root = Node(0)
    global all_nodes
    all_nodes.append(root)
    print 'start build decision tree, %d train data...' % len(train_data)
    decision_tree(root, train_data)
    for r in all_nodes:
        print 'index=%d(l=%d,r=%d,p=%d), para(theta=%.4f,f_i=%d), label=%d' % (r.index, r.lchild, r.rchild, r.parent, r.theta, r.f_i, r.label)
    print 'finished, there is %d nodes in the tree' % (len(all_nodes))
    print '----------------------------------------'
    print '         Homework 3 Question 13         '
    print '----------------------------------------'
    print 'How many internal nodes:'
    print len([r for r in all_nodes if r.label == 0])
    print '----------------------------------------'

    print '----------------------------------------'
    print '         Homework 3 Question 14         '
    print '----------------------------------------'
    print 'Ein (evaluated with 0/1 error):'
    #其实从训练过程中就可以看出，ein就应该为0
    predict_Y = [predict(root, r) for r in train_data]
    N = len(predict_Y)
    ein = sum([predict_Y[i] != train_data[i][2] for i in range(N)]) * 1.0 / N
    print ein
    print '----------------------------------------'


    print '----------------------------------------'
    print '         Homework 3 Question 15         '
    print '----------------------------------------'
    print 'Eout (evaluated with 0/1 error):'
    #其实从训练过程中就可以看出，ein就应该为0
    predict_Y = [predict(root, r) for r in test_data]
    N = len(predict_Y)
    eout = sum([predict_Y[i] != test_data[i][2] for i in range(N)]) * 1.0 / N
    print eout
    print '----------------------------------------'

if __name__ == '__main__':
    main()
