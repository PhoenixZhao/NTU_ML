#coding=utf8
'''
    台大ML作业3的Q13-Q20,自己实现简单的决策树
    C&RT alogrithm and Gini index
'''

all_nodes = [] #记录所有的node，根据index来保存树结构
INF = 10000
LEFT_LABEL = -1
RIGHT_LABEL = 1


train_data_file = '../assignment3/hw3_train.dat'
test_data_file = '../assignment3/hw3_test.dat'

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

def load_data():
    '''
        读入训练集和测试集，格式为:
            feature1    feature2  label
    '''
    lines = open(train_data_file, 'r').readlines()
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
    sorted(train_data, key=lambda d:d[feature_index], reverse=False)
    for i in range(len(train_data) - 1):
        theta_values.append((train_data[i][feature_index] + train_data[i+1][feature_index]) * 0.5)
    theta_values.append(INF)
    return theta_values

def sign(x):
    return 1 if x>= 0 else -1

def impurity(data):
    '''
        计算impurity，使用Gini index
        data为dict类型，key=label, value=records
        使用公式impurity = 1 - sum_k{(1-sum_N[[y==k]] / N)}^2
    '''
    t = 0.00
    N = len(data.get(LEFT_LABEL, []) + data.get(RIGHT_LABEL, []))
    for label, records in data.items():
        correct_num = 0
        for _, _, y in records:
            correct_num += int(label == y)
        t += pow(correct_num * 1.0 / N, 2)
    return 1.0 - t

def get_opt_parameters(train_data):
    '''
        train_data格式: (feature1, feature2, label)
        使用f = sign(x_i - theta)来找到最好的参数组合
        做每个特征的时候,将train_data按照对应的 特征进行排序，得到N个value，然后取每段区间的中点作为theta，再加上
        -inf, inf，一共N+1个值
        然后在特征和N+1个值里进行遍历，找到impurity最小的参数组合
        返回参数组合已经切分的数据, -1为左孩子, +1为右孩子
    '''
    splited_data = {}#key in (-1, 1), values = []
    min_impurity, opt_f_i, opt_theta = 10.0, 0, -INF
    for f_i in range(2):
        theta_values = generate_thetas(f_i, train_data)
        for theta in theta_values:
            splited_data = {}
            for row in train_data:
                y = sign(row[f_i] - theta)
                splited_data.setdefault(y, []).append(row)
            tmp_impurity = impurity(splited_data)

            if tmp_impurity < min_impurity:
                min_impurity = tmp_impurity
                opt_theta = theta
                opt_f_i = f_i
    return opt_f_i, opt_theta, splited_data

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
    print 'create children(l=%d,r=%d) for node %d' % (pos, pos+1, node.index)
    return lchild, rchild

def decision_tree(node, train_data, label=1):
    '''
        递归调用，根据train_data的impurity判断是否需要terminate;
        如果没有terminate，则选择最好的feature_i，和theta，将数据进行划分；
        -1往左走，+1往右走，创建两个新node，继续重复上述过程直到terminate
    '''
    global all_nodes
    if impurity(dict({label: train_data})) == 0:
        all_nodes.append(node)
        return

    theta, f_i, splited_data = get_opt_parameters(train_data)
    node.theta = theta
    node.feature_i = f_i

    lchid, rchild = create_children(node)

    decision_tree(lchid, splited_data[LEFT_LABEL], LEFT_LABEL)

    decision_tree(rchid, splited_data[RIGHT_LABEL], RIGHT_LABEL)

def main():
    train_data, test_data = load_data()
    root = Node(0)
    global all_nodes
    all_nodes.append(root)
    decision_tree(root, train_data)
    print 'finished, there is %d nodes in the tree' % (len(all_nodes))

if __name__ == '__main__':
    main()
