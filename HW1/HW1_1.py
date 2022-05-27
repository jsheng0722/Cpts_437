import math
import random

import numpy as np
import matplotlib.pyplot as plt

class TreeNode:
    def __init__(self, majClass):
        self.split_feature = -1  # -1 indicates leaf node
        self.children = {}  # dictionary of {feature_value: child_tree_node}
        self.majority_class = majClass

def binning_data(datas,bins):
    # for every col there are 562 cols
    for i in range(len(datas[0])-1):
        # create new dic for every col
        dic = {}
        # create new list for every col
        a = []
        # for every data in 3628 rows
        for j in datas[:,i]:
            a.append(j)
        # sort the datas in that col
        a.sort()
        # print('a: ',a)
        # plength = (max - min / bins) for whole col
        plength = float((a[-1]-a[0])/bins)
        # print('plength: ',plength)
        # Division of the number of bins
        for n in range(bins):
            # create new dictionary for every col
            dic[str(n)] = []
            # if there is only 1 bin or there is first bin
            if n == 0:
                # m is each data in a list
                for m in a:
                    # judge data is in first bin, if it is, put in dictionary (such as dic = {'0':[datas < 1/bins]})
                    if m < plength * (n + 1) + a[0]:
                        dic[str(n)].append(m)
            # last bins
            if n == bins - 1:
                for m in a:
                    if m >= a[-1] - plength:
                        dic[str(n)].append(m)
            # middle bin
            if (n > 0) and (n < bins - 1):
                for m in a:
                    if (m >= a[0] + plength * n) and (m < a[0] + plength * (n + 1)):
                        dic[str(n)].append(m)
        # print("dic: ", dic)
        # for every col, change from value to key in dictionary
        for o in range(len(datas[:,i])):
            for k,v in dic.items():
                if datas[:,i][o] in v:
                    datas[:,i][o] = k
    return datas

def build_tree(datas):
    #print("datas: ",datas)
    print("length of datas:",len(datas[0]))
    if not datas:
        return None
    # collect sets of values for each feature index, based on the datas
    features = {}
    for feature_index in range(len(datas[0]) - 1):
        features[feature_index] = set([data[feature_index] for data in datas])
    return build_tree_1(datas, features)


def build_tree_1(datas, features):
    tree_node = TreeNode(majority_class(datas))
    # if datas all have same class, then return leaf node predicting this class
    if same_class(datas):
        return tree_node
    # if no more features to split on, then return leaf node predicting majority class
    if not features:
        return tree_node
    # split on best feature and recursively generate children
    best_feature_index = best_feature(features, datas)
    tree_node.split_feature = best_feature_index
    remaining_features = features.copy()
    remaining_features.pop(best_feature_index)
    for feature_value in features[best_feature_index]:
        split_datas = filter_datas(datas, best_feature_index, feature_value)
        tree_node.children[feature_value] = build_tree_1(split_datas, remaining_features)
    return tree_node


def majority_class(datas):
    classes = [data[-1] for data in datas]
    return max(set(classes), key=classes.count)


def same_class(datas):
    classes = [data[-1] for data in datas]
    return (len(set(classes)) == 1)


def best_feature(features, datas):
    # Return index of feature with lowest entropy after split
    best_feature_index = -1
    best_entropy = 2.0  # max entropy = 1.0
    for feature_index in features:
        se = split_entropy(feature_index, features, datas)
        if se < best_entropy:
            best_entropy = se
            best_feature_index = feature_index
    return best_feature_index


def split_entropy(feature_index, features, datas):
    # Return weighted sum of entropy of each subset of datas by feature value.
    se = 0.0
    for feature_value in features[feature_index]:
        split_datas = filter_datas(datas, feature_index, feature_value)
        se += (float(len(split_datas)) / float(len(datas))) * entropy(split_datas)
    return se


def entropy(datas):
    classes = [data[-1] for data in datas]
    classes_set = set(classes)
    class_counts = [classes.count(c) for c in classes_set]
    e = 0.0
    class_sum = sum(class_counts)
    for class_count in class_counts:
        if class_count > 0:
            class_frac = float(class_count) / float(class_sum)
            e += (-1.0) * class_frac * math.log(class_frac, 2.0)
    return e


def filter_datas(datas, feature_index, feature_value):
    # Return subset of datas with given value for given feature index.
    return list(filter(lambda data: data[feature_index] == feature_value, datas))


def print_tree(tree_node, feature_names, depth=1):
    indent_space = depth * "  "
    if tree_node.split_feature == -1:  # leaf node
        print(indent_space + feature_names[-1] + ": " + tree_node.majority_class)
    else:
        for feature_value in tree_node.children:
            print(indent_space + feature_names[tree_node.split_feature] + " == " + feature_value)
            child_node = tree_node.children[feature_value]
            if child_node:
                print_tree(child_node, feature_names, depth + 1)
            else:
                # no child node for this value, so use majority class of parent (tree_node)
                print(indent_space + "  " + feature_names[-1] + ": " + tree_node.majority_class)


def classify(tree_node, instance):
    if tree_node.split_feature == -1:
        return tree_node.majority_class
    child_node = tree_node.children[instance[tree_node.split_feature]]
    if child_node:
        return classify(child_node, instance)
    else:
        return tree_node.majority_class

def read_data():
    data = np.loadtxt(fname = 'alldata.csv', delimiter = ',')
    print ("my data: ", data)
    print("------Read data successfully!------")
    return data

def feature_define(num):
    feature_names = []
    for i in range(num):
        n = chr(ord('A')+(i%26))
        m = chr(ord('@')+(int)(i/26))
        if m == '@':
            feature_names.append(n)
        else:
            feature_names.append(m + n)
    return feature_names

def draw_plot(datas, number_of_instances, y):
    if y < 10:
        y = 10
        print("y should not less than 10.")
    test_datas = []
    test_classes = []
    test_instances = []
    ax = []
    for n in range(y):
        num_of_corrent = 0
        data_for_testing = datas.copy()
        #print("len of data_for_testing: ",len(data_for_testing))
        for i in range(number_of_instances):
            rand = random.randint(0, len(data_for_testing))
            test_instance = data_for_testing[rand][:-1]
            test_classes.append(data_for_testing[rand][-1])
            test_datas.append(classify(tree, test_instance))
            if classify(tree, test_instance) == data_for_testing[rand][-1]:
                num_of_corrent = num_of_corrent + 1
            test_instances.append(data_for_testing.pop(rand))
        rate_of_corrent = float(num_of_corrent / number_of_instances)
        ax.append(rate_of_corrent)
    print("accuracy of datas: ", ax)
    plt.figure()
    yy = np.arange(1,y+1)
    plt.plot(yy, ax, linewidth = 1, color = 'red')
    plt.show()

if __name__ == "__main__":
    data = read_data()
    feature_names = feature_define(len(data[0]))
    print("feature_names:",feature_names)
    final_data = binning_data(data,5)
    # #print("y[0]",binning_data(X,4))
    li1 = []
    X = []
    y = []
    # change form to list
    for i in range(len(data)):
        li = []
        for j in final_data[i]:
            li.append(str(j))
        y.append(li[-1])
        li2 = li[:-1]
        li1.append(li)
        X.append(li2)
    modify_data = li1
    #print("my list: ", modify_data)
    # X = final_data[:,:-1]
    # y = modify_data[:,-1]
    #print("my x: ", X)
    print("my X's len: ", len(X))
    print("my y: ", y)
    print("my y's len: ", len(y))
    print("Feature num: ", len(modify_data[0]))
    print("data num: ", len(modify_data))

    tree = build_tree(modify_data)
    print("Tree:")
    print_tree(tree, feature_names)
    # # first subset of points for training the model
    # rand1 = random.randint(0,len(y))
    # test_instance1 = modify_data[rand1]
    # test_class1 = classify(tree, test_instance1)
    # print("\nTest instance[number." + str(rand1) + "]: " + str(test_instance1))
    # print("  class = " + test_class1)
    # # second subset of points for testing the model
    # rand2 = random.randint(0, len(y))
    # test_instance2 = modify_data[rand2]
    # test_class2 = classify(tree, test_instance2)
    # print("\nTest instance[number." + str(rand2) + "]: " + str(test_instance2))
    # print("  class = " + test_class2)

    # randomly-selected subset of 100 instances
    draw_plot(modify_data, 100, 10)
