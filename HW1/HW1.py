import math


class TreeNode:
    def __init__(self, majClass):
        self.split_feature = -1  # -1 indicates leaf node
        self.children = {}  # dictionary of {feature_value: child_tree_node}
        self.majority_class = majClass


def build_tree(examples):
    if not examples:
        return None
    # collect sets of values for each feature index, based on the examples
    features = {}
    print("build_tree--------------------",len(examples[0]))
    for feature_index in range(len(examples[0]) - 1):
        features[feature_index] = set([example[feature_index] for example in examples])
        print("build_tree--------------------", len(features[feature_index]))
    print("build_tree--------------------", features)
    return build_tree_1(examples, features)


def build_tree_1(examples, features):
    tree_node = TreeNode(majority_class(examples))
#    print("tree_node: ", tree_node)
    # if examples all have same class, then return leaf node predicting this class
    if same_class(examples):
        return tree_node
    # if no more features to split on, then return leaf node predicting majority class
    if not features:
        return tree_node
    # split on best feature and recursively generate children
    best_feature_index = best_feature(features, examples)
    print("---- best_feature_index---",  best_feature_index)
    tree_node.split_feature = best_feature_index
    remaining_features = features.copy()
    remaining_features.pop(best_feature_index)
    for feature_value in features[best_feature_index]:
        split_examples = filter_examples(examples, best_feature_index, feature_value)
        print("=======split_examples====", split_examples)
        tree_node.children[feature_value] = build_tree_1(split_examples, remaining_features)
    return tree_node


def majority_class(examples):
    classes = [example[-1] for example in examples]
    print("++++++++++++++++", classes)
    print("================", max(set(classes), key=classes.count))
    return max(set(classes), key=classes.count)


def same_class(examples):
    classes = [example[-1] for example in examples]
    return (len(set(classes)) == 1)


def best_feature(features, examples):
    selist = []
    # Return index of feature with lowest entropy after split
    best_feature_index = -1
    best_entropy = 2.0  # max entropy = 1.0
    for feature_index in features:
        print("--------features--------", features)
        se = split_entropy(feature_index, features, examples)
        print("--------se--------", se)
        if se < best_entropy:
            best_entropy = se
            best_feature_index = feature_index
    return best_feature_index


def split_entropy(feature_index, features, examples):
    # Return weighted sum of entropy of each subset of examples by feature value.
    se = 0.0
    print("--features[feature_index]--", features[feature_index])
    for feature_value in features[feature_index]:
        split_examples = filter_examples(examples, feature_index, feature_value)
        print("--split_examples--", split_examples)
        se += (float(len(split_examples)) / float(len(examples))) * entropy(split_examples)
    return se


def entropy(examples):
    classes = [example[-1] for example in examples]
    classes_set = set(classes)
    class_counts = [classes.count(c) for c in classes_set]
    e = 0.0
    class_sum = sum(class_counts)
    for class_count in class_counts:
        if class_count > 0:
            class_frac = float(class_count) / float(class_sum)
            e += (-1.0) * class_frac * math.log(class_frac, 2.0)
    return e


def filter_examples(examples, feature_index, feature_value):
    # Return subset of examples with given value for given feature index.
    return list(filter(lambda example: example[feature_index] == feature_value, examples))


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


if __name__ == "__main__":
    feature_names = ["Color", "Type", "Origin", "Stolen"]

    examples = [
        ["Red", "Sports", "Domestic", "Yes"],
        ["Red", "Sports", "Domestic", "No"],
        ["Red", "Sports", "Domestic", "Yes"],
        ["Yellow", "Sports", "Domestic", "No"],
        ["Yellow", "Sports", "Imported", "Yes"],
        ["Yellow", "SUV", "Imported", "No"],
        ["Yellow", "SUV", "Imported", "Yes"],
        ["Yellow", "SUV", "Domestic", "No"],
        ["Red", "SUV", "Imported", "No"],
        ["Red", "Sports", "Imported", "Yes"]
    ]
    tree = build_tree(examples)
    print("Tree:")
    print_tree(tree, feature_names)
    test_instance = ["Red", "SUV", "Domestic"]
    test_class = classify(tree, test_instance)
    print("\nTest instance: " + str(test_instance))
    print("  class = " + test_class)