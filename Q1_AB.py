import numpy as np


def main():
    print('START Q1_AB\n')
    trainingfile = "datasets/Q1_train.txt"
    testfile = "datasets/Q1_test.txt"
    training = removespecialcharsfromdataset(trainingfile) #function to remove special chara from fie
    test = removespecialcharsfromdataset(testfile)
    for depth in range(1,6):  #iterating for creating decison trees for depth 1-5
        classifier = DecisionTree(tree_depth=depth)  #calling the decision tree class using classifier object
        classifier.fit(training)
        Ytrain_pred = classifier.predict(training)
        Ytest_pred = classifier.predict(test)
        accuracy_train=findaccuracy(training, Ytrain_pred)  #training data accuracy
        accuracy_test=findaccuracy(test,Ytest_pred)   #test data accuracy
        print("DEPTH =", depth)
        print("Accuracy | Train = ",accuracy_train," | Test = ",accuracy_test)
    print("\nFinal Decision tree:\n")
    classifier.print_tree()
    print('END Q1_AB\n')


class Node():
    def __init__(self, feature_idx=None, threshold=None, leftchild=None, rightchild=None, info_gain=None, value=None, leftchild_entropy=None, rightchild_entropy=None):
        self.feature_idx = feature_idx
        self.threshold = threshold #to store threshold values
        self.leftchild = leftchild  #to store left child values
        self.rightchild = rightchild   #to store right child values
        self.info_gain = info_gain      #to store info gain
        self.leftchild_entropy=leftchild_entropy   #to store left child entropy
        self.rightchild_entropy=rightchild_entropy #to store right child entropy

        self.value = value   #store leaf node


class DecisionTree():
    def __init__(self,  tree_depth=5):
        self.root = None
        self.tree_depth = tree_depth

    def build_tree(self, dataset, curr_depth=0):  #function to build the tree
        no_features = len(dataset[0])-1   #get no of features
        if curr_depth < self.tree_depth:
            best_spt_tree = self.get_best_split_tree(dataset, no_features)  #find the best split tree

            if best_spt_tree["info_gain"] > 0:    #check if information gain>0
                left_subtree = self.build_tree(best_spt_tree["left_subset"], curr_depth + 1)  #create left subtree
                right_subtree = self.build_tree(best_spt_tree["right_subset"], curr_depth + 1)   #create right subtree
                return Node(best_spt_tree["feature_idx"], best_spt_tree["threshold"],
                            left_subtree, right_subtree, best_spt_tree["info_gain"],leftchild_entropy=best_spt_tree["leftchild_entropy"],rightchild_entropy=best_spt_tree["rightchild_entropy"])

        leaf_value = self.find_leafnode_value(dataset)  #find the leaf node
        return Node(value=leaf_value)  #returning leaf Node

    def get_best_split_tree(self, data, no_features):#function to find the best split
        best_spt_tree = {}
        max_info_gain = -float("inf")    #setting max info gain as negative inf

        for feature_idx in range(no_features):
            feature_values=[]
            for i in range(len(data)):
                feature_values.append(data[i][feature_idx])   #iterating the feature values one by one to calulate thresholds
            feature_thresholds=self.findthresholdsforfeature(feature_values)  #call to find thresholds
            for threshold in feature_thresholds:  #iterating thresholds
                left_subset, right_subset = self.split_tree(data, feature_idx, threshold)  #return the left and right split using threshold
                if len(left_subset) > 0 and len(right_subset) > 0:   #checking if left and right split is not empty
                    y = [item[-1] for item in data]    #getting all class values of  dataset
                    left_y = [item[-1] for item in left_subset]  #getting all class values of left split
                    right_y = [item[-1] for item in right_subset]  #getting all class values of right split
                    info_gain, leftchild_entropy, rightchild_entropy = self.find_information_gain(y, left_y, right_y)  #call to find info gain
                    if info_gain > max_info_gain:  #check if new info gain is greater than already calculated if so update as best split
                        best_spt_tree["feature_idx"] = feature_idx
                        best_spt_tree["threshold"] = threshold
                        best_spt_tree["left_subset"] = left_subset
                        best_spt_tree["right_subset"] = right_subset
                        best_spt_tree["info_gain"] = info_gain
                        best_spt_tree["leftchild_entropy"] = leftchild_entropy
                        best_spt_tree["rightchild_entropy"] = rightchild_entropy

                        max_info_gain = info_gain  #if new greater info gain found update the max info gain
        return best_spt_tree

    def findthresholdsforfeature(self, feature_values):  #function to thresholds for each feature
        feature_values.sort()
        thresholds=[]
        for i in range(len(feature_values)-1):
          thresholds.append((feature_values[i]+feature_values[i+1])/2) #finding the threshold as mid point of two data points
        return thresholds


    def split_tree(self, data, feature_idx, threshold):
        left_subset = ([row for row in data if row[feature_idx] <= threshold])  #finding the left and right subtree by comparing with threshold
        right_subset =([row for row in data if row[feature_idx] > threshold])
        return left_subset, right_subset

    def find_information_gain(self, parent, left_child, right_child): #function to find information gain
        left_weight = len(left_child) / len(parent)   #finding left child count w.r.t parent
        right_weight = len(right_child) / len(parent)  #finding right child count w.r.t parent
        leftchild_entropy=left_weight * self.findentropy(left_child)
        rightchild_entropy=right_weight * self.findentropy(right_child)  #calculating entropy

        info_gain = self.findentropy(parent) - (leftchild_entropy+rightchild_entropy)
        return info_gain, leftchild_entropy, rightchild_entropy

    def findentropy(self, y):  #function to find entropy
        classes = np.unique(y)   #take distinct classes
        class_entropy = 0
        for cls in classes:
            class_count=y.count(cls)
            prob_class = class_count/ len(y) #find probability of a class over total data
            class_entropy += -prob_class * np.log2(prob_class)  #entropy calculation
        return class_entropy

    def find_leafnode_value(self, X): #function to get the value of leaf node
        X = list(X)
        return max(X, key=X.count)

    def print_tree(self, tree=None, entropy=None, space=" "):  #function to print the decision tree
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value[3]+" (Entropy:",entropy,")")  #printing the entropy along with leaf nodes

        else:
            print("Feature " + str(tree.feature_idx+1), " >=", round(tree.threshold,2), "(Info Gain: ", round(tree.info_gain,3),")")
            print("%sleft:" % (space), end="")
            self.print_tree(tree.leftchild,entropy=tree.leftchild_entropy, space=space + space)
            print("%sright:" % (space), end="")
            self.print_tree(tree.rightchild, entropy=tree.rightchild_entropy, space=space + space)


    def predict(self, X):  #function for returning prediction
        predictions = [self.traverse_and_predict(x, self.root) for x in X]
        return predictions

    def traverse_and_predict(self, x, decisiontree): #recursive function to find the predicted class for data
        if decisiontree.value != None:
            return decisiontree.value[3]
        feature_val = x[decisiontree.feature_idx]
        if feature_val <= decisiontree.threshold:  #checking the data against threshold in decison tree
            return self.traverse_and_predict(x, decisiontree.leftchild)  #checking left or right subtree
        else:
            return self.traverse_and_predict(x, decisiontree.rightchild)

    def fit(self, X):
        self.root = self.build_tree(X)

def removespecialcharsfromdataset(filepath):
    dataset = []
    no_of_cols = 0  # no of different features in data
    # reading training data and test data from text file
    with open(filepath) as fic:
        for line in fic:
            line = line.rstrip(",\r\n")
            patt = line.replace('(', '').replace(')', '').replace(' ', '').strip()
            row = list(patt.split(","))
            no_of_cols = len(row)
            dataset.append(row)
        for x in range(len(dataset)):
            for y in range(no_of_cols - 1):
                dataset[x][y] = float(dataset[x][y])
        return dataset


def findaccuracy(dataset,predictions):  #function to find the accuracy of decision tree
    predictions_correct=0
    for i in range(len(dataset)):
        if dataset[i][3]==predictions[i]:  #checking the actual ouput equal to predicted output
            predictions_correct+=1
    accuracy=predictions_correct/len(dataset)
    return round(accuracy,2)


if __name__ == "__main__":
    main()
