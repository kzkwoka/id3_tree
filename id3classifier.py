import numpy as np
import pandas as pd
from random import choice


def findBestAttribute(df):
    """
    For a dataframe return an attribute with biggest gain.
    If all are equal to 0 - returns a random attribute.
    """
    entropy = calculateEntropy(df)
    best = (0, None)
    for attribute in df.columns[:-1]:
        gain = calculateGain(df, attribute, entropy)
        if gain > best[0]:
            best = (gain, attribute)
    if best[1] is None:
        return choice(df.columns[:-1])
    else:
        return best[1]


def calculateEntropy(df):
    """
    Returns calculated entropy of a dataframe.
    For conditional entropy pass appropriate slice of dataframe.
    """
    len_ = len(df)
    parts = [i*np.log2(i/len_)/len_ for i in df[df.columns[-1]].value_counts()]
    return -sum(parts)


def calculateGain(df, attribute, entropy):
    """
    Returns calculated gain for an attribute.
    """
    parts = []
    for value in df[attribute].unique():
        sliced_df = df.loc[df[attribute] == value]
        cond_entropy = calculateEntropy(sliced_df)
        parts.append(cond_entropy*len(sliced_df)/len(df))
    return entropy - sum(parts)


def chooseMayorityClass(df):
    """
    Returns the most common class in dataframe.
    Used when there are no more attributes to split.
    """
    return df[df.columns[-1]].value_counts().idxmax()


class Node:
    """
    Represents a node of a decision tree.
    Attributes:
    parent : Node (default None)
        A parent node. If None - it's the tree root.
    attribute : str
        The name of the attribute by which we split in this node.
    _class : str (default None)
        Contains the name of class. If not None, the node is a leaf.
    children : dict (default None)
        A dictionary of children nodes keyed with attribute values.
    """
    def __init__(self, df, attribute, parent=None, _class=None):
        self.parent = parent
        self.attribute = attribute
        self._class = _class
        self.children = None
        values = df[df.columns[-1]].unique()
        # If the data in this node represents only one class - no split is performed.
        if len(values) == 1:
            self.attribute = "Class"
            self._class = values[0]
        if self._class is None:
            self.__createChildren(df, attribute)

    def __createChildren(self, df, attribute):
        """
        Creates a dictionary of children keyed by values of attribute passed.
        If after splitting the data for the child node is only the decision variable,
        a leaf node with mayority class is created as a child.
        """
        self.children = {}
        for value in df[attribute].unique():
            sliced = df.loc[df[attribute] == value]
            sliced = sliced.drop(columns=attribute)
            if len(sliced.columns) == 1:
                self.children[value] = Node(None, None, self,
                                            chooseMayorityClass(sliced))
            else:
                self.children[value] = Node(sliced,
                                            findBestAttribute(sliced), self)

    def print(self, level=1):
        """
        Prints the attribute label and values for children or class label.
        """
        if self._class is None:
            for child in self.children.items():
                print("  "*(level-1), "--"*level, self.attribute, "=", child[0])   
                child[1].print(level=level+1)
        else:
            print("  "*(level-1), "--"*level,self._class, "(class)", "\n")



class DecisionTreeID:
    """
    Represent a ID3 decision tree.
    Attributes:
    root : Node (default None)
        The root node of the tree.
    predicted_attribute : str (default None)
        The name of the decision variable.
    """
    def __init__(self):
        self.root = None
        self.predicted_attribute = None

    def learnDT(self, csvname, first_id=True):
        """
        Trains the decision tree with data from CSV file.
        The decision variable is the last column.
        If there is no indexing in the CSV file set first_id=False.
        """
        df = self.readCSV(csvname)
        if first_id:
            df = df[df.columns[1:]]
        self.predicted_attribute = df.columns[-1]
        self.root = Node(df, findBestAttribute(df))

    def drawDecisionTree(self):
        """
        Draws the decision tree using the Node.draw() function.
        """
        # TODO: To be corrected
        self.root.print()


    def prediction(self, csvname):
        """
        Predicts the the values of the decision variable for data given in CSV file.
        Appends a column with the predicted values.
        Returns a new dataframe.
        """
        df = self.readCSV(csvname)
        df.drop(columns=self.predicted_attribute, errors='ignore')
        predictions = []
        for row in df.iloc:
            predictions.append(self.__predict(row, self.root))
        df[self.predicted_attribute] = predictions
        return df

    def __predict(self, df, node):
        """
        Helper function for iterating over a tree.
        Returns predicted class.
        TODO: Consider attribute values not in a tree.
        """
        if node._class is not None:
            return node._class
        else:
            value = df[node.attribute]
            return self.__predict(df, node.children[value])

    def readCSV(self, csvname):
        """
        Reads a CSV data into dataframe using pandas library.
        """
        return pd.read_csv(csvname)


if __name__ == "__main__":
    tree = DecisionTreeID()
    tree.learnDT("data/farmaco.csv")
    print(tree.prediction("data/farmaco_test.csv"))
    tree.drawDecisionTree()

    mushroom_tree = DecisionTreeID()
    mushroom_tree.learnDT("data/agaricus-lepiota-train.csv")
    true = pd.read_csv("data/agaricus-lepiota-test.csv")
    predicted = mushroom_tree.prediction("data/agaricus-lepiota-test.csv")
    predicted["Correct"] = np.where(true["class"]==predicted["class"],1,0)
    print(sum(predicted["Correct"])/len(predicted)*100,"%")

    # used for preparing datasets
    # df = pd.read_csv("data/agaricus-lepiota.csv")
    # # cols = df.columns.tolist()
    # # cols[0:-1],cols[-1] = cols[1:],cols[0]
    # # df = df[cols]
    # train = df.sample(frac=0.8)
    # test = df.drop(train.index)
    # train.to_csv("data/agaricus-lepiota-train.csv", index=False)
    # test.to_csv("data/agaricus-lepiota-test.csv", index=False)
