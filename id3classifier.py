import numpy as np
import pandas as pd
from random import choice, randint
import random
from timeit import default_timer as timer

mutate = False
early_stop = False

def findBestAttribute(df):
    """
    For a dataframe return an attribute with biggest gain.
    If all are equal to 0 - returns a random attribute.

    90% chance a better attribute is chosen.
    """
    entropy = calculateEntropy(df)
    best = (0, None)
    for attribute in df.columns[:-1]:
        gain = calculateGain(df, attribute, entropy)
        if gain >= best[0]:
            if mutate:
                if(randint(1, 100) > 10):
                    best = (gain, attribute)
            else:
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


def hasMajorityClass(df):
    """
    Returns True if there is a majority class in dataframe.
    """
    if (df[df.columns[-1]].value_counts().max()/len(df)) > 0.8:
        return True
    return False


def getAccuracy(traincsv, testcsv, classname):
    """
    Trains the decision tree and returns its accuracy in %
    """
    tree = DecisionTreeID()
    tree.learnDT(traincsv, first_id=False)
    true = pd.read_csv(testcsv)
    predicted = tree.prediction(testcsv)
    predicted["Correct"] = np.where(true[classname] == predicted[classname], 1, 0)
    return sum(predicted["Correct"])/len(predicted)*100


def measure_time_performance(n):
    time_original = 0
    time_mutated = 0
    time_early_stop = 0
    time_early_stop_mutated = 0
    global mutate, early_stop
    for _ in range(n):
        mutate = False
        early_stop = False
        t = DecisionTreeID()
        time_original += t.learnDT("data/chess-train.csv",first_id=False)

        mutate = True
        early_stop = False
        t = DecisionTreeID()
        time_mutated += t.learnDT("data/chess-train.csv",first_id=False)

        mutate = False
        early_stop = True
        t = DecisionTreeID()
        time_early_stop += t.learnDT("data/chess-train.csv",first_id=False)

        mutate = True
        early_stop = True
        t = DecisionTreeID()
        time_early_stop_mutated += t.learnDT("data/chess-train.csv",first_id=False)

    print("original", time_original/n, "s")
    print("mutated", time_mutated/n, "s")
    print("early stop", time_early_stop/n, "s")
    print("early stop & mutated", time_early_stop_mutated/n, "s")

def measure_accuracy(n):
    predictions = []
    predictions_mutated = []
    predictions_early_stop = []
    predictions_early_stop_mutated = []
    global mutate, early_stop
    for _ in range(n):
        mutate = False
        early_stop = False
        predictions.append(round(getAccuracy("data/chess-train.csv",
                                             "data/chess-test.csv",
                                             "class"), 2))
        mutate = True
        early_stop = False
        predictions_mutated.append(round(getAccuracy("data/chess-train.csv",
                                                     "data/chess-test.csv",
                                                     "class"), 2))
        mutate = False
        early_stop = True
        predictions_early_stop.append(round(getAccuracy("data/chess-train.csv",
                                                        "data/chess-test.csv",
                                                        "class"), 2))
        mutate = True
        early_stop = True
        predictions_early_stop_mutated.append(round(getAccuracy("data/chess-train.csv",
                                                                "data/chess-test.csv",
                                                                "class"), 2))
    print("original- avg:", sum(predictions)/n, "%, max:",
          max(predictions), "%")
    print("mutated- avg:", sum(predictions_mutated)/n, "%, max:",
          max(predictions_mutated), "%")
    print("early stop- avg:", sum(predictions_early_stop)/n, "%, max:",
          max(predictions_early_stop), "%")
    print("early stop & mutated- avg:", sum(predictions_early_stop_mutated)/n, "%, max:",
          max(predictions_early_stop_mutated), "%")


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
        if df is not None:
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
            elif early_stop and hasMajorityClass(sliced):
                self.children[value] = Node(None, None, self,
                                            chooseMayorityClass(sliced))
            else:
                self.children[value] = Node(sliced,
                                            findBestAttribute(sliced),
                                            self)

    def print(self, level=1):
        """
        Prints the attribute label and values for children or class label.
        """
        if self._class is None:
            for child in self.children.items():
                print("  "*(level-1), "--"*level, self.attribute, "=", child[0])
                child[1].print(level=level+1)
        else:
            print("  "*(level-1), "--"*level, self._class, "(class)", "\n")


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

    def learnDT(self, csvname=None, first_id=True, data=None):
        """
        Trains the decision tree with data from CSV file.
        The decision variable is the last column.
        If there is no indexing in the CSV file set first_id=False.
        If no CSV file a dataframe data should be passed
        """
        df = data
        if csvname:
            df = self.readCSV(csvname)
        if first_id:
            df = df[df.columns[1:]]
        self.predicted_attribute = df.columns[-1]
        start = timer()
        self.root = Node(df, findBestAttribute(df))
        end = timer()
        return end - start

    def drawDecisionTree(self):
        """
        Draws the decision tree using the Node.draw() function.
        """
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
        When attribute value is not in a tree a random branch is chosen.
        """
        if node._class is not None:
            return node._class
        else:
            try:
                value = df[node.attribute]
                child = node.children[value]
            except KeyError:
                child = random.choice(list(node.children.values()))
            return self.__predict(df, child)

    def readCSV(self, csvname):
        """
        Reads a CSV data into dataframe using pandas library.
        """
        return pd.read_csv(csvname)


if __name__ == "__main__":

    # -- Example of drawn decision tree -------
    tree = DecisionTreeID()
    tree.learnDT("data/farmaco.csv")
    tree.drawDecisionTree()

    # -- Comparison of efficiency for modifications ---
    measure_accuracy(5)

    # -- Measuring the time performance with modifications ---
    measure_time_performance(5)

    # -- Example for 4 class tree
    car_tree = DecisionTreeID()
    car_tree.learnDT("data/car-train.csv")
    car_tree.drawDecisionTree()
    print(round(getAccuracy("data/car-train.csv", "data/car-test.csv", "class"),2),"%")

    # -- Example of data that gives 100% accuracy--------
    print(round(getAccuracy("data/agaricus-lepiota-train.csv",
                            "data/agaricus-lepiota-test.csv", "class"), 2), "%")

    # # -- Code used for preparing datasets ---------------
    # df = pd.read_csv("data/car.csv")
    # # cols = df.columns.tolist()
    # # cols[0:-1],cols[-1] = cols[1:],cols[0]
    # # df = df[cols]
    # train = df.sample(frac=0.8)
    # test = df.drop(train.index)
    # train.to_csv("data/car-train.csv", index=False)
    # test.to_csv("data/car-test.csv", index=False)

    # # Example of use of Random Forest
    # import randomforest as rf
    # forest = rf.RandomForest("data/chess-train.csv", 5)
    # forest.predict("data/chess-test.csv")
