import pandas as pd
from id3classifier import DecisionTreeID


class RandomForest:
    """
    Represent a Random Forest decision model.
    Attributes:
    trees : list (default None)
        The list of ID3 trees in the model.
    Parameters:
    n : int (default 2)
        Number of trees in the model.
    f : float (default 0.5)
        Fraction of columns to be sampled for each tree.
    """

    def __init__(self, csvname, n=2, f=0.5):
        self.trees = [DecisionTreeID() for i in range(n)]
        data = pd.read_csv(csvname)
        self.learn_forest(n, f, data)

    def learn_forest(self, n, f, data):
        """
        Learns the n trees in the forest from sampled data.
        """
        df = data[data.columns[:-1]]
        for i in range(n):
            sampled_df = df.sample(frac=f, axis='columns')
            sampled_df[data.columns[-1]] = data[data.columns[-1]]
            sampled_df = sampled_df.sample(frac=(1/n), axis='index')
            self.trees[i].learnDT(data=sampled_df)
            print(f"Trained tree {i}")

    def predict(self, csvname):
        """
        Returns a list of averaged predictions for the forest.
        """
        predictions = []
        res = []
        for tree in self.trees:
            predictions.append(tree.prediction(csvname).iloc[:, -1])
        for i in range(len(predictions[0])):
            results = []
            for pred in predictions:
                results.append(pred.iloc[i])
            res.append(max(set(results), key=results.count))
        return res
