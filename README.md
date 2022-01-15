# Aprendizaje ID3

Assumptions about the datasets:
- The variables are discrete and composed of a set of labels.
- The decision variable is in the last column.
- The tree is capable of classifying in more than 2 classes.
- The missing or unlabeled values are not considered (will be treated as a different label ?).

##### Project steps

1. CVS file processing
2. Generating a tree
    - entropy
    - gain
3. Prediction
4. Visual presentation of the tree
5. Performance improvements
    - better build time
    - better clasification

##### Datasets
- Mushroom dataset (22 attributes, 8124 instances, 2 classes) https://archive.ics.uci.edu/ml/datasets/Mushroom
- Chess dataset (36 attributes, 3196 instances, 2 classes) https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29
- Connect 4 dataset (42 attributes, 67557 instances, 3 classes) https://archive.ics.uci.edu/ml/datasets/Connect-4
- Example dataset from class (5 attributes, 14 instances, 2 classes)