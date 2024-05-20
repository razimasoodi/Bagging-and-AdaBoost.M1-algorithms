# Bagging-and-AdaBoost.M1-algorithms
As you know these algorithms need base classifiers, for example decision trees. You need to use
sklearn.tree.DecisionTreeClassifier(…) classifier of Scikitlearn package as the base learner of you
ensemble models.

Note1: max_depth parameter of the base learner in AdaBoost.M1 algorithm should be tuned
experimentally so that the decision tree performs a little better than a random classifier. For the
Bagging algorithm, use the default parameters of the base learner decision tree.

Add Gaussian noise with the following parameters 𝒩 ~ (0, 1) to 10%, 20%, and 30% of the features
randomly on each data set and compare the results with noiseless setting.

You should split each data set to train and test parts. Use 70% of the data for training phase and the
reaming 30% for testing phase. Run your codes for 10 individual runs and report the average test
accuracies of 10 runs on each dataset.

Note2: the iteration number 𝑇 in Bagging and AdaBoost.M1 algorithms should be obtained from sets
{11, 21, 31, 41} and {21, 31, 41, 51}, respectively. In other words, you should test the performance of
the algorithms with the given 𝑇 values of each algorithm and report your best results over a fixed 𝑇
value.
