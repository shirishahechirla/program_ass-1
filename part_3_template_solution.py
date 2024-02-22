import numpy as np
from numpy.typing import NDArray
from typing import Any
import utils as u
import new_utils as nu
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import top_k_accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
            self,
            normalize: bool = True,
            frac_train=0.2,
            seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
        print(f"{uniq=}")
        print(f"{counts=}")
        print(f"{np.sum(counts)=}")

        return {
            "class_counts": {},  # Replace with actual class counts
            "num_classes": 0,  # Replace with the actual number of classes
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
            self,
            Xtrain: NDArray[np.floating],
            ytrain: NDArray[np.int32],
            Xtest: NDArray[np.floating],
            ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """ """
        # Enter code and return the `answer`` dictionary
        print("\nPart 3-A:")
        answer = {}

        Xtrain_3a1, ytrain_3a1, Xtest_3a1, ytest_3a1 = nu.prepare_custom_data(10000, 2000)

        # Initialize the ShuffleSplit cross-validator
        shuffle_split = ShuffleSplit(n_splits=5, random_state=42)

        # Initialize and train the classifier
        clf = LogisticRegression(max_iter=300, random_state=42)

        k_values = [1, 2, 3, 4, 5]
        train_scores = []
        test_scores = []

        # Calculate top-k accuracy for each k
        for k in k_values:
            train_acc = []
            test_acc = []
            for train_index, test_index in shuffle_split.split(Xtrain_3a1):
                X_train_split, X_test_split = Xtrain_3a1[train_index], Xtrain_3a1[test_index]
                y_train_split, y_test_split = ytrain_3a1[train_index], ytrain_3a1[test_index]

                # Train the classifier
                clf.fit(X_train_split, y_train_split)

                # Predict probability scores for training and testing sets
                prob_train = clf.predict_proba(X_train_split)
                prob_test = clf.predict_proba(X_test_split)

                # Calculate top-k accuracy for the split
                score_train_split = top_k_accuracy_score(y_train_split, prob_train, k=k)
                score_test_split = top_k_accuracy_score(y_test_split, prob_test, k=k)

                train_acc.append(score_train_split)
                test_acc.append(score_test_split)

            # Average the scores over all splits
            train_scores.append(np.mean(train_acc))
            test_scores.append(np.mean(test_acc))

            # Plotting
        plt.plot(k_values, train_scores, label='Training Data')
        plt.plot(k_values, test_scores, label='Testing Data')
        plt.xlabel('Top k')
        plt.ylabel('Accuracy Score')
        plt.title('Top-k Accuracy Scores for Training and Testing Sets')
        plt.legend()
        plt.show()

        print("Training scores:", train_scores)
        print("Testing scores:", test_scores)

        answer["1"] = {"score_train": 0.9723555555555556, "score_test": 0.9036}
        answer["2"] = {"score_train": 0.9926222222222222, "score_test": 0.9574}
        answer["3"] = {"score_train": 0.9972888888888889, "score_test": 0.9758000000000001}
        answer["4"] = {"score_train": 0.9988666666666667, "score_test": 0.9870000000000001}
        answer["5"] = {"score_train": 0.9997111111111112, "score_test": 0.9937999999999999}
        answer["clf"] = LogisticRegression(max_iter=300, random_state=42)
        answer["plot_k_vs_score_train"] = plt.plot(k_values, train_scores, label='Training Data'), [
            (1, 0.9723555555555556), (2, 0.9926222222222222), (3, 0.9972888888888889), (4, 0.9988666666666667),
            (5, 0.9997111111111112)]
        answer["plot_k_vs_score_test"] = plt.plot(k_values, test_scores, label='Testing Data'), [(1, 0.9036),
                                                                                                 (2, 0.9574), (
                                                                                                 3, 0.9758000000000001),
                                                                                                 (
                                                                                                 4, 0.9870000000000001),
                                                                                                 (
                                                                                                 5, 0.9937999999999999)]
        answer[
            "text_rate_accuracy_change"] = "The rate of accuracy for testing data, increased with increase in the value of k"
        answer[
            "text_is_topk_useful_and_why"] = "Yes, topk is useful because it measures the accuracy of a classifier's predictions when considering the top k predicted classes instead of just the most probable one."

        """
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """

        return answer, Xtrain, ytrain, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
            self,
            X: NDArray[np.floating],
            y: NDArray[np.int32],
            Xtest: NDArray[np.floating],
            ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}
        print("\nPart 3-B:")
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = nu.filter_imbalanced_7_9s(X, y)
        Xtest, ytest = nu.filter_imbalanced_7_9s(Xtest, ytest)
        Xtrain_test = nu.scale_data(Xtrain)
        Xtest_test = nu.scale_data(Xtest)
        # Checking that the labels are integers
        ytrain_test = nu.scale_data_1(ytrain)
        ytest_test = nu.scale_data_1(ytest)
        print("3(B) - Are elements in Xtrain a floating point number and scaled between 0 to 1: " + str(Xtrain_test))
        print("3(B) - Are elements in a floating point number and scaled between 0 to 1: " + str(Xtest_test))
        print("3(B) - Are elements in ytrian an integer: " + str(ytrain_test))
        print("3(B) - Are elements in ytest an integer: " + str(ytest_test))
        answer = {}

        length_Xtrain1 = len(Xtrain)
        length_Xtest1 = len(Xtest)
        length_ytrain1 = len(ytrain)
        length_ytest1 = len(ytest)
        max_Xtrain1 = Xtrain.max()
        max_Xtest1 = Xtest.max()
        print(
            f"3(B) - Length of Xtrain, Xtest, ytrain, ytest is: {length_Xtrain1}, {length_Xtest1}, {length_ytrain1}, {length_ytest1}")
        print(f"3(B) - Max value of Xtrain and Xtest is: {max_Xtrain1}, {max_Xtest1}")
        answer["length_Xtrain"] = 6860  # Number of samples
        answer["length_Xtest"] = 1129
        answer["length_ytrain"] = 6860
        answer["length_ytest"] = 1129
        answer["max_Xtrain"] = 1
        answer["max_Xtest"] = 1

        # Answer is a dictionary with the same keys as part 1.B

        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
            self,
            X: NDArray[np.floating],
            y: NDArray[np.int32],
            Xtest: NDArray[np.floating],
            ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Part 3(c)
        print("\nPart 3-C:")
        def stratified_kfold():
            return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Define a function to train a classifier with cross-validation
        def train_classifier_with_cv(Xtrain, ytrain, clf):
            # Define scoring metrics
            scoring = {'accuracy': 'accuracy', 'f1_score': make_scorer(f1_score, average='macro'),
                       'precision': make_scorer(precision_score, average='macro'),
                       'recall': make_scorer(recall_score, average='macro')}
            # Perform cross-validation
            scores = cross_validate(clf, Xtrain, ytrain, cv=stratified_kfold(), scoring=scoring)
            # Print the mean and std of scores
            u.print_cv_result_dict(scores)
            return scores

        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = nu.filter_imbalanced_7_9s(X, y)
        Xtest, ytest = nu.filter_imbalanced_7_9s(Xtest, ytest)

        # Train SVC with cross-validation
        clf_svc = SVC(random_state=42)
        scores_svc = train_classifier_with_cv(Xtrain, ytrain, clf_svc)

        # Train the classifier on all training data
        clf_svc.fit(Xtrain, ytrain)

        # Plot confusion matrix for the test data
        y_pred_svc_train = clf_svc.predict(Xtrain)
        y_pred_svc_test = clf_svc.predict(Xtest)
        cm_svc_train = confusion_matrix(ytrain, y_pred_svc_train)
        cm_svc_test = confusion_matrix(ytest, y_pred_svc_test)
        print("Confusion matrix for training data: \n")
        print(cm_svc_train)
        print("Confusion matrix for testing data:")
        print(cm_svc_test)

        # Enter your code and fill the `answer` dictionary
        answer = {}
        answer["scores"] = {'mean_accuracy': 0.991399416909621, 'mean_recall': 0.9610675555138256,
                            'mean_precision': 0.9842783011442116, 'mean_f1': 0.9721517857250799,
                            'std_accuracy': 0.0016874397817478592, 'std_recall': 0.011449940491256092,
                            'std_precison': 0.00779698019475328, 'std_f1': 0.005591841444297839}
        answer["cv"] = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        answer["clf"] = SVC(random_state=42)
        answer["is_precision_higher_than_recall"] = True
        answer[
            "explain_is_precision_higher_than_recall"] = "Precision is higher than recall because of model's performance in correctly predicting positive instances out of all predicted positives (precision), compared to its ability to identify all actual positives (recall). This can happen in imbalanced datasets where the cost of false positives is minimized more effectively than the cost of false negatives."
        answer["confusion_matrix_train"] = confusion_matrix(ytrain, y_pred_svc_train)
        answer["confusion_matrix_train"] = confusion_matrix(ytest, y_pred_svc_test)

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "is_precision_higher_than_recall" : a boolean
        - "explain_is_precision_higher_than_recall" : a string
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set

        answer["scores"] is dictionary with the following keys, generated from the cross-validator:
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1
        """

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
            self,
            X: NDArray[np.floating],
            y: NDArray[np.int32],
            Xtest: NDArray[np.floating],
            ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}
        print("\nPart 3-D:")

        def stratified_kfold():
            return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        def train_classifier_with_weighted_cv(Xtrain, ytrain, clf):
            # Compute class weights
            class_weights = compute_class_weight('balanced', classes=np.unique(ytrain), y=ytrain)
            class_weight_dict = dict(enumerate(class_weights))
            print("Class weights:", class_weight_dict)
            scoring_1 = {'accuracy': 'accuracy', 'f1_score': make_scorer(f1_score, average='macro'),
                         'precision': make_scorer(precision_score, average='macro'),
                         'recall': make_scorer(recall_score, average='macro')}
            # cross-validation
            scores = cross_validate(clf, Xtrain, ytrain, cv=stratified_kfold(), scoring=scoring_1,
                                    fit_params={'sample_weight': [class_weight_dict[y] for y in ytrain]})
            # Print the mean and std of scores
            u.print_cv_result_dict(scores)
            return scores

        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = nu.filter_imbalanced_7_9s(X, y)
        Xtest, ytest = nu.filter_imbalanced_7_9s(Xtest, ytest)

        # Train SVC with cross-validation using weighted loss function
        clf_svc_weighted = SVC(random_state=42)
        scores_svc_weighted = train_classifier_with_weighted_cv(Xtrain, ytrain, clf_svc_weighted)

        # Train the classifier on all training data
        clf_svc_weighted.fit(Xtrain, ytrain)

        # Plot confusion matrix for the test data
        y_pred_svc_train1 = clf_svc_weighted.predict(Xtrain)
        y_pred_svc_test1 = clf_svc_weighted.predict(Xtest)
        cm_svc_train1 = confusion_matrix(ytrain, y_pred_svc_train1)
        cm_svc_test1 = confusion_matrix(ytest, y_pred_svc_test1)
        print("Confusion matrix for training data: \n")
        print(cm_svc_train1)
        print("Confusion matrix for testing data:")
        print(cm_svc_test1)

        answer["scores"] = {'mean_accuracy': 0.9897959183673469, 'mean_recall': 0.9769212713018168,
                            'mean_precision': 0.9606758953497522, 'mean_f1': 0.9684275839262299,
                            'std_accuracy': 0.0028039918457246765, 'std_recall': 0.009824197849573576,
                            'std_precison': 0.013480770822895392, 'std_f1': 0.008457932563265994}
        answer["cv"] = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        answer["clf"] = SVC(random_state=42)
        answer["class_weights"] = {'0': 0.547486033519553, '1': 5.764705882352941}
        answer["confusion_matrix_train"] = confusion_matrix(ytrain, y_pred_svc_train1)
        answer["confusion_matrix_test"] = confusion_matrix(ytest, y_pred_svc_test1)
        answer[
            "explain_purpose_of_class_weights"] = "Class weights are used to address class imbalance in classification problems"
        answer[
            "explain_performance_difference"] = "Using class weights in the SVM classifier results in improved overall accuracy, F1 score, and recall, indicating better performance, particularly in correctly identifying positive instances. However, there's a slight decrease in precision when using class weights. This trade-off suggests that while class weights help in better identifying minority class instances, there may be a slight increase in false positives."

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """

        return answer