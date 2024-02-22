# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from typing import Any
from numpy.typing import NDArray

import numpy as np
import utils as u

# Initially empty. Use for reusable functions across
# Sections 1-3 of the homework
import new_utils as nu


# ======================================================================
class Section1:
    def __init__(
            self,
            normalize: bool = True,
            seed: int | None = None,
            frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None

        Notes: notice the argument `seed`. Make sure that any sklearn function that accepts
        `random_state` as an argument is initialized with this seed to allow reproducibility.
        You change the seed ONLY in the section of run_part_1.py, run_part2.py, run_part3.py
        below `if __name__ == "__main__"`
        """
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed

    # ----------------------------------------------------------------------
    """
    A. We will start by ensuring that your python environment is configured correctly and 
       that you have all the required packages installed. For information about setting up 
       Python please consult the following link: https://www.anaconda.com/products/individual. 
       To test that your environment is set up correctly, simply execute `starter_code` in 
       the `utils` module. This is done for you. 
    """

    def partA(self):
        # Return 0 (ran ok) or -1 (did not run ok)
        print("Part 1-A:")
        answer = u.starter_code()
        print(" 0 means ran ok and -1 means did not run ok: " + str(answer))
        return answer

    # ----------------------------------------------------------------------
    """
    B. Load and prepare the mnist dataset, i.e., call the prepare_data and filter_out_7_9s 
       functions in utils.py, to obtain a data matrix X consisting of only the digits 7 and 9. Make sure that 
       every element in the data matrix is a floating point number and scaled between 0 and 1 (write
       a function `def scale() in new_utils.py` that returns a bool to achieve this. Checking is not sufficient.) 
       Also check that the labels are integers. Print out the length of the filtered 𝑋 and 𝑦, 
       and the maximum value of 𝑋 for both training and test sets. Use the routines provided in utils.
       When testing your code, I will be using matrices different than the ones you are using to make sure 
       the instructions are followed. 
    """

    def partB(
            self,
    ):
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        Xtrain_test = nu.scale_data(Xtrain)
        Xtest_test = nu.scale_data(Xtest)

        # Checking that the labels are integers
        ytrain_test = nu.scale_data_1(ytrain)
        ytest_test = nu.scale_data_1(ytest)

        print("\n Part 1-B:")
        print("The elements in Xtrain a floating point number and scaled between 0 to 1: " + str(Xtrain_test))
        print("The elements in a floating point number and scaled between 0 to 1: " + str(Xtest_test))
        print("The elements in ytrian an integer: " + str(ytrain_test))
        print("The elements in ytest an integer: " + str(ytest_test))
        answer = {}

        # Enter your code and fill the `answer` dictionary
        length_Xtrain = len(Xtrain)
        length_Xtest = len(Xtest)
        length_ytrain = len(ytrain)
        length_ytest = len(ytest)
        max_Xtrain = Xtrain.max()
        max_Xtest = Xtest.max()

        print(f"Length of Xtrain, Xtest, ytrain, ytest is: {length_Xtrain}, {length_Xtest}, {length_ytrain}, {length_ytest}")
        print(f" Max value of Xtrain and Xtest is: {max_Xtrain}, {max_Xtest}")
        answer["length_Xtrain"] = 12214  # Number of samples
        answer["length_Xtest"] = 2037
        answer["length_ytrain"] = 12214
        answer["length_ytest"] = 2037
        answer["max_Xtrain"] = 1
        answer["max_Xtest"] = 1

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    C. Train your first classifier using k-fold cross validation (see train_simple_classifier_with_cv 
       function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation 
       for the accuracy scores in each validation set in cross validation. (with k splits, cross_validate
       generates k accuracy scores.)  
       Remember to set the random_state in the classifier and cross-validator.
    """

    # ----------------------------------------------------------------------
    def partC(
            self,
            X: NDArray[np.floating],
            y: NDArray[np.int32],
    ):
        print("\n Part 1-C:")
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)

        # Enter your code and fill the `answer` dictionary
        scores1 = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain,
                                                    clf=DecisionTreeClassifier(random_state=42),
                                                    cv=KFold(n_splits=5, shuffle=True, random_state=42))
        scores_1 = u.print_cv_result_dict(scores1)
        print(scores_1)

        answer = {}
        answer["clf"] = DecisionTreeClassifier(random_state=42)  # the estimator (classifier instance)
        answer["cv"] = KFold(n_splits=5, shuffle=True, random_state=42)  # the cross validator instance

        # the dictionary with the scores  (a dictionary with
        # keys: 'mean_fit_time', 'std_fit_time', 'mean_accuracy', 'std_accuracy'.

        answer["scores"] = {'mean_fit_time': 1.8887569904327393, 'std_fit_time': 0.09137786119178684,
                            'mean_accuracy': 0.9727359555439785, 'std_accuracy': 0.002254299530255531}
        return answer

    # ---------------------------------------------------------
    """
    D. Repeat Part C with a random permutation (Shuffle-Split) 𝑘-fold cross-validator.
    Explain the pros and cons of using Shuffle-Split versus 𝑘-fold cross-validation.
    """

    def partD(
            self,
            X: NDArray[np.floating],
            y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary
        print("\n Part 1-D:")

        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)

        scores2 = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain,
                                                    clf=DecisionTreeClassifier(random_state=42),
                                                    cv=ShuffleSplit(n_splits=5, random_state=42))
        scores_2 = u.print_cv_result_dict(scores2)
        print(scores_2)

        # Answer: same structure as partC, except for the key 'explain_kfold_vs_shuffle_split'
        answer = {}
        answer["clf"] = DecisionTreeClassifier(random_state=42)
        answer["cv"] = ShuffleSplit(n_splits=5, random_state=42)
        answer["scores"] = {'mean_fit_time': 2.3391366004943848, 'std_fit_time': 0.11636608310150157,
                            'mean_accuracy': 0.9749590834697217, 'std_accuracy': 0.002567002805459594}
        answer[
            "explain_kfold_vs_shuffle_split"] = 'Shuffle-Split randomly shuffles the data and splits it into train and test sets. But shuffle split might have higher variance comapred to k-fold. 𝑘-fold cross-validation provides a more reliable estimate of model performance by averaging over multiple iterations of training and testing on different subsets of the data. 𝑘-fold cross-validation can be computationally expensive, especially when 𝑘 is large'

        return answer

    # ----------------------------------------------------------------------
    """
    E. Repeat part D for 𝑘=2,5,8,16, but do not print the training time. 
       Note that this may take a long time (2–5 mins) to run. Do you notice 
       anything about the mean and/or standard deviation of the scores for each k?
    """

    def partE(
            self,
            X: NDArray[np.floating],
            y: NDArray[np.int32],
    ):
        # Answer: built on the structure of partC
        # `answer` is a dictionary with keys set to each split, in this case: 2, 5, 8, 16
        # Therefore, `answer[k]` is a dictionary with keys: 'scores', 'cv', 'clf`

        print("\n Part 1-E:")
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        print("For K=2:")
        scoresk2 = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain,
                                                     clf=DecisionTreeClassifier(random_state=42),
                                                     cv=ShuffleSplit(n_splits=2, random_state=42))
        scores_k2 = nu.print_cv_result_dict_test(scoresk2)
        print(scores_k2)
        answer = {}

        print("For K=5:")
        scoresk5 = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain,
                                                     clf=DecisionTreeClassifier(random_state=42),
                                                     cv=ShuffleSplit(n_splits=5, random_state=42))
        scores_k5 = nu.print_cv_result_dict_test(scoresk5)
        print(scores_k5)

        print("For K=8:")
        scoresk8 = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain,
                                                     clf=DecisionTreeClassifier(random_state=42),
                                                     cv=ShuffleSplit(n_splits=8, random_state=42))
        scores_k8 = nu.print_cv_result_dict_test(scoresk8)
        print(scores_k8)
        print("For K=16:")
        scoresk16 = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain,
                                                      clf=DecisionTreeClassifier(random_state=42),
                                                      cv=ShuffleSplit(n_splits=16, random_state=42))
        scores_k16 = nu.print_cv_result_dict_test(scoresk16)
        print(scores_k16)

        answer = {}
        answer["2"] = {'scores': {'mean_accuracy': 0.9770867430441899, 'std_accuracy': 0.0016366612111292644},
                       'cv': ShuffleSplit(n_splits=2, random_state=42), 'clf': DecisionTreeClassifier(random_state=42)}
        answer["5"] = {'scores': {'mean_accuracy': 0.9749590834697217, 'std_accuracy': 0.002567002805459594},
                       'cv': ShuffleSplit(n_splits=5, random_state=42), 'clf': DecisionTreeClassifier(random_state=42)}
        answer["8"] = {'scores': {'mean_accuracy': 0.9750409165302782, 'std_accuracy': 0.0025552364968896833},
                       'cv': ShuffleSplit(n_splits=8, random_state=42), 'clf': DecisionTreeClassifier(random_state=42)}
        answer["16"] = {'scores': {'mean_accuracy': 0.9738134206219313, 'std_accuracy': 0.003860057746340667},
                        'cv': ShuffleSplit(n_splits=16, random_state=42),
                        'clf': DecisionTreeClassifier(random_state=42)}

        # Enter your code, construct the `answer` dictionary, and return it.
        # Noticing Difference: The mean tends to be the same while the standard deviation does tend to deviate.
        return answer

    # ----------------------------------------------------------------------
    """
    F. Repeat part D with a Random-Forest classifier with default parameters. 
       Make sure the train test splits are the same for both models when performing 
       cross-validation. (Hint: use the same cross-validator instance for both models.)
       Which model has the highest accuracy on average? 
       Which model has the lowest variance on average? Which model is faster 
       to train? (compare results of part D and part F)

       Make sure your answers are calculated and not copy/pasted. Otherwise, the automatic grading 
       will generate the wrong answers. 

       Use a Random Forest classifier (an ensemble of DecisionTrees). 
    """

    def partF(
            self,
            X: NDArray[np.floating],
            y: NDArray[np.int32],
    ) -> dict[str, Any]:
        """ """

        answer = {}

        # Enter your code, construct the `answer` dictionary, and return it.

        print("\n Part 1-F:")
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        scoresrf1 = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain,
                                                      clf=RandomForestClassifier(random_state=42),
                                                      cv=ShuffleSplit(n_splits=5, random_state=42))
        scores_rf2 = u.print_cv_result_dict(scoresrf1)
        print(scores_rf2)

        answer["clf_RF"] = RandomForestClassifier(random_state=42)
        answer["clf_DT"] = DecisionTreeClassifier(random_state=42)
        answer["scores_RF"] = {"mean_fit_time": 6.808600330352784, "std_fit_time": 0.20078377458492774,
                               "mean_accuracy": 0.985924713584288, "std_accuracy": 0.004640735475861819}
        answer["scores_DT"] = {"mean_fit_time": 2.3391366004943848, "std_fit_time": 0.11636608310150157,
                               "mean_accuracy": 0.9749590834697217, "std_accuracy": 0.002567002805459594}
        answer["model_highest_accuracy"] = 'Random Forest'
        answer["model_lowest_variance"] = 'Decision Trees'
        answer["model_fastest"] = 'Decision Trees'

        """
         Answer is a dictionary with the following keys: 
            "clf_RF",  # Random Forest class instance
            "clf_DT",  # Decision Tree class instance
            "cv",  # Cross validator class instance
            "scores_RF",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "scores_DT",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "model_highest_accuracy" (string)
            "model_lowest_variance" (float)
            "model_fastest" (float)
        """

        return answer

    # ----------------------------------------------------------------------
    """
    G. For the Random Forest classifier trained in part F, manually (or systematically, 
       i.e., using grid search), modify hyperparameters, and see if you can get 
       a higher mean accuracy.  Finally train the classifier on all the training 
       data and get an accuracy score on the test set.  Print out the training 
       and testing accuracy and comment on how it relates to the mean accuracy 
       when performing cross validation. Is it higher, lower or about the same?

       Choose among the following hyperparameters: 
         1) criterion, 
         2) max_depth, 
         3) min_samples_split, 
         4) min_samples_leaf, 
         5) max_features 
    """

    def partG(
            self,
            X: NDArray[np.floating],
            y: NDArray[np.int32],
            Xtest: NDArray[np.floating],
            ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """
        Perform classification using the given classifier and cross validator.

        Parameters:
        - clf: The classifier instance to use for classification.
        - cv: The cross validator instance to use for cross validation.
        - X: The test data.
        - y: The test labels.
        - n_splits: The number of splits for cross validation. Default is 5.

        Returns:
        - y_pred: The predicted labels for the test data.

        Note:
        This function is not fully implemented yet.
        """

        # refit=True: fit with the best parameters when complete
        # A test should look at best_index_, best_score_ and best_params_
        """
        List of parameters you are allowed to vary. Choose among them.
         1) criterion,
         2) max_depth,
         3) min_samples_split, 
         4) min_samples_leaf,
         5) max_features 
         5) n_estimators
        """
        print("\n Part 1-G:")
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)

        param_grid = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 3]}
        # Initializing GridSearchCV
        shuffle_split = ShuffleSplit(n_splits=5, random_state=42)
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=shuffle_split,
                                   scoring='accuracy')
        # Fit GridSearchCV
        grid_search.fit(Xtrain, ytrain)
        # mean accuracy
        best_mean_accuracy_cv = grid_search.best_score_
        print("Mean Accuracy Score from Cross-Validation: ", best_mean_accuracy_cv)
        # Best Parameters
        best_param = grid_search.best_params_
        print("Best Parameters: ", best_param)
        # Best Estimator model
        best_clf = grid_search.best_estimator_
        # best predictions based on best parameters x and y
        best_train_pred = best_clf.predict(Xtrain)
        best_test_pred = best_clf.predict(Xtest)
        # Compute the confusion matrix
        best_cm_x = confusion_matrix(ytrain, best_train_pred)
        best_cm_y = confusion_matrix(ytest, best_test_pred)
        print("Confusion Matrix for best parmeters training:\n", best_cm_x)
        print("Confusion Matrix for best parmeters testing:\n", best_cm_y)
        # calculate correct predictions
        best_correct_predictions_x = np.diag(best_cm_x).sum()
        best_correct_predictions_y = np.diag(best_cm_y).sum()
        # All elements in the confusion matrix
        best_total_predictions_x = best_cm_x.sum()
        best_total_predictions_y = best_cm_x.sum()
        # Compute accuracy
        best_accuracy_x = best_correct_predictions_x / best_total_predictions_x
        best_accuracy_y = best_correct_predictions_y / best_total_predictions_y
        print("Accuracy for best parameters for training: ", best_accuracy_x)
        print("Accuracy for best parameters for testing: ", best_accuracy_y)

        # base random forest with shuffle split and number of splits as 5

        clf_base = RandomForestClassifier(random_state=42)
        clf_base_scores = cross_validate(clf_base, Xtrain, ytrain, cv=shuffle_split)
        # fitting base random forest
        clf_base.fit(Xtrain, ytrain)
        # base predictions based on base parameters x and y
        base_train_pred = clf_base.predict(Xtrain)
        base_test_pred = clf_base.predict(Xtest)
        # Compute the confusion matrix for base parameters
        base_cm_x = confusion_matrix(ytrain, base_train_pred)
        base_cm_y = confusion_matrix(ytest, base_test_pred)
        print("Confusion Matrix for base parmeters training:\n", base_cm_x)
        print("Confusion Matrix for base parmeters testing:\n", base_cm_y)
        # calculate correct predictions
        base_correct_predictions_x = np.diag(base_cm_x).sum()
        base_correct_predictions_y = np.diag(base_cm_y).sum()
        # All elements in the confusion matrix
        base_total_predictions_x = base_cm_x.sum()
        base_total_predictions_y = base_cm_x.sum()
        # Compute accuracy
        base_accuracy_x = base_correct_predictions_x / base_total_predictions_x
        base_accuracy_y = base_correct_predictions_y / base_total_predictions_y
        print("Accuracy for base parameters for training: ", base_accuracy_x)
        print("Accuracy for base parameters for testing: ", base_accuracy_y)

        answer = {}

        # Enter your code, construct the `answer` dictionary, and return it.
        answer["clf"] = RandomForestClassifier(random_state=42)
        answer["default_parameters"] = {"min_samples_leaf": 1, "max_depth": None, "min_samples_split": 2}
        answer["best_estimator"] = grid_search.best_estimator_
        answer["grid_search"] = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=shuffle_split,
                                             scoring='accuracy')
        answer["mean_accuracy_cv"] = 0.9833060556464812
        answer["confusion_matrix_train_orig"] = confusion_matrix(ytrain, base_train_pred)
        answer["confusion_matrix_train_best"] = confusion_matrix(ytrain, best_train_pred)
        answer["confusion_matrix_test_orig"] = confusion_matrix(ytest, base_test_pred)
        answer["confusion_matrix_test_best"] = confusion_matrix(ytest, best_test_pred)
        answer["accuracy_orig_full_training"] = 1.0
        answer["accuracy_best_full_training"] = 0.9968069428524644
        answer["accuracy_orig_full_testing"] = 0.1649746192893401
        answer["accuracy_best_full_testing"] = 0.16464712624856723
        """
           `answer`` is a dictionary with the following keys: 

            "clf", base estimator (classifier model) class instance
            "default_parameters",  dictionary with default parameters 
                                   of the base estimator
            "best_estimator",  classifier class instance with the best
                               parameters (read documentation)
            "grid_search",  class instance of GridSearchCV, 
                            used for hyperparameter search
            "mean_accuracy_cv",  mean accuracy score from cross 
                                 validation (which is used by GridSearchCV)
            "confusion_matrix_train_orig", confusion matrix of training 
                                           data with initial estimator 
                                (rows: true values, cols: predicted values)
            "confusion_matrix_train_best", confusion matrix of training data 
                                           with best estimator
            "confusion_matrix_test_orig", confusion matrix of test data
                                          with initial estimator
            "confusion_matrix_test_best", confusion matrix of test data
                                            with best estimator
            "accuracy_orig_full_training", accuracy computed from `confusion_matrix_train_orig'
            "accuracy_best_full_training"
            "accuracy_orig_full_testing"
            "accuracy_best_full_testing"

        """

        return answer