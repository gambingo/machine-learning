Updated Oct. 20, 2017

## Classification Algorithms
This document is intended for review and to spark discussion. I do not recommend trying to memorize what is below. This list is not a substitute understanding the foundations of each algorithm.

#### General Advice For All Models
* Whenever you are optimizing on distance, you must standardize your data.
* Your models will have more predictive power if you bucket continuous data.*
* You will need to either drop or impute missing data.*
* Beware the Curse of Dimensionality.

*Except with tree based algorithms.

#### K-Nearest Neighbors
##### Assumptions
* Standardized Data

##### Pros
* Very interpretable
* Inherently multi-class

##### Cons
* Limited Complexity
* Does not scale well with large numbers of features or observations
* Does not handle diverse (non-homogenous) features

##### Terminology
* Curse of Dimensionality

##### Hyperparameters
* k

##### Sklearn Tips
* `n_jobs = -1`

#### Logistic Regression
##### Assumptions
##### Pros
* Interpretable
* Scales well with large numbers of features and observations

##### Cons
* Not inherently multi-class
* Does not inherently handle non-homogenous features

##### Terminology
* Sigmoid Function
* Log Odds

##### Hyperparameters
* Regularization Strength
* Learning Rate

##### Sklearn Tips
* By default, `LogisticRegression()` has regularization turned on.

#### Support Vector Machines
##### Assumptions
##### Pros
* Interpretable
* Scales with large amounts of data

##### Cons
* Not inherently multi-class; must do one vs. rest (one vs. all)
* Does not scale with large number of features (curse of dimensionality)
* Does not inherently handle non-homogenous features

##### Terminology
* Hinge Loss
* C
* Linearly Separable
* Kernel Trick
* Support Vectors
* Hyperplane
* RBF Kernel
* Margin

##### Hyperparameters
* C: Budget for Misclassification
* Learing Rate

##### Sklearn Tips
* When choosing a linear kernel, use `LinearSVC()` over `SVC(kernel = 'linear')`. The former is much faster.

#### Decision Trees
##### Assumptions
* Stationary Data

##### Pros
* Easily Interpretable
* Scale Invariant
* Can handle missing data
* Inherently multiclass
* Naturally includes feature interactions
* No coefficients
* Robust to outliers
* Inherently handles non-homogenous data

##### Cons
* Prone to overfitting
* Simplistic
* Does not scale well with large numbers of feautures.
* Slow with large amounts of data (use XGBoost)

##### Terminology
* Entropy, Gini
* Bit
* Information Gain

##### Sklearn Tips

#### Random Forest
##### Assumptions
* Stationary Data

##### Pros
* Relatively strong on complex data
* Outputs feature importance
* No need for cross-validation (in fact, you should not use cross-validation)
* Highly parallel
* Inherently multi-class
* Scales well with number of features as each tree chooses a different subset of features
* Better results than Decision Trees

##### Cons
* Difficult to interpret

##### Terminology
* Bagging, Out-Of-Bag Estimates
* Greedy Algorithm
* Ensemble

##### Notes
* Do not use cross-validation with a random forest model. Bagging inherently leaves a test set. If you use cross-validation, you won't be consuming the entire training set. Use and out-of-bag estimate instead.

##### Sklearn Tips
* `n_jobs = -1`
* Do not use `GridSearchCV()` or `cross_val_score()`
* The out-of-bag estimate only scores on accuracy. If you want to score on precision, recall, or somehting else, you'll need to write your own verson.

#### Naive Bayes
##### Assumptions
* Independent Features

##### Pros
* Fast
* Performs well even when you break assumptions
* Works well with limited data (Good place to start with small datasets)
* Inherently multi-class
* Handles non-homogenous data

##### Cons
* Limited Complexity
* Does not scale well with large numbers of features or observations

##### Terminology
* Laplace Smoothing
* Underflow/Overflow
* Gaussian Naive Bayes
* Bernoulli Naive Bayes
* Multinomial Naive Bayes

#### Neural Networks
##### Assumptions
##### Pros
##### Cons
##### Terminology
* Activation Function
* Feed Forward
* Backpropagation
* Step Function
* Sigmoid Function
* Linear Function
* Rectified Linear Unit
* Hyperbolic Tangent 
* Deep Neural Network
* Fully Connected Network

##### Sklearn Tips