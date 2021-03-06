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
* Somewhat handles diverse features (must dummify categorical data)

##### Cons
* Limited Complexity
* Does not scale well with large numbers of features or observations

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
* Depending on the implementation, can be inherently multi-class
  * One-vs-All will yield most probable class, but not true probabilities
  * The multinomial implementation will yield true probabilities, at the expense of computation time
* Scales well with large numbers of features and observations
* Somewhat handles diverse features (must dummify categorical data)

##### Cons
* 

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
* Scales with large amounts features and observations
* Does not break down as fast as other distance based models as number of features increase

##### Cons
* Not inherently multi-class; must do one vs. rest (one vs. all)
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
* Does not scale well with large numbers of feautures
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
* Better results than Decision Trees
* Robust to overfitting

##### Cons
* Difficult to interpret
* Slow with large amounts of data (use XGBoost)
* Scales slightly better with the number of features than a decision tree, because each tree only looks at a subset of features. But it is still inefficient with lots of columns.

##### Terminology
* Bagging, Out-Of-Bag Estimates
* Greedy Algorithm
* Ensemble

##### Notes
* Do not use cross-validation with a random forest model. Bagging inherently leaves a test set. If you use cross-validation, you won't be consuming the entire training set. Use and out-of-bag estimate instead.

##### Sklearn Tips
* `n_jobs = -1`
* Do not use `GridSearchCV()` or `cross_val_score()`
* The out-of-bag estimate only scores on accuracy. If you want to score on precision, recall, or something else, you'll need to write your own verson.

#### Naive Bayes
##### Assumptions
* Independent Features

##### Pros
* Fast
* Performs well even when you break assumptions
* Works well with limited data (Good place to start with small datasets)
* Inherently multi-class
* Scales well with lots of features and observations. Computers are really good at counting

##### Cons
* Limited Complexity
* Must match implementaion to datatypes

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
