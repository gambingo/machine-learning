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

##### Cons
* Limited Complexity

##### Terminology
* Curse of Dimensionality

##### Hyperparameters
* k

##### Sklearn Tips
* `n_jobs = -1`

#### Logistic Regression
##### Assumptions
##### Pros
##### Cons
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
##### Cons
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

##### Cons
* Prone to overfitting
* Simplistic
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

##### Cons
* Difficult to interpret

##### Terminology
* Bagging, Out-Of-Bag Estimates
* Greedy Algorithm
* Ensemble

##### Sklearn Tips
* `n_jobs = -1`

#### Naive Bayes
##### Assumptions
* Independent Features

##### Pros
* Fast
* Performs well even when you break assumptions
* Works well with limited data (Good place to start with small datasets)

##### Cons
* Limited Complexity

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