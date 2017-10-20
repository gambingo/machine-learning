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
##### Sklearn Tips

#### Support Vector Machines
##### Assumptions
##### Pros
##### Cons
##### Terminology
##### Sklearn Tips
* When choosing a linear kernel, use `LinearSVC()` over `SVC(kernel = 'linear')`. The former is much faster.

#### Decision Trees
##### Assumptions
##### Pros
##### Cons
##### Terminology
##### Sklearn Tips

#### Random Forest
##### Assumptions
##### Pros
##### Cons
##### Terminology
##### Sklearn Tips

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
##### Sklearn Tips