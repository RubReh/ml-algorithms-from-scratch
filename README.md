# Machine learning algorithms from scratch
In order to understand the underlying functionality of some common machine learning algorithms I dedicated some time to implement a few common ML-algorithms from scratch witht the goal of using as few abstractions as possible.


## K means clustering

![](https://github.com/RubReh/ml-algorithms-from-scratch/blob/master/images/kmeanscluster.png)

## Linear Regression

![](https://github.com/RubReh/ml-algorithms-from-scratch/blob/master/images/linear_regression.png)


## Multiple linear regression

![](https://github.com/RubReh/ml-algorithms-from-scratch/blob/master/images/multiple_regression_error.png)

## Naive Bayes


## Support Vector Machine
![](https://github.com/RubReh/ml-algorithms-from-scratch/blob/master/images/Overview_svm.png)


### Deriving the math
With support vector machines we're trying to maximize the width of the lane that separates the + and - samples as seen in the image.
We start by choosing a vector w that is a normal vector to the separating line / (or hyperplane in an arbitrary dimension depending on how
many features the feature vector x holds.)

We can derive an expression for the width using the vectors on the boundary of the street like so:


![](https://github.com/RubReh/ml-algorithms-from-scratch/blob/master/images/equation_1.png)

This comes from the fact that the dot product projects the vector x+ -  x- on a vector that is normal to the separation line.
w is chosen to be such a vector, but we normalize it in order to get a unit vector in the direction of w. The result is the width 
of the street. This is the expression we want to maximize. But we have other constraints that must hold.

Let's say we have a sample x. The rule we use to decide if that it is a + sample is:


w dot_product x + b >= 0. Again this comes from the fact that w is a normal vector to the line and if we project the feature vector onto
w we get a length that is either larger than some constant b or smaller than it.

Furthermore, we decide that for a plus sample and minus sample, this holds:

![](https://github.com/RubReh/ml-algorithms-from-scratch/blob/master/images/equation_2.png)

To merge these two equations into something more condense, we multiply the left side of the equations by yi, such
that yi is -1 for a minus sample, and +1 for a - sample.

Thus we get the two equations:


![](https://github.com/RubReh/ml-algorithms-from-scratch/blob/master/images/equation_3.png)


Now we see that given the properties of yi for each sample type we get that a common rule rule holds for a given x sample xi:
We can also move the 0 to the left side giving us: 


![](https://github.com/RubReh/ml-algorithms-from-scratch/blob/master/images/equation_4.png)

Moreover we say that if a sample is on the boundary of the street (the support vectors that is) we say that (4)
is exactly 0 which we'll use later.

Now we use y's properties and (5) in (1). If x is a plus (minus) sample yi = 1 (-1) which gives that:

![](https://github.com/RubReh/ml-algorithms-from-scratch/blob/master/images/equation_5.png)

Put this into (1) and we get that the width we're after can be expressed as:

![](https://github.com/RubReh/ml-algorithms-from-scratch/blob/master/images/equation_6.png)


The reframing of what we minimize is used to simplify the derivatives later on by the way (you'll see this later).
We now have a minimization problem under some constraints. This calls for La Grange multipliers giving us the
following La Grange function:

![](https://github.com/RubReh/ml-algorithms-from-scratch/blob/master/images/equation_7.png)

This is close to an expression we can ge the gradient of with respect to w and b and use stochastic gradient descent on.

### How should we update the SGD?

We need to take account of two things now. The first term in L is trying to minimize || w || and the second term 
keeps track of our misclassification error. From our decision rule we see that if a sample is correctly
classified (yi*(w dot_product xi + b) - 1) becomes 0 or less. This means that the classification error is zero. 
Thus, if this expression evaluates to 0 or less we pin it to 0 since no error has occured for that classification.

However, if (yi*(w dot_product xi + b) - 1) is larger than 0 it means it's misclassified. Then we use it to update
our w and b using SGD.

So we get two different gradients to use in SGD depending on how a sample i classified

 
 ![](https://github.com/RubReh/ml-algorithms-from-scratch/blob/master/images/equation_8.png)
 
 and
 
 
 ![](https://github.com/RubReh/ml-algorithms-from-scratch/blob/master/images/equation_9.png)
 

 Math source courtesy of MIT.
 https://www.youtube.com/watch?v=_PwhiWxHK8o


 



   

