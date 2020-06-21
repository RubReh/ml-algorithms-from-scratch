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

(x+ - x-) dot_product w / || w ||. (1)

This comes from the fact that the dot product projects the vector x+ -  x- on a vector that is normal to the separation line.
w is chosen to be such a vector, but we normalize it in order to get a unit vector in the direction of w. The result is the width 
of the street. This is the expression we want to maximize. But we have other constraints that must hold.

Let's say we have a sample x. The rule we use to decide if that it is a + sample is:
w dot_product x + b >= 0. Again this comes from the fact that w is a normal vector to the line and if we project the feature vector onto
w we get a length that is either larger than some constant b or smaller than it.

Furthermore, we decide that for a plus sample, x+, this holds:

w dot_product x+ + b >= 1

And for a minus sample this holds:

w dot_product x- + b <= -1 


To merge these two equations into something more condense, we multiply the left side of the equations by yi, such
that yi is -1 for a minus sample, and +1 for a - sample.

Thus we get the two equations:

yi*(w dot_product x- + b) <= -1
yi*(w dot_product x+ + b) >= 1

Now we see that given the properties of yi for each sample type we get that a common rule rule holds for a given x sample xi:

yi*(w dot_product xi + b) >=1 (3)

We can move the 0 to the left side giving us: 

yi*(w dot_product xi + b) -1 >=0 (4)

Moreover we say that if a sample is on the boundary of the street (the support vectors that is) we say that (4)
is exactly 0. So if xi on boundary:

yi*(w dot_product xi + b) -1 =0 (5)

Now we use yi:s properties and (5) in (1). If x is a plus (minus) sample yi = 1 (-1) which gives that:

w dot_product x+  = 1 - b

and w dot_product x- = 1 + b

Put this into (1) and we get that the width of the lane is:

seperation_width = 2 / || w ||. 

This is what we want to maximize. Which means that we want to minimize || w ||. Which means that we
want to minimize (1/2)/ || w || ^2

The reframing of what we minimize is used to simplify the derivatives later on by the way (you'll see).
We now have a minimization problem under some constraints. We can say:

We want to minimize (1/2)/ || w || ^ 2 under the constraint that yi*(w dot_product xi + b) -1 = 0. This calls for La Grange multipliers giving us the
following La Grange function:

L = (1/2)/ || w || ^ 2 - sum[lambai * (yi*(w dot_product xi + b) - 1)]

This is close to an expression we can ge the gradient of with respect to w and b and use stochastic gradient descent on.

### How should we update the SGD?

L = (1/2)/ || w || ^ 2 - sum[lambai * (yi*(w dot_product xi + b) - 1)]

We need to take account of two things now. The first term in L is trying to minimize || w || and the second term 
keeps track of our misclassification error. From our decision rule we see that if a sample is correctly
classified (yi*(w dot_product xi + b) - 1) becomes 0 or less. This means that the classification error is zero. 
Thus, if this expression evaluates to 0 or less we pin it to 0 since no error has occured for that classification.

However, if (yi*(w dot_product xi + b) - 1) is larger than 0 it means it's misclassified. Then we use it to update
our w and b using SGD.

So we get two different gradients to use in SGD:

grad(L) = w if  (yi*(w dot_product xi + b) - 1) <= 0

or grad(L) = w + lambai * yi*xi
 


 Math source courtesy of MIT.
 https://www.youtube.com/watch?v=_PwhiWxHK8o


 



   

