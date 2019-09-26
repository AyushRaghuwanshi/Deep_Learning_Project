data set description ============
This is the data set used by Gorman and Sejnowski in their study
of the classification of sonar signals using a neural network [1].  The
task is to train a network to discriminate between sonar signals bounced
off a metal cylinder and those bounced off a roughly cylindrical rock.
The label associated with each record contains the letter "R" if the object
is a rock and "M" if it is a mine (metal cylinder).



network description ============
we use 0.5 as a hard limiter 
if output > 0.5 it will be of class 1 otherwise 0.

weight matrix deimesion = 1X61(one for bias)

accuracy after tarining = 
	on training set = 90% (avg)
	on testing set = 76% (avg)

we assumed "R" as class 1 and "M" as class 0


function description ======
forwardpass = the forward function which takes the input vector and return output

deltaweight = calculate delta weight and assume R as 1 class and M as 0 class

updation = function to update the weight

