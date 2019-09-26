data set description ============
This is the data set used by Gorman and Sejnowski in their study
of the classification of sonar signals using a neural network [1].  The
task is to train a network to discriminate between sonar signals bounced
off a metal cylinder and those bounced off a roughly cylindrical rock.
The label associated with each record contains the letter "R" if the object
is a rock and "M" if it is a mine (metal cylinder).



neural network description ===========

size of input vector = 60 + 1 = 61 ( including bias).

number of hidden layer = 1

number of neuron in hidden layer = 24
dimensions of weight vector for hidden layer = 24 X 61

number of neuron in output layer = 2
dimensions of weight vector for output  layer = 2 X 25

here we use logistic function as activation function 

data in training = 104
data in testing = 104

number of epoch = 300

accuracy after training the network =
	on training set = 100%
	on testing set = 80%(avg)

function description =================== 

activation = it is logistic function which takes input x and give f(x) as return value where f() is a logistic function .

forwardpass = the forward function which takes the input vector and return three vector including output vector which we will use in backpropogation .

backpropogation = the backpropogation funtion which will take the input vector with bias as one of the input , output vector which comes from hidden layer with bias as one of the input and output vector which comes from output layer and return the delta weight matices for output and hidden layer 

weightupdation = weightupdation function which will take the previous weight matrices and delta weigh matrices and update the previous weight matrices