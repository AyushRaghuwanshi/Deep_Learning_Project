

# # importing libraries




import pandas as pd
import random
import numpy as np
import math
import matplotlib.pyplot as plt


# # import the data set



full_data = pd.read_csv("sonar.csv",header = None)


# # converting this dataframes into list




data = []
data_complete = []
for i in range(208):
    for j in range(61):
        data.append(full_data.loc[i][j])
    data_complete.append(data)
    data = []


# # shuffling the dataset




random.shuffle(data_complete)


# # we are using random 104 rows for testing and other for taining




testing_data = []
for i in range(104):
    ran = random.randrange(0,208-i)
    testing_data.append(data_complete.pop(ran))
training_data = data_complete


# # here we  are seprating training and testing input data vector and output data vector




training_data_input = []
training_data_output = []
for i in range(len(training_data)):
    training_data_input.append(training_data[i][0:60])
    training_data_output.append(training_data[i][60])
    
testing_data_input = []
testing_data_output = []
for i in range(len(testing_data)):
    testing_data_input.append(testing_data[i][0:60])
    testing_data_output.append(testing_data[i][60])


# # generating the weight tensor randomly in the range -.3 to .3 and declaring the number of neuron in hidden layer and output layer




neuron_in_hiddenlayer = 24
neuron_in_outputlayer = 2
weight_matrix_hiddenlayer = []
weight_matrix_outputlayer = []
temp = []
for i in range(neuron_in_hiddenlayer):
    for j in range(len(training_data_input[0])+1):
        temp.append(random.uniform(-1,1))
    weight_matrix_hiddenlayer.append(temp)
    temp = []
    
for i in range(neuron_in_outputlayer):
    for j in range(neuron_in_hiddenlayer+1):
        temp.append(random.uniform(-1,1))
    weight_matrix_outputlayer.append(temp)
    temp = []


# # defining the activation function (logistic function)




def activation(input_value):
    return 1.0 / (1.0 + math.exp(-input_value))


# # defining the forward function which takes the input vector and return three vector including output vector which we will use in backpropogation 




def forwardpass(input_vector):
    #inserting a 1 as bias
    input_vector.insert(0,1)
    #input to hidden layer
    input_vector_to_hidden = []
    for i in range(neuron_in_hiddenlayer):
        input_vector_to_hidden.append(np.dot(weight_matrix_hiddenlayer[i],np.transpose(input_vector)))
    
    
    #output from hidden by applying activation function
    output_vector_from_hidden = []
    for i in range(neuron_in_hiddenlayer):
        output_vector_from_hidden.append(activation(input_vector_to_hidden[i]))
    
    #inserting a 1 as bias
    output_vector_from_hidden.insert(0,1)
    
    #input to output layer
    input_vector_to_output = []
    for i in range(neuron_in_outputlayer):
        input_vector_to_output.append(np.dot(weight_matrix_outputlayer[i],np.transpose(output_vector_from_hidden)))
    
    
    #output from output by applying activation function
    output_vector_from_output = []
    for i in range(neuron_in_outputlayer):
        output_vector_from_output.append(activation(input_vector_to_output[i]))
    
    return (input_vector,output_vector_from_hidden,output_vector_from_output)


# # defining the backpropogation funtion which will take the input vector with bias as one of the input , output vector which comes from hidden layer with bias as one of the input and output vector which comes from output layer and return the delta weight matices for output and hidden layer 




def backpropogation(input_vector,output_vector_from_hidden,output_vector_from_output,iteration):
    if training_data_output[iteration] == 'R':
        desired_output = [1,0]
    else:
        desired_output = [0,1]
    
    local_gradient_outputlayer = []
    for i in range(neuron_in_outputlayer):
        local_gradient_outputlayer.append((desired_output[i]-output_vector_from_output[i])*(output_vector_from_output[i])*(1-output_vector_from_output[i]))
    
    local_gradient_hiddenlayer = []
    for i in range(1,neuron_in_hiddenlayer+1):
        weighted_sum_of_local_gradient_outputlayer = 0
        for j in range(neuron_in_outputlayer):
            weighted_sum_of_local_gradient_outputlayer = weighted_sum_of_local_gradient_outputlayer +             weight_matrix_outputlayer[j][i]*local_gradient_outputlayer[j]
        local_gradient_hiddenlayer.append((output_vector_from_hidden[i])*(1-output_vector_from_hidden[i])*weighted_sum_of_local_gradient_outputlayer)
        
    delta_weight_outputlayer = []
    for i in range(neuron_in_outputlayer):
        delta_weight_row = []
        for j in range(neuron_in_hiddenlayer+1):
            delta_weight_row.append(2*output_vector_from_hidden[j]*local_gradient_outputlayer[i])
        delta_weight_outputlayer.append(delta_weight_row)
    
    delta_weight_hiddenlayer = []
    for i in range(neuron_in_hiddenlayer):
        delta_weight_row = []
        for j in range(len(training_data_input[0])+1):
            delta_weight_row.append(2*input_vector[j]*local_gradient_hiddenlayer[i])
        delta_weight_hiddenlayer.append(delta_weight_row)
   
    
    return (delta_weight_outputlayer,delta_weight_hiddenlayer)


# # defining the weightupdation function which will take the previous weight matrices and delta weigh matrices and update the previous weight matrices




def weightupdation(previous_weight_output,previous_weight_hidden,delta_weight_outputlayer,delta_weight_hiddenlayer):
    previous_weight_output = np.add(previous_weight_output, delta_weight_outputlayer) 
    previous_weight_hidden = np.add(previous_weight_hidden, delta_weight_hiddenlayer)
    return (previous_weight_output,previous_weight_hidden)


# # training the network by 300 epochs and calculating e_avg after every epoch


e_avg_of_all_epoch = []
for epoch in range(300):
    e_avg = 0
    for input_vector_number in range(len(training_data_input)):
        
        #calling the forward function
        input_vector,output_vector_from_hidden,output_vector_from_output = forwardpass(list(training_data_input[input_vector_number]))
        
        
        if training_data_output[input_vector_number] == 'R':
            desired_output = [1,0]
        else:
            desired_output = [0,1]
        
        #calculating the lsm error
        e = 0    
        for i in range(neuron_in_outputlayer):
            e = e + (desired_output[i] - output_vector_from_output[i])**2
        e = (e)/2
        e_avg = e_avg + e
        
        # calling the backpropogation function
        delta_weight_outputlayer,delta_weight_hiddenlayer = backpropogation(input_vector,output_vector_from_hidden,output_vector_from_output,input_vector_number)
        
        #calling the weightupdation function
        weight_matrix_outputlayer,weight_matrix_hiddenlayer = weightupdation(weight_matrix_outputlayer, weight_matrix_hiddenlayer,delta_weight_outputlayer, delta_weight_hiddenlayer)
    
    e_avg = e_avg/len(training_data_input)    
    print("E_avg for epoch ",epoch + 1," = ",e_avg)
    e_avg_of_all_epoch.append(e_avg)

epoch_number = list(range(1,301))
plt.plot(epoch_number,e_avg_of_all_epoch)

plt.xlabel('epoch number') 
plt.ylabel('e_avg')

plt.title('e_avg vs epoch number')


plt.savefig('e_avg vs epoch number.png')

plt.plot(epoch_number,e_avg_of_all_epoch)

plt.xlabel('epoch number') 
plt.ylabel('e_avg')

plt.title('e_avg vs epoch number')
plt.show()


# # testing the network




counter = 0
for input_vector_number in range(len(training_data_input)):
    
    #calling the forward function
    input_vector,output_vector_from_hidden,output_vector_from_output = forwardpass(list(training_data_input[input_vector_number]))
    
    if output_vector_from_output[0]>output_vector_from_output[1] and training_data_output[input_vector_number] == 'R':
            counter = counter + 1
    elif output_vector_from_output[0]<output_vector_from_output[1] and training_data_output[input_vector_number] == 'M':
            counter = counter + 1

print(counter," out of ",len(training_data_input)," are correctly predicted")
print("accuracy in training set= ",(counter/len(training_data_input))*100,"%")
 





counter = 0
for input_vector_number in range(len(testing_data_input)):
    
    #calling the forward function
    input_vector,output_vector_from_hidden,output_vector_from_output = forwardpass(list(testing_data_input[input_vector_number]))
    
    if output_vector_from_output[0]>output_vector_from_output[1] and testing_data_output[input_vector_number] == 'R':
            counter = counter + 1
    elif output_vector_from_output[0]<output_vector_from_output[1] and testing_data_output[input_vector_number] == 'M':
            counter = counter + 1

print(counter," out of ",len(testing_data_input)," are correctly predicted")
print("accuracy in testing set= ",(counter/len(testing_data_input))*100,"%")
 

