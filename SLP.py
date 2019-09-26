

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


# # generating the weight tensor  to 0




weight_vector = []
for i in range(len(training_data_input[0])+1):
    weight_vector.append(0)


# # defining the forward function which takes the input vector and return output




def forwardpass(input_vector):
    
    input_vector.insert(0,1)
    
    output = np.dot(weight_vector,np.transpose(input_vector))
    
    if output  > 0.5:
        output_final = 1
    else:
        output_final = 0
        
   
        
    return input_vector,output_final


# # calulating delta weight and assume R as 1 class and M as 0 class



def deltaweight(input_vector,output_final,iteration):
    
    if training_data_output[iteration] == "R":
        desired_output = 1
    else:
        desired_output = 0
        
    error = (desired_output - output_final)
    
    delta_weight = []
    for i in range(len(input_vector)):
        delta_weight.append((input_vector[i]*error))
        
    return delta_weight


# # defining function to update the weight




def updation(weight_vector_pre,delta_weight):
    return np.add(weight_vector_pre,delta_weight)


# # training with 300 epoch








for epoch in range(300):

    for input_vector_number in range(len(training_data_input)):
        
        input_vector,output_final = forwardpass(list(training_data_input[input_vector_number]))
        
        delta_weight = deltaweight(list(input_vector),output_final,input_vector_number)
        
        weight_vector = updation(list(weight_vector),delta_weight) 


 


# accuracy measurment in training

counter = 0
for input_vector_number in range(len(training_data_input)):
    input_vector,output_final = forwardpass(list(training_data_input[input_vector_number]))
    
    if output_final == 1 and training_data_output[input_vector_number] == "R":
        counter = counter + 1
    elif output_final == 0 and training_data_output[input_vector_number] == "M":
        counter = counter + 1
print(counter," out of ",len(training_data_input)," are correctly predicted")
print("accuracy in training set = ",(counter/len(training_data_input))*100,"%")



# accuracy measurment in testing
counter = 0
for input_vector_number in range(len(testing_data_input)):
    input_vector,output_final = forwardpass(list(testing_data_input[input_vector_number]))
    
    if output_final == 1 and testing_data_output[input_vector_number] == "R":
        counter = counter + 1
    elif output_final == 0 and testing_data_output[input_vector_number] == "M":
        counter = counter + 1
print(counter," out of ",len(testing_data_input)," are correctly predicted")
print("accuracy in testing set= ",(counter/len(testing_data_input))*100,"%")




