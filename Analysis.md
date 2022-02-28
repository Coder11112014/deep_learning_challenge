# INTRO

In this Notebook file I build, train, test, and optimize a deep neural network that models charity success from nine features in a loan application data set.

I employ the TensorFlow Keras Sequential model with Dense hidden layers and a binary classification output layer and optimize the model

# DATA Processing

I preprocessed the data set charity_data.csv by reading our data and noting the target, feature, and identification variables:

Target Variable: IS_SUCCESSFUL

Feature Variables:

APPLICATION_TYPE,
AFFILIATION,
CLASSIFICATION,
USE_CASE,
ORGANIZATION,
STATUS,
INCOME_AMT,
SPECIAL_CONSIDERATIONS,
ASK_AMT

Removed Variables:
EIN,
NAME

Encode categorical variables using sklearn.preprocessing. Noticed APPLICATION_TYPE and CLASSIFICATION with many unique values, after encoding them, we split our data into the target and features, split further into training and testing sets, and scale our training and testing data using sklearn.preprocessing.StandardScaler.

# Compiling, Training, and Evaluating the Model

Used two layer baseline model numbers because it was close to the number of features.
Model: "sequential"

---

###### Layer (type)                Output Shape              Param #

 dense (Dense)               (None, 50)                2750

 dense_1 (Dense)             (None, 70)                3570

 dense_2 (Dense)             (None, 1)                 71

=================================================================
Total params: 6,391
Trainable params: 6,391
Non-trainable params: 0

---

###### model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")



```
268/268 - 0s - loss: 1.0040 - accuracy: 0.7402 - 251ms/epoch - 937us/step
Loss: 1.0040357112884521, Accuracy: 0.7401749491691589
```


I couldnt achieve the target model performance with multiple attempts, I could not get the accuracy above 74%. With more time, I believe I could get the desired results.

To try and increase model performance I tried decreasing the amount of features, increasing the number of epochs, different activation types, and increased the number of nodes.

# Analysis

The loss and accuracy seem to be relational barring a few outliers. With more data or possible tweaks with current data, we could generate more models to achieve the score we are looking for. I would attempt to add more layers, increase the number of nodes, and add a large number of epochs in order to achieve a higher accuracy score.
