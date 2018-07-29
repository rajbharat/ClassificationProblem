
install.packages('nnet')
install.packages('caret')
install.packages('RCurl')
install.packages('Metrics')
install.packages('e1071')
install.packages('nnet')
install.packages('NeuralNetTools')
install.packages('RSNNS')
library(nnet)
library(caret)
library(RCurl)
library(Metrics)
library(e1071)
library(RSNNS)
library(nnet)
library(NeuralNetTools)

#Importing the Dataset
spam_data<-read.csv("C:\\Users\\user\\Documents\\spam.csv")

#Splitting the Data into Train and Test
indexed <- sample(1:nrow(spam_data), size = 0.25*nrow(spam_data))
spam_train <- spam_data[-indexed,]
spam_test <- spam_data[indexed,]
spam_test_without_target<-spam_test[,-58] #removing target column for predictiion

#Building the Model
spam_nnet <- nnet(target ~., data = spam_train, size=10, maxit=5000)
spam_nnet_pred <- predict(spam_nnet, spam_test_without_target)
spam_nnet_pred <- round(spam_nnet_pred)
spam_nnet_table <- table(Actual=spam_test$target, Predicted=spam_nnet_pred)
spam_nnet_table

"
      Predicted
Actual   0   1
      0 666  32
      1  42 410
"

spam_accuracy <- sum(diag(spam_nnet_table))/sum(spam_nnet_table)
spam_accuracy

"0.9356522"

#Normalization of Data

set.seed(123)
spam_norm <- spam_data[,-58]
min <- apply(spam_norm, 2, min)
max <- apply(spam_norm, 2, max)
spam_norm <- as.data.frame(scale(spam_norm, center = min, scale = max-min))
spam_norm$target <- spam_data$target

#Splitting the normalized data into Train and Test
indexed <- sample(1:nrow(spam_norm), size = 0.25*nrow(spam_norm))
spam_norm_train <- spam_norm[-indexed,]
spam_norm_test <- spam_norm[indexed,]
spam_norm_test_without_target<-spam_norm_test[,-58]

#Building the model for Normalized data
spam_norm_nnet <- nnet(target ~., data = spam_norm_train, size=10, maxit=5000)
spam_norm_nnet_pred <- predict(spam_norm_nnet, spam_norm_test_without_target)
spam_norm_nnet_pred <- round(spam_norm_nnet_pred)
spam_nnet_table <- table(Actual=spam_norm_test$target, Predicted=spam_norm_nnet_pred)
spam_nnet_table

"
      Predicted
Actual   0   1
      0 666  48
      1  29 407
"

spam_nnet_norm_accuracy <- sum(diag(spam_nnet_table))/sum(spam_nnet_table)
spam_nnet_norm_accuracy
"0.9330435"


### Splitting data fro RBF
spam_training<-spam_train
inputs<-spam_training[,-58]
outputs<-spam_training[,58]

### Building RBF Model
model <- rbf(inputs, outputs, size=50, maxit=500)
model_predict<-predict(model, spam_test_without_target)
model_predict<-round(model_predict)

"        Predicted
Actual   0   1
0        454 249
1        88 359

"

### Checking Accuracy of Model
spam_norm_table <- table(Actual=spam_test$target, Predicted=model_predict)
spam_norm_table
spam_norm_accuracy <- sum(diag(spam_norm_table))/sum(spam_norm_table)
spam_norm_accuracy

"70.69"

###Multinomial Logistic Regression

model <- multinom(target ~ ., data=spam_train, family="binomial")
model_predict<- predict(model, spam_test_without_target)
spam_logit_table <- table(Actual=spam_test$target, Predicted=model_predict)
spam_logit_table

"
      Predicted
Actual   0   1
      0 669  30
      1  54 397
"
multinom_accuracy <- sum(diag(spam_logit_table))/sum(spam_logit_table)
multinom_accuracy 
"0.9269565"

### Creating Ensemble Model

write.csv(model_predict,"C:\\Users\\user\\Desktop\\Neural Networks\\rbf_predict.csv")
write.csv(spam_nnet_pred,"C:\\Users\\user\\Desktop\\Neural Networks\\output_nnet_predict.csv")
write.csv(spam_norm_nnet_pred,"C:\\Users\\user\\Desktop\\Neural Networks\\output_nnet_normal_predict.csv")
write.csv(spam_pred,"C:\\Users\\user\\Desktop\\Neural Networks\\mlm_predict.csv")



### Ensemble using of the Three Models Multi layer Feed Forward, Multi layer Feed Forward with Normalized data,
### RBF Network
combine2<-read.csv("C:\\Users\\user\\Desktop\\Neural Networks\\combine2.csv")
table2 <- table(observed=spam_test$target, predicted=combine2$Final)
table2 
spam_norm_accuracy2 <- sum(diag(table2))/sum(table2)
spam_norm_accuracy2

"
        predicted
observed   0   1
        0 443 259
        1 256 192
"

"0.5521739"

### The accuracy is poor in the above ensemble because the Models are similar

### Ensemble of Four Models- Multi layer Feed Forward, Multi layer Feed Forward with Normalized data,
### RBF Network and Multinomial Logistic Regression
combine<-read.csv("C:\\Users\\user\\Desktop\\Neural Networks\\combine.csv")
table <- table(observed=spam_test$target, predicted=combine$Final)
table 

"        predicted
observed   0   1
       0 434 268
       1 149 299 "

spam_norm_accuracy <- sum(diag(table))/sum(table)
spam_norm_accuracy
" 0.6373913"

### The Ensemle Accuracy got improved because we have included a logit model whaich caused a
### variation in Ensemble