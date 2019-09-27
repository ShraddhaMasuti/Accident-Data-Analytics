#lets install all the packages we will sue up front to avoid conflicts

install.packages(c("ipred","parallel","iterators","lattice","ggplot2","doParallel", "caret", "mice", "pROC", "VIM", "ipred", "ada", "randomForest"))
install.packages("caret")
library("lattice")
library("ggplot2")
library("iterators")
library("parallel")
library("doParallel")
library("ipred")
library("caret")
library("pROC")

library(mice) #imputation package will discuss later
library(VIM)

# Read the data
#set working folder to location of data files
setwd("~/Accidents files")

accidents <- read.csv("Accident_CSV_only_categorical_data.csv")
ncol(accidents)
nrow(accidents)
str(accidents)
inTrain<-createDataPartition(y=accidents$accident_severity, p=.80, list=FALSE)
nrow(inTrain)
accidents$accident_severity
imbal_accident_train <- accidents[(inTrain),]
str(imbal_accident_train)
nrow(imbal_accident_train)
imbal_accident_test <- accidents[(-inTrain),]
str(imbal_accident_test)
nrow(imbal_accident_test)
imbal_accident_train
table(imbal_accident_train$accident_severity)
install.packages("DMwR")
install.packages("grid")
library(grid)
library(DMwR)
#hybrid both up and down
set.seed(192)
str(imbal_accident_train)
summary(imbal_accident_train)
 smote_train <- SMOTE(accident_severity~ ., imbal_accident_train, perc.over = 270, perc.under=200)                         
table(smote_train$accident_severity)
## write the downsampled data in a csv file
write.table(smote_train, "~/Accidents files/smote_train.csv", sep=",")

write.table(imbal_accident_test, "~/Accidents files/test_data.csv", sep=",")
## ROSE Part
# Read the data
#set working folder to location of data files
setwd("~/Accidents files")
accidents <- read.csv("Accident_CSV_only_categorical_data.csv")
ncol(accidents)
nrow(accidents)
str(accidents)
inTrain<-createDataPartition(y=accidents$accident_severity, p=.70, list=FALSE)
nrow(inTrain)
accidents$accident_severity
imbal_accident_train <- accidents[(inTrain),]
str(imbal_accident_train)
nrow(imbal_accident_train)
imbal_accident_test <- accidents[(-inTrain),]
str(imbal_accident_test)
nrow(imbal_accident_test)
imbal_accident_train
table(imbal_accident_train$accident_severity)
#install.packages("DMwR")
#install.packages("grid")
#library(grid)
#library(DMwR)

#hybrid both up and down
set.seed(192)
str(imbal_accident_train)
summary(imbal_accident_train)
#smote_train <- SMOTE(accident_severity~ ., imbal_accident_train, perc.over = 270, perc.under=200)                         
install.packages("ROSE")
library(ROSE)
set.seed(192)
rose_train <-ROSE(accident_severity ~ ., data  = imbal_accident_train)$data                         
table(rose_train$accident_severity)

#table(smote_train$accident_severity)
## write the downsampled data in a csv file
write.table(rose_train, "~/Accidents files/rose_train.csv", sep=",")

write.table(imbal_accident_test, "~/Accidents files/test_data.csv", sep=",")
##Main Code for Models and Performance
#lets install all the packages we will sue up front to avoid conflicts
install.packages(c("ipred","parallel","iterators","lattice","ggplot2","doParallel", "caret", "mice", "pROC", "VIM", "ipred", "ada", "randomForest"))
library(lattice)
library(ggplot2)
library(caret)
library(pROC)
library(mice)
library(Rcpp)
library(doParallel)
library(MASS)
library(kernlab)
library(e1071)
library(ISLR)
library(rpart)
library(tree)
library(randomForest)
library(klaR)
library(survival)
library(dplyr)
library(plyr)
library(gbm)
library(mgcv)
library(nlme)
library(rpart.plot)

library(mice) #imputation package will discuss later
library(VIM)

# Read the data
#set working folder to location of data files
setwd("~/Accidents files")

accident_train<- read.csv("rose_train.csv")
head(accident_train)
summary(accident_train)
ncol(accident_train)
nrow(accident_train)
str(accident_train)

accident_test <- read.csv("test_data.csv")
head(accident_test)
summary(accident_test)
ncol(accident_test)
nrow(accident_test)
str(accident_test)

#with 15 variables

drops<-c("road_surface_conditions", "pedestrian_crossing_human_control", "carriageway_hazards",
         "vehicle_location_restricted_lane", "day_of_week", "was_vehicle_left_hand_drive", "special_conditions_at_site")

accident_train <- accident_train [,!(names(accident_train) %in% drops)]
colnames(accident_train)

## Create train data split for DV and Predictors
y.train <- accident_train$accident_severity
x.train <- accident_train [,-ncol(accident_train)]
head(y.train)
colnames(x.train)
## Create test data split for DV and Predictors
y.test <- accident_test$accident_severity
x.test <- accident_test [,-ncol(accident_test)]
head(y.test)
colnames(x.test)

setdiff(levels(x.train$lsoa_of_accident_location),
        levels(x.test$lsoa_of_accident_location))



library("doParallel")
## to run parallel with 8 cores
cl <- makeCluster(8)
registerDoParallel(cl)

##lets start modeling

#some parameters to control the sampling during parameter tuning and testing
#10 fold crossvalidation, using 10-folds instead of 10 to reduce computation time in class demo, use 10 and with more computation to spare use
#repeated cv
ctrl <- trainControl(method="cv", number=10,
                     classProbs=TRUE,
                     #function used to measure performance
                     summaryFunction = twoClassSummary, #multiClassSummary for non binary
                     allowParallel =  TRUE) 


set.seed(192)

library(rpart)
set.seed(192)

m.rpart <- train(y=y.train, x=x.train, 
                 method = "rpart",tuneLength=7,
                 metric = "ROC",
                 trControl = ctrl)

getTrainPerf(m.rpart)
varImp(m.rpart)

##ROC of Train data
#dt_train.roc<-roc(response=y.train,predictor=x.train$)

#confusionMatrix(m.rpart,y.train) #calc accuracies with confuction matrix on downsampled training data set


#the best performing model trained on the full training set is saved 
##preprocessing using predict function with caret train object will be applied to new data
p.rpart <- predict(m.rpart,x.test)
p.rpart.prob <- predict(m.rpart, x.test, type="prob")



confusionMatrix(p.rpart,y.test) #calc accuracies with confuction matrix on test set

p.rpart.newthresh <- factor(ifelse(p.rpart.prob[[1]]>0.37, "Serious", "Slight"))
p.rpart.newthresh 
confusionMatrix(p.rpart.newthresh, y.test)

test.rpart.roc<- roc(response= y.test, predictor= p.rpart.prob[[1]])
plot(test.rpart.roc)

plot(test.rpart.roc)
newthresh<- coords(test.rpart.roc, x="best", best.method="closest.topleft")

p.rpart.prob[[1]]
levels(p.rpart.newthresh)
plot(p.rpart)


fit <- rpart(y.train~. ,data=x.train,  
             method="class",
             control=rpart.control(minsplit=1),
             parms=list(split='information'))
plot(fit)
library(rpart.plot)
rpart.plot(fit, type=4, extra=2,clip.right.labs=FALSE, varlen=0, faclen=3)
rpart.plot(fit)


printcp(fit) #display crossvalidated error for each tree size
plotcp(fit) #plot cv error


auc(test.rpart.roc)

#we can grab this from the plotcp table automatically with 
opt.cp <- fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]

#lets prune the tree
fit.pruned <- prune(fit,cp=0.016911)

#lets review the final tree
rpart.plot(fit.pruned)


##Naive Bayes

m.nb <- train(y=y.train, x=x.train,
              trControl = ctrl,
              metric = "ROC", #using AUC to find best performing parameters
              method = "nb")
m.nb

getTrainPerf(m.nb)
varImp(m.nb)
plot(m.nb)
#confusionMatrix(m.nb,y.train)


p.nb<- predict(m.nb,x.test)
p.nb.prob<- predict(m.nb,x.test,type="prob")
plot(p.nb)
confusionMatrix(p.nb,y.test) #calc accuracies with confuction matrix on test set

p.nb.newthresh <- factor(ifelse(p.nb.prob[[1]]>0.35, "Serious", "Slight"))
p.nb.newthresh 
confusionMatrix(p.nb.newthresh, y.test)


test.nb.roc<- roc(response= y.test, predictor= p.nb.prob[[1]])
plot(test.nb.roc)
newthresh<- coords(test.rpart.roc, x="best", best.method="closest.topleft")

p.rpart.prob[[1]]
levels(p.rpart.newthresh)
plot(p.rpart)

auc(test.nb.roc)

## Boosting

install.packages("rpart")
library(ada)
set.seed(192)
#boosted decision trees
#using dummy codeds because this function internally does it and its better to handle it yourself (i.e., less error prone)

m.ada <- train(y=y.train, x=x.train,
               trControl = ctrl,
               metric = "ROC", #using AUC to find best performing parameters
               method = "ada")


m.ada
getTrainPerf(m.ada)
varImp(m.ada)
plot(m.ada)
p.ada<- predict(m.ada,x.test)
confusionMatrix(p.ada,y.test)

p.ada.prob <- predict(m.ada, x.test, type="prob")
p.ada.newthresh <- factor(ifelse(p.ada.prob[[1]]>0.40, "Serious", "Slight"))
p.ada.newthresh 
confusionMatrix(p.ada.newthresh, y.test)

test.ada.roc<- roc(response= y.test, predictor= p.ada.prob[[1]])
plot(test.ada.roc)

p.ada.prob[[1]]
levels(p.ada.newthresh)
plot(p.ada)

auc(test.ada.roc)

##Random Forest
#random forest approach to many classification models created and voted on
#less prone to ovrefitting and used on large datasets
library(randomForest)

m.rf <- train(y=y.train, x=x.train,
              trControl = ctrl, probmodel=TRUE,forClass=TRUE,
              metric = "ROC", #using AUC to find best performing parameters
              method = c("rf") )
m.rf
getTrainPerf(m.rf)
varImp(m.rf)
plot(m.rf)

p.rf<- predict(m.rf,x.test)
plot(p.rf)
confusionMatrix(p.rf,y.test)

p.rf.prob<- predict(m.rf,x.test,type="prob")
p.rf.newthresh <- factor(ifelse(p.rf.prob[[1]]>0.31, "Serious", "Slight"))
p.rf.newthresh 
confusionMatrix(p.rf.newthresh, y.test)

test.rf.roc<- roc(response= y.test, predictor= p.rf.prob[[1]])
plot(test.rf.roc)

p.rf.prob[[1]]
levels(p.rf.newthresh)

auc(test.rf.roc)


## Bagging

library(ipred)
set.seed(192)

m.bag <- train(y=y.train, x=x.train,
               trControl = ctrl,
               metric = "ROC", #using AUC to find best performing parameters
               method = "treebag")
m.bag

getTrainPerf(m.bag)
varImp(m.bag)
plot(m.bag)

p.bag<- predict(m.bag,x.test)
plot(p.bag)
confusionMatrix(p.bag,y.test)

p.bag.prob<- predict(m.bag,x.test,type="prob")
p.bag.newthresh <- factor(ifelse(p.bag.prob[[1]]>0.30, "Serious", "Slight"))
p.bag.newthresh 
confusionMatrix(p.bag.newthresh, y.test)

test.bag.roc<- roc(response= y.test, predictor= p.bag.prob[[1]])
plot(test.bag.roc)

p.bag.prob[[1]]
levels(p.bag.newthresh)

auc(test.bag.roc)


##Neural Netwrok

# Read the data
#set working folder to location of data files
setwd("~/Accidents files")

acc_num <- read.csv("Accident_CSV_only_numeric_data_Neural_network.csv")
ncol(acc_num)
nrow(acc_num)
str(acc_num)


inTrain_num<-createDataPartition(y=acc_num$accident_severity, p=.70, list=FALSE)
nrow(inTrain_num)
acc_num$accident_severity

imbal_accident_train_num <- acc_num[(inTrain_num),]
str(imbal_accident_train_num)
nrow(imbal_accident_train_num)
imbal_accident_test_num <- acc_num[(-inTrain_num),]
str(imbal_accident_test_num)
nrow(imbal_accident_test_num)
nrow(imbal_accident_test_num)
imbal_accident_train_num

table(imbal_accident_train_num$accident_severity)

#hybrid both up and down
set.seed(192)
str(imbal_accident_train_num)
summary(imbal_accident_train_num)

install.packages("ROSE")

library(ROSE)

set.seed(192)
rose_train_num <-ROSE(accident_severity ~ ., data  = imbal_accident_train_num)$data                         
table(rose_train_num$accident_severity)

#table(smote_train$accident_severity)

## write the downsampled data in a csv file
write.table(rose_train_num, "~/Accidents files/rose_train_num.csv", sep=",")

write.table(imbal_accident_test_num, "~/Accidents files/test_data_num.csv", sep=",")


install.packages("neuralnet")
install.packages("devtools")
install.packages("NeuralNetTools")
install.packages("ggplot2")

library("NeuralNetTools")
library("devtools")
library("neuralnet")

accident_train_num<- read.csv("rose_train_num.csv")
head(accident_train_num)
summary(accident_train_num)
ncol(accident_train_num)
nrow(accident_train_num)
str(accident_train_num)
table(accident_train_num$accident_severity)

accident_test_num <- read.csv("test_data_num.csv")
head(accident_test_num)
summary(accident_test_num)
ncol(accident_test_num)
nrow(accident_test_num)
str(accident_test_num)
table(accident_test_num$accident_severity)


#with 15 variables

drops<-c("road_surface_conditions", "pedestrian_crossing_human_control", "carriageway_hazards",
         "vehicle_location_restricted_lane", "day_of_week", "was_vehicle_left_hand_drive", "special_conditions_at_site")

accident_train_num <- accident_train_num [,!(names(accident_train_num) %in% drops)]
colnames(accident_train_num)


## Create train data split for DV and Predictors
y.train_num <- accident_train_num$accident_severity
x.train_num <- accident_train_num [,-ncol(accident_train_num)]

dummy_accident_xtrain <- dummyVars(" ~ .", data = x.train_num)
x.train_num1 <- data.frame(predict(dummy_accident_xtrain, newdata = x.train_num))
#print(trsf)
head(x.train_num1)
names(x.train_num1)


head(y.train_num)
colnames(x.train_num1)

accident_test_num <- accident_test_num [,!(names(accident_test_num) %in% drops)]

#dummy_accident_test_num <- dummyVars(" ~ .", data = accident_test_num)
#accident_test_num1 <- data.frame(predict(dummy_accident_test_num, newdata = accident_test_num))
## Create test data split for DV and Predictors
y.test_num <- accident_test_num$accident_severity
x.test_num <- accident_test_num [,-ncol(accident_test_num)]
names(x.test_num)
dummy_accident_xtest1 <- dummyVars(" ~ .", data = x.test_num)
x.test_num1 <- data.frame(predict(dummy_accident_xtest1, newdata = x.test_num))
#print(trsf)
#x.test_num1
head(y.test_num)
colnames(x.test_num1)
names(x.test_num1)
#setdiff(levels(x.train_num$lsoa_of_accident_location),
#      levels(x.test_num$lsoa_of_accident_location))
#library("doParallel")
## to run parallel with 8 cores
#cl <- makeCluster(8)
#registerDoParallel(cl)

##lets start modeling

#some parameters to control the sampling during parameter tuning and testing
#10 fold crossvalidation, using 10-folds instead of 10 to reduce computation time in class demo, use 10 and with more computation to spare use
#repeated cv
ctrl <- trainControl(method="cv", number=10,
                     classProbs=TRUE,
                     #function used to measure performance
                     summaryFunction = twoClassSummary, #multiClassSummary for non binary
                     allowParallel =  TRUE) 
set.seed(192)
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}
accident_norm_train <- as.data.frame(lapply(x.train_num1, normalize))
accident_norm_test <- as.data.frame(lapply(x.test_num1, normalize))
accident_norm_train_total<-cbind(x.train_num1,y.train_num)
head(accident_norm_train_total)
set.seed(192)

nn <- train(y=y.train_num, x=x.train_num1, 
            method = "nnet",tuneLength=2,
            metric = "ROC",
            trControl = ctrl)

getTrainPerf(nn)
varImp(nn)
plot(nn)
plotnet(nn)
p.nn<- predict(nn,x.test_num1)
plot(p.nn)
confusionMatrix(p.nn,y.test_num)
p.nn.prob <- predict(nn, x.test_num1, type="prob")
test.nn.roc<- roc(response= y.test_num, predictor= p.nn.prob[[1]])
plot(test.nn.roc)
p.nn.prob[[1]]
auc(test.nn.roc)
#compare training performance
#create list of cross validation runs (resamples)
rValues <- resamples(list(rpart=m.rpart, naivebayes=m.nb, randomForest=m.rf, bagging=m.bag, boosting=m.ada, nn=nn))
#create plot comparing them
bwplot(rValues, metric="ROC")
bwplot(rValues, metric="Sens") #Sensitvity
bwplot(rValues, metric="Spec")

#create dot plot comparing them
dotplot(rValues, metric="ROC")
dotplot(rValues, metric="Sens") #Sensitvity
dotplot(rValues, metric="Spec")
xyplot(rValues, metric="ROC")
xyplot(rValues, metric="Sens") #Sensitvity
xyplot(rValues, metric="Spec")
summary(rValues)
#using no probability as positive class
rpart.roc<- roc(y.test, rpart.prob$Serious)
nb.roc<- roc(y.test, nb.prob$no)
#lets see auc
auc(rpart.roc)
auc(nb.roc)

