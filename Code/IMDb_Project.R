library(tm)
library(SnowballC)
library(ggplot2)
library(dplyr)
library(ranger)
library(caret)
library(kernlab)
library(doParallel) 
library(e1071)
library(keras)
library(tensorflow)
use_condaenv("keras-tf", required = TRUE)
use_condaenv("r-reticulate")

train_file <- read.csv("Desktop/w22projii/W22_P2_train.csv", 
                       header = TRUE, colClasses = c("description" = "character"))
test_file <- read.csv("Desktop/w22projii/W22_P2_test.csv", 
                      header = TRUE, colClasses = c("description" = "character"))
glimpse(train_file)
summary(train_file)

#data visualization
ggplot(data=train_file, aes(x=genre, y=nchar(genre))) + geom_col() + ylab("count")

#First, do some preprocessing
comb_dat <- Corpus(VectorSource(c(train_file$description,
                                  test_file$description)))
comb_dat <- tm_map(comb_dat, stripWhitespace)
comb_dat <- tm_map(comb_dat, removePunctuation)
comb_dat <- tm_map(comb_dat, content_transformer(tolower))
comb_dat <- tm_map(comb_dat, removeWords, stopwords("english"))
comb_dat <- tm_map(comb_dat, stemDocument)

inspect(comb_dat[1:3])

#Second, create word occurrence matrices: 
#each column corresponds to a word, and each row is a description of a movie
#(i,j)-th entry is the number of occurrence for word j in i-th movie

minDocFreq= 100
maxDocFreq = 6400

comb_dtm <- DocumentTermMatrix(comb_dat, control = list(bounds = list(global = c(minDocFreq, maxDocFreq))))

inspect(comb_dtm)

#third, remove sparse terms
comb_mat = removeSparseTerms(comb_dtm, .99)
inspect(comb_mat)
findFreqTerms(comb_mat)

#create training and test data in the form of matrices and vectors
Xtrain = as.matrix(comb_mat[1:nrow(train_file),])
Ytrain = train_file$genre

Xtest = as.matrix(comb_mat[(nrow(train_file)+1):(nrow(train_file)+nrow(test_file)), ])
testID = test_file$id #use for output

trainID = sample(1:nrow(Xtrain), floor(0.8*nrow(Xtrain)))
Xtrain_dat = Xtrain[trainID, ]
Xtest_dat = Xtrain[-trainID, ]

Ytrain_dat = Ytrain[trainID]
Ytest_dat = Ytrain[-trainID]

#model 1: random Forest
ctrl <- trainControl(method = "cv",
                     number = 5)

rf.Grid = expand.grid(mtry = 2*(1:10),
                      splitrule ='gini', 
                      min.node.size = 1)

rf.cv.model <- train(Xtrain_dat, Ytrain_dat,
                     method = "ranger",
                     trControl = ctrl,
                     tuneGrid = rf.Grid)
rf.cv.model

yhat.cv.test = predict(rf.cv.model, Xtest_dat)
table(yhat.cv.test, Ytest_dat)
mean(yhat.cv.test == Ytest_dat)
#0.553221

# write the file for kaggle submission
rf_pred = predict(rf.cv.model, Xtest)
out.df <- data.frame(id = testID, genre = rf_pred)
colnames(out.df) <- c('id', 'genre')
write.csv(out.df, file = "Desktop/w22projii/W22_P2_submission_rf.csv", row.names = FALSE)

#model #2: SVM
ctrl <- trainControl(method = "cv",
                     number = 5,
                     allowParallel = TRUE)

lmSVM.Grid = expand.grid(C = (1:10)*10)

registerDoParallel(cores=6) 

lmSVM.cv.model <- train(Xtrain_dat, Ytrain_dat,
                        method = "svmLinear",
                        trControl = ctrl,
                        tuneGrid = lmSVM.Grid)
lmSVM.cv.model
lmSVM.cv.pred = predict(lmSVM.cv.model, Xtest_dat)
table(lmSVM.cv.pred, Ytest_dat)
mean(lmSVM.cv.pred == Ytest_dat)

# write the file for kaggle submission
lmSVM_pred = predict(lmSVM.cv.model, Xtest)
out.df_3 <- data.frame(id = testID, genre = lmSVM_pred)
colnames(out.df_3) <- c('id', 'genre')
write.csv(out.df_3, file = "Desktop/w22projii/W22_P2_submission_SVM.csv", row.names = FALSE)

#Model #3: Naive Bayes
nb.mod=naiveBayes(Xtrain_dat, Ytrain_dat)
nb_pred = predict(nb.mod, Xtest_dat)
table(nb_pred, Ytest_dat)
mean(nb_pred == Ytest_dat)
#0.5375

# write the file for kaggle submission
nb.pred = predict(nb.mod, Xtest)
out.df_3 <- data.frame(id = testID, genre = nb.pred)
colnames(out.df_3) <- c('id', 'genre')
write.csv(out.df_3, file = "Desktop/w22projii/W22_P2_submission_NaiveBayes.csv", row.names = FALSE)


#model #4: Neural Network:
#transform labels from characters to integers
Ytrain_lstm = Ytrain
for (i in 1:length(Ytrain_lstm)){
  if (Ytrain_lstm[i] == " comedy "){
    Ytrain_lstm[i] = 0
  }
  if (Ytrain_lstm[i] == " documentary "){
    Ytrain_lstm[i] = 1
  }
  if (Ytrain_lstm[i] == " drama "){
    Ytrain_lstm[i] = 2
  }
  if (Ytrain_lstm[i] == " short "){
    Ytrain_lstm[i] = 3
  }
}
Ytrain_lstm = as.numeric(Ytrain_lstm)

Ytrain_data = Ytrain_lstm[trainID]
Ytest_data = Ytrain_lstm[-trainID]

#padding sequences to same length
train_x <- pad_sequences(Xtrain_dat, maxlen = 6400)
test_x <- pad_sequences(Xtest_dat, maxlen = 6400)
test_X <- pad_sequences(Xtest, maxlen = 6400)

#build the model
model <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu",) %>%
  layer_dropout(0.8) %>%
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(0.8) %>%
  layer_dense(units = 4, activation = "softmax")

model %>% compile(loss = "sparse_categorical_crossentropy",
                  optimizer = "adam",
                  metrics = c("accuracy"))

lstm_model <- model %>% fit(train_x, Ytrain_data,
                         epochs = 10,
                         batch_size = 64,
                         validation_data = list(test_x, Ytest_data))
plot(lstm_model)
model %>% evaluate(test_x, Ytest_data)
#0.633000

probs <- predict(model, test_x)
#choose the genre with largest probability
pred = apply(probs, 1, which.max)
pred = pred-1

table(Predicted=pred, Actual=Ytest_data) 
mean(pred == Ytest_data)
#0.633

# write the file for kaggle submission
prob_lstm <- predict(model, test_X)
pred = apply(prob_lstm, 1, which.max)
pred = pred-1

for (i in 1:length(pred)){
  if (pred[i] == 0){
    pred[i] = " comedy "
  }
  if (pred[i] == 1){
    pred[i] = " documentary "
  }
  if (pred[i] == 2){
    pred[i] = " drama "
  }
  if (pred[i] == 3){
    pred[i] = " short "
  }
}

# write the file for kaggle submission
out.df_4 <- data.frame(id = testID, genre = pred)
colnames(out.df_4) <- c('id', 'genre')
write.csv(out.df_4, file = "Desktop/w22projii/W22_P2_submission_lstm.csv", row.names = FALSE)
