
rm(list=ls())
# set path 
setwd('D:/t-SNE/R code/')

# install.packages('PRROC')
library(PRROC) # AUC, AUPR package
library(mltools) # MCC package 
library(glmnet) # LR package
library(e1071) # svm package
library(rpart) # DT package
library(tsne) # tsne package

# load data
dat <- read.csv('dataset-2.csv', header = T)

# variables and group
vars <- dat[,-c(1, 2)]
y <- ifelse(dat$group=='Control', 0, 1)

# remove the columns with 90% zeros
s <- apply(vars, 2, function(x)sum(x!=0)/length(x))
vars <- vars[, which(s>0.1)]

# repalace zero by 1e-12
vars[vars==0] <- 1e-12

# Aitchison distance
vars0 <- vars
A.vars <- t(apply(vars0, 1, function(x)log(x/exp(mean(log(x))))))
A.dist <- as.matrix(dist(A.vars))

# train and test 
p = 0.8
sub <- sample(1:nrow(dat), round(nrow(dat)*p))

xtrain <- A.vars[sub, ]
xtest <- A.vars[-sub, ]

ytrain <- y[sub]
ytest <- y[-sub]

# set parameters
k <- 7; per <- 30; dim <- 3; iter <- 2000 # per = 15 for dataset-4 

# find k nearest neighbors based on Aitchison distance
test.dist <- A.dist[-sub, sub]
test.neighbors.index <- t(apply(test.dist, 1, function(x)order(x)[1:k]))
test.neighbors.dist <- t(apply(test.dist, 1, function(x)sort(x)[1:k]))

# t-sne dimensionality reduction and visualization
colors = rainbow(length(unique(ytrain)))
names(colors) = unique(ytrain)
ecb = function(x,y){ plot(x, xlab='x', ylab='y', t='n', main = 't-SNE'); 
  text(x, cex=0.7, labels=ytrain, col=colors[ytrain]) }

# model fit
tsne.fit = tsne::tsne(dist(xtrain), k = dim, perplexity = per, max_iter = iter, epoch_callback = ecb, whiten = F) 

lowdata = tsne.fit

# train lr model
obj1 = cv.glmnet(as.matrix(lowdata), ytrain, standardize=F, family="binomial")
lr = glmnet(as.matrix(lowdata), ytrain, standardize=F, family="binomial", lambda = obj1$lambda.min)

# train svm model
obj2 = tune.svm(lowdata, as.factor(ytrain), scale=F, gamma = 10^(-6:1), cost = c(1:5))
svm = svm(lowdata, as.factor(ytrain), probability=T, scale=F,
          gamma = obj2$best.parameters[1], cost = obj2$best.parameters[2])

# train DT model
dt.dat <- data.frame(y = ytrain, x = lowdata)
obj3 = tune.rpart(y~., data = dt.dat, 
                  minsplit = c(5, 10, 15, 20),
                  minbucket = c(5, 10, 15, 20), 
                  cp = c(0.01, 0.05, 0.1), 
                  maxdepth = c(5, 10, 15)) # e1071 package

dt = rpart(y~., data = dt.dat, method = 'class', 
           minsplit = obj3$best.parameters[1],
           minbucket = obj3$best.parameters[2],
           control = rpart.control(cp = obj3$best.parameters[3]), 
           maxdepth = obj3$best.parameters[4])

# make predictions on the test data
lbs1 <- c(); p1 <- c()
lbs2 <- c(); p2 <- c() 
lbs3 <- c(); p3 <- c()

for(i in 1:length(ytest)){
  
  # compute weights
  w0 = exp(-test.neighbors.dist[i,])
  w = w0/sum(w0)
  
  # test lowdata
  test <- w%*%as.matrix(lowdata)[test.neighbors.index[i,],]
  
  # predictive labels and probs
  lbs1[i] <- predict(lr, test, type = "class")
  p1[i] <- 1 - predict(lr, test, type = "response")
  
  lbs2[i] <- predict(svm, test)
  p2[i] <- attr(predict(svm, test, probability = TRUE), "probabilities")[1]
  
  lbs3[i] <- predict(dt, data.frame(x = test), type = "class")
  p3[i] <- predict(dt, data.frame(x = test), type = "prob")[1]   

}

# confusion matrix
table(ytest, lbs1)
table(ytest, lbs2)
table(ytest, lbs3) 

# acc, nmcc, auc, aupr
acc <- function(x, y){ sum(diag(table(x, y)))/sum(table(x, y)) }
nmcc <- function(x, y){ (mltools::mcc(confusionM = matrix(table(x, y), 2, 2)) + 1)/2 }
auc <- function(p0, p1){ PRROC::roc.curve(p0, p1)$auc }
aupr <- function(p0, p1){ PRROC::pr.curve(p0, p1)$auc.integral }

# lr_res
acc(ytest, lbs1)
nmcc(ytest, lbs1)
auc(p0 = p1[ytest==0], p1 = p1[ytest==1])
aupr(p0 = p1[ytest==0], p1 = p1[ytest==1])


