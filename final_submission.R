library(xgboost)
library(Matrix)
library(caret)
library(ROCR)

set.seed(1234)

train <- read.csv("train.csv")
test  <- read.csv("test.csv")

##### Removing IDs
train$ID <- NULL
test.id <- test$ID
test$ID <- NULL

##### Extracting TARGET
train.y <- train$TARGET
train$TARGET <- NULL

##### 0 count per line
count0 <- function(x) {
    return( sum(x == 0) )
}
train$n0 <- apply(train, 1, FUN=count0)
test$n0 <- apply(test, 1, FUN=count0)

##### Removing constant features
cat("\n## Removing the constants features.\n")
for (f in names(train)) {
    if (length(unique(train[[f]])) == 1) {
        cat(f, "is constant in train. We delete it.\n")
        train[[f]] <- NULL
        test[[f]] <- NULL
    }
}

##### Removing identical features
features_pair <- combn(names(train), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
    f1 <- pair[1]
    f2 <- pair[2]
    
    if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
        if (all(train[[f1]] == train[[f2]])) {
            cat(f1, "and", f2, "are equals.\n")
            toRemove <- c(toRemove, f2)
        }
    }
}

#### Setting -999999 to 2 in var3 (possibly country)
train$var3[train$var3 == -999999] <- 2
test$var3[test$var3 == -999999] <- 2

#### Dealing with var38 (possibly mortgage)
train$var38mc <- as.numeric(train$var38 == 117310.979016494)
test$var38mc <- as.numeric(test$var38 == 117310.979016494)
train$logvar38 <- 0
test$logvar38 <- 0
train$logvar38[train$var38mc == 0] <- log(train$var38[train$var38mc == 0])
test$logvar38[test$var38mc == 0] <- log(test$var38[test$var38mc == 0])

#### Playing with var36
train$var36_is99 <- as.numeric(train$var36 == 99)
test$var36_is99 <- as.numeric(test$var36 == 99)

##### Removing fully correlated features
# for (pair in features_pair) {
#     f1 <- pair[1]
#     f2 <- pair[2]
#     
#     if(!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
#         if (cor(train[[f1]], train[[f2]]) == 1) {
#             cat (f1, "and", f2, "are fully correlated.\n")
#             toRemove <- c(toRemove, f2)
#         }
#     }
# }

##### Removing very highly correlated features
# for (pair in features_pair) {
#     f1 <- pair[1]
#     f2 <- pair[2]
#     
#     if(!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
#         if (cor(train[[f1]], train[[f2]]) > 0.99) {
#             cat (f1, "and", f2, "are very highly correlated.\n")
#             # toRemove <- c(toRemove, f2)
#         }
#     }
# }
# 
# ##### Removing highly correlated features
# for (pair in features_pair) {
#     f1 <- pair[1]
#     f2 <- pair[2]
#     
#     if(!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
#         if (cor(train[[f1]], train[[f2]]) > 0.9) {
#             cat (f1, "and", f2, "are highly correlated.\n")
#             toRemove <- c(toRemove, f2)
#         }
#     }
# }


feature.names <- setdiff(names(train), toRemove)

train <- train[, feature.names]
test <- test[, feature.names]

train$TARGET <- train.y

folds <- createFolds(as.factor(train$TARGET), 5)
fold_auc <- c()
# 
# ##########################################################
# ## Model building with stratified 5-fold CV
for (fold in folds) {
    x_train <- train[-fold,]
    x_train.y <- train$TARGET[-fold]
    x_test <- train[fold,]
    x_test.y <- train$TARGET[fold]
    
    x_train <- sparse.model.matrix(TARGET ~ ., data= x_train)
    x_test <- sparse.model.matrix(TARGET ~ ., data = x_test)
    
    d_train <- xgb.DMatrix(data = x_train, label = x_train.y)
    d_test <- xgb.DMatrix(data = x_test, label = x_test.y)
    watchlist <- list(train=d_train, test=d_test)
    
    param <- list(  objective           = "binary:logistic", 
                    booster             = "gbtree",
                    eval_metric         = "auc",
                    eta                 = 0.0203,
                    max_depth           = 5,
                    subsample           = 0.683,
                    colsample_bytree    = 0.7
    )
    
    clf <- xgb.train(   params              = param, 
                        data                = d_train, 
                        nrounds             = 574, 
                        verbose             = 2,
                        watchlist           = watchlist,
                        maximize            = FALSE
    )
    
    fold_pred <- predict(clf, x_test)
    pred <- prediction(fold_pred, x_test.y)
    perf <- performance(pred, measure = "tpr", x.measure = "fpr")
    auc <- performance(pred, measure = "auc")
    fold_auc <- c(fold_auc, auc@y.values[[1]])
}

##########################################################

# split <- createDataPartition(y = train.y, p = 0.7, list = F)
# 
# xtrain <- train[split,]
# xtrain.y <- train.y[split]
# xtest <- train[-split,]
# xtest.y <- train.y[-split]
# 
# xtrain <- sparse.model.matrix(TARGET ~ ., data = xtrain)
# xtest <- sparse.model.matrix(TARGET ~ ., data = xtest)
# 
# dtrain <- xgb.DMatrix(data=xtrain, label=xtrain.y)
# dtest <- xgb.DMatrix(data = xtest, label=xtest.y)
# watchlist <- list(train=dtrain, test = dtest)

train <- sparse.model.matrix(TARGET ~ ., data = train)
dtrain <- xgb.DMatrix(data=train, label=train.y)
watchlist <- list(train=dtrain)
# 
param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.0203,
                max_depth           = 5,
                subsample           = 0.683,
                colsample_bytree    = 0.7
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 574, 
                    verbose             = 2,
                    watchlist           = watchlist,
                    maximize            = FALSE
)
# test_pred <- predict(clf, x_test)
# 
# 
test$TARGET <- -1
test <- sparse.model.matrix(TARGET ~ ., data = test)

preds <- predict(clf, test)
submission <- data.frame(ID=test.id, TARGET=preds)
cat("saving the submission file\n")
write.csv(submission, "submission_kscript_xgb2.csv", row.names = F)
    