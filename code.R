## Loading packages and relevant dependencies
## This code chunk will not be displayed on the R Markdown document
library(caret)
library(parallel)
library(doParallel)
library(caretEnsemble)
library(ranger)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

## Data loading
download.file('https://github.com/sgomezgomez/DevelopingDataProducts_FinalAssignment/raw/main/europeansoccerdatabase.zip', 
              method = 'curl', destfile = 'europeansoccerdatabase.zip')
unzip('europeansoccerdatabase.zip')
football = read.csv('FootballDataEurope.csv')

## Removing na values
football = football[complete.cases(football[, c('country_name', 'league_name', 'home_team', 'away_team','B365H', 'B365D', 'B365A', 'BSH', 'BSD', 'BSA', 'diff_goals')]),]
football = football[, c('country_name', 'league_name', 'home_team', 'away_team','B365H', 'B365D', 'B365A', 'BSH', 'BSD', 'BSA', 'diff_goals')]
football$result = football$diff_goals
football$result[football$result > 0] = 'home wins'
football$result[football$result < 0] = 'away wins'
football$result[football$result == 0] = 'draw'
football$result = as.factor(football$result)
football$diff_goals = as.factor(football$diff_goals)

## Model training and testing sets
## Setting seed for reproducibility purposes
set.seed(100)
## Creating model training partition
modTrainingPartition = createDataPartition(y = football$diff_goals, p = 0.6, list = FALSE)
fbmodeltraining = football[modTrainingPartition, ]
fbmodeltesting = football[-modTrainingPartition, ]


## Fitting proposed models for results
data = fbmodeltraining[, !names(fbmodeltraining) %in% c('diff_goals', 'country_name', 'league_name', 'home_team', 'away_team')]
##data = fbmodeltraining[, !names(fbmodeltraining) %in% c('diff_goals', 'country_name', 'league_name', 'home_team', 'away_team')]
## Cross validation settings : 10 folds repeat 5 times
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 3, 
        allowParallel = TRUE)
fitControlStacked = trainControl(method = 'cv', number = 10, classProbs = FALSE, 
        savePredictions = 'final', index = createFolds(data$result, 10), allowParallel = TRUE)
## RPART model
fitTreeResult = train(result ~ ., data = data, method = 'rpart', trControl = fitControl)
## Random Forest model
fitRFResult = train(result ~ ., data = data, method = 'rf', trControl = fitControl)
## Neural Network model
fitNNResult = train(result ~ ., data = data, method = 'nnet', trControl = fitControl)
## Linear Discriminant Analysis - LDA model
fitLDAResult = train(result ~ ., data = data, method = 'lda', trControl = fitControl)
## Ranger model
fitRangeresult = train(result ~ ., data = data, method = 'ranger', trControl = fitControl)
## K nearest neighbor model
fitKNNResult = train(result ~ ., data = data, method = 'knn', trControl = fitControl)
## SVM Radial model
fitSVMRadialesult = train(result ~ ., data = data, method = 'svmRadial', trControl = fitControl)
## Stacked models
##algorithmList = c('lda', 'rpart', 'nnet', 'ranger', 'rf', 'knn', 'svmRadial')
##fitStackedResult = caretList(result ~ ., data = data, metric = 'Accuracy',
##        methodList = algorithmList, trControl = fitControlStacked)
##stackResults = resamples(fitStackedResult)
##RFstacked = caretStack(fitStackedResult, method = 'rf', metric = 'Accuracy', trControl = fitControlStacked)

## Fitting proposed models for goal difference
##fitTreeGoalDiff = train(diff_goals ~ ., data = fbmodeltraining[, !names(fbmodeltraining) %in% c('result', 'country_name', 'league_name', 'home_team', 'away_team')],
##                      method = 'rpart', trControl = fitControl, na.action = na.exclude)
##fitRFGoalDiff = train(diff_goals ~ ., data = fbmodeltraining[, !names(fbmodeltraining) %in% c('result', 'country_name', 'league_name', 'home_team', 'away_team')], 
##              method = 'rf', trControl = fitControl)

## Predict training values for result
predTrainTreeResult = predict(fitTreeResult, newdata = fbmodeltraining, na.action = na.exclude)
predTrainRFResult = predict(fitRFResult, newdata = fbmodeltraining, na.action = na.exclude)
predTrainNNResult = predict(fitNNResult, newdata = fbmodeltraining, na.action = na.exclude)
predTrainLDAResult = predict(fitLDAResult, newdata = fbmodeltraining, na.action = na.exclude)
predTrainRangerResult = predict(fitRangeresult, newdata = fbmodeltraining, na.action = na.exclude)
predTrainKNNResult = predict(fitKNNResult, newdata = fbmodeltraining, na.action = na.exclude)
predTrainSVMRadialResult = predict(fitSVMRadialesult, newdata = fbmodeltraining, na.action = na.exclude)

## Stacked trained model
stackedTrainingPredictions = data.frame(matrix(NA, nrow = length(fbmodeltraining$result), ncol = 0))
stackedTrainingPredictions$result = fbmodeltraining$result
stackedTrainingPredictions$predTreeResult = predTrainTreeResult
stackedTrainingPredictions$predRFResult = predTrainRFResult
stackedTrainingPredictions$predNNResult = predTrainNNResult
stackedTrainingPredictions$predLDAResult = predTrainLDAResult
stackedTrainingPredictions$predRangerResult = predTrainRangerResult
stackedTrainingPredictions$predKNNResult = predTrainKNNResult
stackedTrainingPredictions$predSVMRadialResult = predTrainSVMRadialResult
fitRFStackedResult = train(result ~ ., data = stackedTrainingPredictions, method = 'rf', trControl = fitControl)
predTrainRFStackedResult = predict(fitRFStackedResult, newdata = stackedTrainingPredictions, na.action = na.exclude)

stopCluster(cluster)
registerDoSEQ()

## Metrics
metrics = data.frame()
confTrainTreeResult = confusionMatrix(predTrainTreeResult, fbmodeltraining$result)
metrics = rbind(metrics, c('Result', 'Training','RPart Tree',confTrainTreeResult$overall))
names(metrics) = c('Predicted Variable', 'Data Source', 'Model', names(confTrainTreeResult$overall))
confTrainRFResult = confusionMatrix(predTrainRFResult, fbmodeltraining$result)
metrics = rbind(metrics, c('Result', 'Training', 'Random Forests', confTrainRFResult$overall))
confTrainNNResult = confusionMatrix(predTrainNNResult, fbmodeltraining$result)
metrics = rbind(metrics, c('Result', 'Training', 'Neural Networks', confTrainNNResult$overall))
confTrainLDAResult = confusionMatrix(predTrainLDAResult, fbmodeltraining$result)
metrics = rbind(metrics, c('Result', 'Training', 'LDA', confTrainLDAResult$overall))
confTrainRangerResult = confusionMatrix(predTrainRangerResult, fbmodeltraining$result)
metrics = rbind(metrics, c('Result', 'Training', 'Ranger', confTrainRangerResult$overall))
confTrainKNNResult = confusionMatrix(predTrainKNNResult, fbmodeltraining$result)
metrics = rbind(metrics, c('Result', 'Training', 'KNN', confTrainKNNResult$overall))
confTrainSVMRadialResult = confusionMatrix(predTrainSVMRadialResult, fbmodeltraining$result)
metrics = rbind(metrics, c('Result', 'Training', 'SVM Radial', confTrainSVMRadialResult$overall))
confTrainRFStackedResult = confusionMatrix(predTrainRFStackedResult, fbmodeltraining$result)
metrics = rbind(metrics, c('Result', 'Training', 'Stacked Random Forest', confTrainRFStackedResult$overall))


## Predict testing values for result
predTestTreeResult = predict(fitTreeResult, newdata = fbmodeltesting, na.action = na.exclude)
predTestRFResult = predict(fitRFResult, newdata = fbmodeltesting, na.action = na.exclude)
predTestNNResult = predict(fitNNResult, newdata = fbmodeltesting, na.action = na.exclude)
predTestLDAResult = predict(fitLDAResult, newdata = fbmodeltesting, na.action = na.exclude)
predTestRangerResult = predict(fitRangeresult, newdata = fbmodeltesting, na.action = na.exclude)
predTestKNNResult = predict(fitKNNResult, newdata = fbmodeltesting, na.action = na.exclude)
predTestSVMRadialResult = predict(fitSVMRadialesult, newdata = fbmodeltesting, na.action = na.exclude)

## Stacked testing predictions
stackedTestingPredictions = data.frame(matrix(NA, nrow = length(fbmodeltesting$result), ncol = 0))
stackedTestingPredictions$result = fbmodeltesting$result
stackedTestingPredictions$predTreeResult = predTestTreeResult
stackedTestingPredictions$predRFResult = predTestRFResult
stackedTestingPredictions$predNNResult = predTestNNResult
stackedTestingPredictions$predLDAResult = predTestLDAResult
stackedTestingPredictions$predRangerResult = predTestRangerResult
stackedTestingPredictions$predKNNResult = predTestKNNResult
stackedTestingPredictions$predSVMRadialResult = predTestSVMRadialResult
predTestRFStackedResult = predict(fitRFStackedResult, newdata = stackedTestingPredictions, na.action = na.exclude)

## Metrics
confTestTreeResult = confusionMatrix(predTestTreeResult, fbmodeltesting$result)
metrics = rbind(metrics, c('Result', 'Testing', 'RPart Tree', confTestTreeResult$overall))
confTestRFResult = confusionMatrix(predTestRFResult, fbmodeltesting$result)
metrics = rbind(metrics, c('Result', 'Testing', 'Random Forests', confTestRFResult$overall))
confTestNNResult = confusionMatrix(predTestNNResult, fbmodeltesting$result)
metrics = rbind(metrics, c('Result', 'Testing', 'Neural Networks', confTestNNResult$overall))
confTestLDAResult = confusionMatrix(predTestLDAResult, fbmodeltesting$result)
metrics = rbind(metrics, c('Result', 'Testing', 'LDA', confTestLDAResult$overall))
confTestRangerResult = confusionMatrix(predTestRangerResult, fbmodeltesting$result)
metrics = rbind(metrics, c('Result', 'Testing', 'Ranger', confTestRangerResult$overall))
confTestKNNResult = confusionMatrix(predTestKNNResult, fbmodeltesting$result)
metrics = rbind(metrics, c('Result', 'Testing', 'KNN', confTestKNNResult$overall))
confTestSVMRadialResult = confusionMatrix(predTestSVMRadialResult, fbmodeltesting$result)
metrics = rbind(metrics, c('Result', 'Testing', 'SVM Radial', confTestSVMRadialResult$overall))
confTestRFStackedResult = confusionMatrix(predTestRFStackedResult, fbmodeltesting$result)
metrics = rbind(metrics, c('Result', 'Testing', 'Stacked Random Forest', confTestRFStackedResult$overall))



metrics

saveRDS(fitTreeResult, file = 'ShinyAppFinalAssignment/EuropeanFootballResultPredictionModel_FinalAssignment/fitTreeResult.rds')
saveRDS(fitRFResult, file = 'ShinyAppFinalAssignment/EuropeanFootballResultPredictionModel_FinalAssignment/fitRFResult.rds')
saveRDS(fitNNResult, file = 'ShinyAppFinalAssignment/EuropeanFootballResultPredictionModel_FinalAssignment/fitNNResult.rds')
saveRDS(fitLDAResult, file = 'ShinyAppFinalAssignment/EuropeanFootballResultPredictionModel_FinalAssignment/fitLDAResult.rds')
saveRDS(fitRangeresult, file = 'ShinyAppFinalAssignment/EuropeanFootballResultPredictionModel_FinalAssignment/fitRangeresult.rds')
saveRDS(fitKNNResult, file = 'ShinyAppFinalAssignment/EuropeanFootballResultPredictionModel_FinalAssignment/fitKNNResult.rds')
saveRDS(fitSVMRadialesult, file = 'ShinyAppFinalAssignment/EuropeanFootballResultPredictionModel_FinalAssignment/fitSVMRadialesult.rds')
saveRDS(fitRFStackedResult, file = 'ShinyAppFinalAssignment/EuropeanFootballResultPredictionModel_FinalAssignment/fitRFStackedResult.rds')