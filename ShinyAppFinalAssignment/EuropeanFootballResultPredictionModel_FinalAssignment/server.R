#
# This is the server logic of a Shiny web application. You can run the
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(caret)
library(parallel)
library(doParallel)
library(caretEnsemble)
library(ranger)
library(randomForest)
library(e1071)
library(kernlab)
library(ggplot2)

## Model object loading
fitTreeResult <- readRDS(file = 'fitTreeResult.rds',.GlobalEnv)
fitRFResult <- readRDS(file = 'fitRFResult.rds',.GlobalEnv)
fitNNResult <- readRDS(file = 'fitNNResult.rds',.GlobalEnv)
fitLDAResult <- readRDS(file = 'fitLDAResult.rds',.GlobalEnv)
fitRangeresult <- readRDS(file = 'fitRangeresult.rds',.GlobalEnv)
fitKNNResult <- readRDS(file = 'fitKNNResult.rds',.GlobalEnv)
fitSVMRadialesult <- readRDS(file = 'fitSVMRadialesult.rds',.GlobalEnv)
fitRFStackedResult <- readRDS(file = 'fitRFStackedResult.rds',.GlobalEnv)
set.seed(100)


# Define server logic required to draw a histogram
shinyServer(function(input, output) {
    
    ## Calculate predictions for all models
    predictions = eventReactive(input$calculate, {
        B365H = input$b365h
        B365D = input$b365d
        B365A = input$b365a
        BSH = input$bsh
        BSD = input$bsd
        BSA = input$bsa
        data = data.frame(B365H = B365H, B365D = B365D, B365A = B365A, 
                          BSH = BSH, BSD = BSD, BSA = BSA)
        rpartpred = (predict(fitTreeResult, newdata = data))
        rfpred = (predict(fitRFResult, newdata = data))
        nnetpred = (predict(fitNNResult, newdata = data))
        ldapred = (predict(fitLDAResult, newdata = data))
        rangerpred = (predict(fitRangeresult, newdata = data))
        knnpred = (predict(fitKNNResult, newdata = data))
        svmrpred = (predict(fitSVMRadialesult, newdata = data))
        stackedData = data.frame(predTreeResult = as.factor(rpartpred), 
            predRFResult = as.factor(rfpred), predNNResult = as.factor(nnetpred),
            predLDAResult = as.factor(ldapred), predRangerResult = as.factor(rangerpred),
            predKNNResult = as.factor(knnpred), predSVMRadialResult = as.factor(svmrpred))
        rfstackedpred = predict(fitRFStackedResult, newdata = stackedData)
        predictions = data.frame(rpartpred = rpartpred, rfpred = rfpred, nnetpred = nnetpred,
            ldapred = ldapred, rangerpred = rangerpred, knnpred = knnpred, svmrpred = svmrpred,
            rfstackedpred = rfstackedpred)
        return(predictions)
    })
    
    ## RPart Model prediction
    output$rpartpred = renderText({
        rpartpred = toupper(predictions()$rpartpred)
        return(rpartpred)
    })
    
    ## Random Forest Model prediction
    output$rfpred = renderText({
        rfpred = toupper(predictions()$rfpred)
        return(rfpred)
    })
    
    ## Neural Network Model prediction
    output$nnetpred = renderText({
        nnetpred = toupper(predictions()$nnetpred)
        return(nnetpred)
    })
    
    ## Linear Discriminant Analysis - LDA Model prediction
    output$ldapred = renderText({
        ldapred = toupper(predictions()$ldapred)
        return(ldapred)
    })
    
    ## Ranger Random Forest Model prediction
    output$rangerpred = renderText({
        rangerpred = toupper(predictions()$rangerpred)
        return(rangerpred)
    })
    
    ## KNN Model prediction
    output$knnpred = renderText({
        knnpred = toupper(predictions()$knnpred)
        return(knnpred)
    })
    
    ## SVM Radial prediction
    output$svmrpred = renderText({
        svmrpred = toupper(predictions()$svmrpred)
        return(svmrpred)
    })
    
    ## Stacked Random Forest Model prediction
    output$rfstackedpred = renderText({
        rfstackedpred = toupper(predictions()$rfstackedpred)
        return(rfstackedpred)
    })
    
    ## Selected model name
    output$selectedModelName = eventReactive(input$calculate, {
        if (input$selectedModel == 'rpart') {
            selectedModelName = 'RPART Decision Tree'
        } else if (input$selectedModel == 'rf') {
            selectedModelName = 'Random Forest'
        } else if (input$selectedModel == 'nnet') {
            selectedModelName = 'Neural Network'
        } else if (input$selectedModel == 'lda') {
            selectedModelName = 'Linear Discriminant Analysis'
        } else if (input$selectedModel == 'ranger') {
            selectedModelName = 'Ranger Random Forest'
        } else if (input$selectedModel == 'knn') {
            selectedModelName = 'K-Nearast Neighbors'
        } else if (input$selectedModel == 'svmradial') {
            selectedModelName = 'Radial Support Vector Machine'
        }else {
            selectedModelName = 'Stacked Ramdom Forest - Including all predictions'
        }
        return(selectedModelName)
    })
    
    ## Stacked Random Forest Model prediction
    output$selectedModelPrediction = eventReactive(input$calculate, {
        if (input$selectedModel == 'rpart') {
            selectedModelPrediction = toupper(predictions()$rpartpred)
        } else if (input$selectedModel == 'rf') {
            selectedModelPrediction = toupper(predictions()$rfpred)
        } else if (input$selectedModel == 'nnet') {
            selectedModelPrediction = toupper(predictions()$nnetpred)
        } else if (input$selectedModel == 'lda') {
            selectedModelPrediction = toupper(predictions()$ldapred)
        } else if (input$selectedModel == 'ranger') {
            selectedModelPrediction = toupper(predictions()$rangerpred)
        } else if (input$selectedModel == 'knn') {
            selectedModelPrediction = toupper(predictions()$knnpred)
        } else if (input$selectedModel == 'svmradial') {
            selectedModelPrediction = toupper(predictions()$svmrpred)
        }else {
            selectedModelPrediction = toupper(predictions()$rfstackedpred)
        }
        return(selectedModelPrediction)
    })
    
    ## Donwload RPart Model
    output$downloadRpartModel <- downloadHandler(
        filename = "fitTreeResult.rds",
        content = function(file) {
            file.copy("fitTreeResult.rds", file)
        }
    )
    
    ## Donwload Random Forest Model
    output$downloadRfModel <- downloadHandler(
        filename = "fitRFResult.rds",
        content = function(file) {
            file.copy("fitRFResult.rds", file)
        }
    )
    
    ## Donwload Neural Network Model
    output$downloadNNModel <- downloadHandler(
        filename = "fitNNResult.rds",
        content = function(file) {
            file.copy("fitNNResult.rds", file)
        }
    )
    
    ## Donwload LDA Model
    output$downloadLDAModel <- downloadHandler(
        filename = "fitLDAResult.rds",
        content = function(file) {
            file.copy("fitLDAResult.rds", file)
        }
    )
    
    ## Donwload Ranger Model
    output$downloadRangerModel <- downloadHandler(
        filename = "fitRangeresult.rds",
        content = function(file) {
            file.copy("fitRangeresult.rds", file)
        }
    )
    
    ## Donwload KNN Model
    output$downloadKNNModel <- downloadHandler(
        filename = "fitKNNResult.rds",
        content = function(file) {
            file.copy("fitKNNResult.rds", file)
        }
    )
    
    ## Donwload SVM Radial Model
    output$downloadSVMRadialModel <- downloadHandler(
        filename = "fitSVMRadialesult.rds",
        content = function(file) {
            file.copy("fitSVMRadialesult.rds", file)
        }
    )
    
    ## Donwload Stacked Random Forest Model
    output$downloadStackedRFModel <- downloadHandler(
        filename = "fitRFStackedResult.rds",
        content = function(file) {
            file.copy("fitRFStackedResult.rds", file)
        }
    )
    
    ## Donwload European Soccer Database set
    output$downloadData <- downloadHandler(
        filename = "europeansoccerdatabase.zip",
        content = function(file) {
            file.copy("europeansoccerdatabase.zip", file)
        }
    )

})
