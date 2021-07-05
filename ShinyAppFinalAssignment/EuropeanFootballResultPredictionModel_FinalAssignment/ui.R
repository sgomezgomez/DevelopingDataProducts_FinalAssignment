#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
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
library(e1071)
library(kernlab)
library(ggplot2)

##kaggle datasets download -d caesarlupum/betsstrategy

# Define UI for application that draws a histogram
shinyUI(fluidPage(

    # Application title
    titlePanel("Prediction Model for European Football Match Result"),
    h4('By Sebastian Gomez Gomez'),

    # Sidebar with a slider input for number of bins
    sidebarLayout(
        sidebarPanel(
            h3('Input predictor values'),
            span('Please fill all input fields below to calculate predictions'), br(), br(),
            numericInput(inputId = 'b365h', label = 'Bet365 home win odds*', min = 1, max = 1000, value = 0),
            numericInput(inputId = 'b365d', label = 'Bet365 draw odds*', min = 1, max = 1000, value = 0),
            numericInput(inputId = 'b365a', label = 'Bet365 away win odds*', min = 1, max = 1000, value = 0),
            numericInput(inputId = 'bsh', label = 'Blue Square home win odds*', min = 1, max = 1000, value = 0),
            numericInput(inputId = 'bsd', label = 'Blue Square draw odds*', min = 1, max = 30, value = 0),
            numericInput(inputId = 'bsa', label = 'Blue Square away win odds*', min = 1, max = 1000, value = 0),
            selectInput(inputId = 'selectedModel', label = 'Selected Model', selected = 'stackedrf',
                        choices = c('RPART Decision Tree' = 'rpart', 
                                    'Random Forest' = 'rf',
                                    'Neural Network' = 'nnet',
                                    'Linear Discriminant Analysis' = 'lda',
                                    'Ranger Random Forest' = 'ranger',
                                    'K-Nearast Neighbors' = 'knn',
                                    'Radial Support Vector Machine' = 'svmradial',
                                    'Stacked Ramdom Forest - Including all predictions' = 'stackedrf'), multiple = FALSE),
            # submit button
            actionButton('calculate', 'Calculate predictions', icon('refresh'),
                         class = 'btn btn-primary')
            ##submitButton('Calculate Predictions')
        ),

        # Show a plot of the generated distribution
        mainPanel(
            ##  - https://www.kaggle.com/caesarlupum/betsstrategy
            
            span('This page will predict the result (HOME WINS/DRAW/AWAY WINS) of a football/soccer match based on the different odds
                 published by two of the most famous betting sites: Bet 365 and Blue Square. In order to do this, the following 
                 machine learning algorithms were developed:'), br(),
            tags$div(tags$ul(
                tags$li(tags$span("RPart Decision Tree")),
                tags$li(tags$span("Random Forest")),
                tags$li(tags$span("Neural Network")),
                tags$li(tags$span("Linear Discriminant Analysis")),
                tags$li(tags$span("Ranger Random Forest")),
                tags$li(tags$span("K-Nearast Neighbors")),
                tags$li(tags$span("Radial Support Vector Machine")))),
            span('Additionally, a stacked Random Forest model was created using the predictions 
                 from all the previous algorithms.'), br(), br(),
            span('All models were created using the caret and ranger R packages, and trained/verified using the 
                 European Soccer Database available from kaggle 
                 (https://www.kaggle.com/caesarlupum/betsstrategy). Data can be downloaded by clicking on the button below:'), br(), br(),
            downloadButton("downloadData", "Download European Soccer Data"), br(), br(),
            span('Open each tab below to find predictions especifically from the selected model, predictions from all models,
                 or to download model files for each of the developed. Also, open the following link for more information regarding the 
                 actual process and code used:'),
            helpText(a("Click Here to open Slidify presentation", href="https://sgomezgomez.github.io/DevelopingDataProducts_FinalAssignment/FootballMatchPrediction_DevelopingDataProduct_ReproduciblePitchPresentation.html")), br(), br(),
            tabsetPanel(type = 'tabs',
                        tabPanel('Selected Model Prediction', br(), 
                                span('The prediction from the selected model is ...'), br(), br(),
                                span('Model: '), span(textOutput('selectedModelName')),
                                h3(textOutput('selectedModelPrediction'))
                                ),
                        tabPanel('All Model Predictions', br(),
                                span('Prediction from from all calculated models are ...'), br(), br(),
                                span('RPART Decision Tree:'),
                                h3(textOutput('rpartpred')),
                                span('Random Forest:'),
                                h3(textOutput('rfpred')),
                                span('Neural Network:'),
                                h3(textOutput('nnetpred')),
                                span('Linear Discriminant Analysis:'),
                                h3(textOutput('ldapred')),
                                span('Ranger Random Forest:'),
                                h3(textOutput('rangerpred')),
                                span('K-Nearast Neighbors:'),
                                h3(textOutput('knnpred')),
                                span('Radial Support Vector Machine:'),
                                h3(textOutput('svmrpred')),
                                span('Stacked Ramdom Forest - Including predictions from all the other models:'),
                                h3(textOutput('rfstackedpred'))
                                ),
                        tabPanel('Model Files', br(), 
                                span('Click to download the model files for reproducibility purposes:'), br(), br(),
                                h4('RPART Decision Tree'),
                                # Download Rpart Model Button
                                downloadButton("downloadRpartModel", "Download Rpart Model"),
                                h4('Random Forest'),
                                # Download Random Forest Model Button
                                downloadButton("downloadRfModel", "Download Random Forest Model"),
                                h4('Neural Network'),
                                # Download NN Model Button
                                downloadButton("downloadNNModel", "Download Neural Network Model"),
                                h4('Linear Discriminant Analysis'),
                                # Download LDA Model Button
                                downloadButton("downloadLDAModel", "Download LDA Model"),
                                h4('Ranger Random Forest'),
                                # Download Ranger Model Button
                                downloadButton("downloadRangerModel", "Download Ranger Model"),
                                h4('K-Nearast Neighbors'),
                                # Download KNN Model Button
                                downloadButton("downloadKNNModel", "Download KNN Model"),
                                h4('Radial Support Vector Machine'),
                                # Download SVM Radial Model Button
                                downloadButton("downloadSVMRadialModel", "Download SVM Radial Model"),
                                h4('Stacked Ramdom Forest - Including all predictions'),
                                # Download Stacked Random Forest Model Button
                                downloadButton("downloadStackedRFModel", "Download Stacked Random Forest Model")
                                )
                               
                        ),
            
            
        )
    )
))
