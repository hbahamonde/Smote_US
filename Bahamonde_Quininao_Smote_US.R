############################## 
# Cleaning
##############################
cat("\014")
rm(list=ls())
graphics.off()
if (!require("pacman")) install.packages("pacman"); library(pacman)

# Import Performance Data
setwd("/Users/hectorbahamonde/research/Smote_US") # Hector


AUC3_lin = read.csv("https://github.com/hbahamonde/Smote_US/raw/main/Datos/AUC_lin3.csv", header = TRUE, sep = ",",  dec = ".", fill = TRUE, col.names="AUC3_lin");AUC3_lin$folds = 3;AUC3_lin$metric = "SVM Lineal"
AUC4_lin = read.csv("https://github.com/hbahamonde/Smote_US/raw/main/Datos/AUC_lin4.csv", header = TRUE, sep = ",",  dec = ".", fill = TRUE, col.names="AUC4_lin");AUC4_lin$folds = 3;AUC4_lin$metric = "SVM Lineal"
AUC5_lin = read.csv("https://github.com/hbahamonde/Smote_US/raw/main/Datos/AUC_lin5.csv", header = TRUE, sep = ",",  dec = ".", fill = TRUE, col.names="AUC5_lin");AUC5_lin$folds = 5; AUC5_lin$metric = "SVM Lineal"
AUC10_lin = read.csv("https://github.com/hbahamonde/Smote_US/raw/main/Datos/AUC_lin10.csv", header = TRUE, sep = ",",  dec = ".", fill = TRUE, col.names="AUC10_lin");AUC10_lin$folds = 10; AUC10_lin$metric = "SVM Lineal"


AUC3_mlp = read.csv("https://github.com/hbahamonde/Smote_US/raw/main/Datos/AUC_mlp3.csv", header = TRUE, sep = ",",  dec = ".", fill = TRUE, col.names="AUC3_MLP");AUC3_MLP$folds = 3;AUC3_MLP$metric = "SVM MLP"
AUC4_mlp = read.csv("https://github.com/hbahamonde/Smote_US/raw/main/Datos/AUC_mlp4.csv", header = TRUE, sep = ",",  dec = ".", fill = TRUE, col.names="AUC4_MLP");AUC4_MLP$folds = 3;AUC4_MLP$metric = "SVM MLP"
AUC5_mlp = read.csv("https://github.com/hbahamonde/Smote_US/raw/main/Datos/AUC_mlp5.csv", header = TRUE, sep = ",",  dec = ".", fill = TRUE, col.names="AUC5_MLP");AUC5_MLP$folds = 5; AUC5_MLP$metric = "SVM MLP"
AUC10_mlp = read.csv("https://github.com/hbahamonde/Smote_US/raw/main/Datos/AUC_mlp10.csv", header = TRUE, sep = ",",  dec = ".", fill = TRUE, col.names="AUC10_MLP");AUC10_MLP$folds = 10; AUC10_MLP$metric = "SVM MLP"################
#### ABSTRACT
################

## ---- abstract ----
fileConn <- file ("abstract.txt")
abstract.c = as.character(c("Abstract here."))
writeLines(abstract.c, fileConn)
close(fileConn)
## ----




## ---- abstract.length ----
abstract.c.l = sapply(strsplit(abstract.c, " "), length)
## ----
