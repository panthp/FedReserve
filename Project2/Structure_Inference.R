#Import libraries
library(readr)
library(readxl)
library(wktmo)
library("seasonal")
library("zoo")
library("glasso")
library("pcalg")
library("igraph")
library(bnstruct)
library(TSA)
library(bnlearn)


#Clear Data
rm(list=ls())

Processed_Data <- read_csv("C:/Users/Panth/Desktop/Spring 2019/Data to Models/Project/Fred_Project/Project2/Processed_Data_PC.csv")

####PC####
copy_Processed_data = Processed_Data
copy_Processed_data = subset(copy_Processed_data, select = -c(DATE))
suffStat <- list(C = cor(copy_Processed_data), n = nrow(copy_Processed_data))
pc.data <- pc(suffStat, indepTest = gaussCItest, labels = names(copy_Processed_data), alpha = 0.05)
#Plot results of pc algorithm
plot(pc.data, main=paste("PC Estimated network with alpha: ", 0.05))
current_amat <- as(pc.data, "amat")
g_graph <- graph_from_adjacency_matrix(current_amat)
plot.igraph(g_graph,vertex.size=25 ,main=paste("PC Estimated network with alpha 0.05"))

####BN#### Need matrix?
#copy_Month_data = subset(copy_Processed_data, select = -c(RECPROUSM156N) )
#which(is.na(copy_Month_data), arr.ind=TRUE)
res = gs(copy_Processed_data)
plot(res)

rsmax2(copy_Processed_data, whitelist = NULL, blacklist = NULL, restrict = "si.hiton.pc",
       maximize = "hc", restrict.args = list(), maximize.args = list(), debug = FALSE)
#res2 = hc(copy_Processed_data, restart = 5, perturb=50, maxp = 2)
plot(res2)
#dataset <- BNDataset(data = Monthly_data,  discreteness = rep(F, ncol(Monthly_data)), variables = names(Monthly_data), node.sizes = rep(3, ncol(Monthly_data)))
#BN(Monthly_data)
#learn.network(Monthly_data)