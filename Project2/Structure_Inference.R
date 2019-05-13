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
  library(tidyr)
  library(ggplot2)
  library(reshape2)
  library(huge)
  library(network)
  
  
  
  #Clear Data
  rm(list=ls())
  
  Processed_Data <- read_csv("C:/Users/Panth/Desktop/Spring 2019/Data to Models/Project/Fred_Project/Project2/Processed_Data_PC.csv")
  
  ####Gaussian Check####
  copy_Processed_data = Processed_Data
  copy_Processed_data = subset(copy_Processed_data, select = -c(DATE))
  d <- melt(copy_Processed_data[])
  ggplot(d,aes(x = value)) + 
    facet_wrap(~variable,scales = "free_x") + 
    geom_histogram()
  
  #NonParanormal Transform
  NonParaTrans = huge.npn(copy_Processed_data)#, npn.func = "skeptic")
  NonParaTrans = as.data.frame(NonParaTrans)
  d <- melt(NonParaTrans)
  ggplot(d,aes(x = value)) + 
    facet_wrap(~variable,scales = "free_x") + 
    geom_histogram(binwidth = 0.50)
  
  ####PC####
  copy_Processed_data = NonParaTrans
  copy_Processed_data = subset(copy_Processed_data, select = -c(DATE))
  suffStat <- list(C = cor(copy_Processed_data), n = nrow(copy_Processed_data))
  pc.data <- pc(suffStat, indepTest = gaussCItest, labels = names(copy_Processed_data), alpha = 0.05)
  #Plot results of pc algorithm
  plot(pc.data, main=paste("PC Estimated network with alpha: ", 0.05))
  current_amat <- as(pc.data, "amat")
  g_graph <- graph_from_adjacency_matrix(current_amat)
  plot.igraph(g_graph,vertex.size=25 ,main=paste("PC Estimated Network with Alpha=0.05"))
  
  ####BN#### Need matrix?
  #copy_Month_data = subset(copy_Processed_data, select = -c(RECPROUSM156N) )
  #which(is.na(copy_Month_data), arr.ind=TRUE)
  res = gs(copy_Processed_data)
  plot(res)
  
  res2 = rsmax2(copy_Processed_data, whitelist = NULL, blacklist = NULL, restrict = "si.hiton.pc",
         maximize = "hc", restrict.args = list(), maximize.args = list(), debug = FALSE)
  #res2 = hc(copy_Processed_data, restart = 5, perturb=50, maxp = 2)
  plot(res2)
  #dataset <- BNDataset(data = Monthly_data,  discreteness = rep(F, ncol(Monthly_data)), variables = names(Monthly_data), node.sizes = rep(3, ncol(Monthly_data)))
  #BN(Monthly_data)
  #learn.network(Monthly_data)
  
  
  ####Glasso####
  copy_Processed_data = Processed_Data
  copy_Processed_data = subset(copy_Processed_data, select = -c(DATE))
  data_var <- var(copy_Processed_data) #Compute covariance of data
  
  #compute best rho (BIC) (using: https://www4.stat.ncsu.edu/~reich/BigData/code/glasso.html)
  nr  <- 100
  rho <- seq(0.1,1,length=nr)
  bic <- rho
  for(j in 1:nr){
    a       <- glasso(data_var,rho[j])
    p_off_d <- sum(a$wi!=0 & col(data_var)<row(data_var))
    bic[j]  <- -2*(a$loglik) + p_off_d*log(500)
  }
  best <- which.min(bic)
  plot(rho,bic,main=paste("Tuning Parameter for Glasso:", rho[best]))
  points(rho[best],bic[best],pch=19)
  best_rho = formatC(rho[best], digits = 5, format = "f")
  
  #Perform glasso with best rho, Plot Network
  glasso_data<-glasso(data_var, rho[best])
  round(glasso_data$w, 2)
  round(glasso_data$wi, 2)
  P <- a$wi #precision
  A <- ifelse(P!=0 & row(P)!=col(P),1,0) #Adjacency matrix
  g <- network(A)
  g_graph_one <- graph_from_adjacency_matrix(A)
  #Output graph
  label_vert = names(copy_Processed_data)
  plot.igraph(g_graph_one, label=label_vert,edge.arrow.size=0.01,main=paste("Glasso Estimated network with rho:", best_rho))
  plot(g,label=label_vert,main=paste("Glasso Estimated network with rho: ", best_rho))
  #plot(g,label=1:20,main=paste("Glasso Estimated network with rho: ", best_rho))
