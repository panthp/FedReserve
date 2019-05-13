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

GSPC <- read_csv("C:/Users/Panth/Desktop/Spring 2019/Data to Models/Project/Fred_Project/Data/GSPC.csv")
Monthly_data<- read_excel("C:/Users/Panth/Desktop/Spring 2019/Data to Models/Project/Fred_Project/Data/5919.xls", sheet = "Monthly")
Quarterly_data<- read_excel("C:/Users/Panth/Desktop/Spring 2019/Data to Models/Project/Fred_Project/Data/5919.xls", sheet = "Quarterly")
Weekly_data<- read_excel("C:/Users/Panth/Desktop/Spring 2019/Data to Models/Project/Fred_Project/Data/5919.xls", sheet = "Weekly,_Ending_Saturday")

####Convert Quarterly to Monthly####
monthly_seq = seq(Quarterly_data$DATE[1], tail(Quarterly_data$DATE,1), by="month")
Quarterly_data = transform(Quarterly_data, PCECC96 = PCECC96/3)
#Consumer spend
monthly_consumer_expend = Quarterly_data[c("DATE", "PCECC96")]
monthly_consumer_expend_interpol_df = data.frame(DATE=monthly_seq, monthly_consumer_expend_interpol=spline(monthly_consumer_expend, method = "fmm", xout=monthly_seq)$y)
#Income
monthly_income = Quarterly_data[c("DATE", "A4102C1Q027SBEA_PC1")]
monthly_income_interpol_df = data.frame(DATE=monthly_seq, monthly_income_interpol=spline(monthly_income, method = "fmm", xout=monthly_seq)$y)
#Nat Rate Unemp
monthly_nat_rate_unemp = Quarterly_data[c("DATE", "NROU")]
monthly_nat_rate_unemp_interpol_df = data.frame(DATE=monthly_seq, monthly_nat_rate_unemp_interpol=spline(monthly_nat_rate_unemp, method = "fmm", xout=monthly_seq)$y)
####PC####
copy_Month_data = Monthly_data
copy_Month_data = subset(copy_Month_data, select = -c(DATE) )
suffStat <- list(C = cor(copy_Month_data), n = nrow(copy_Month_data))
pc.data <- pc(suffStat, indepTest = gaussCItest, labels = names(copy_Month_data), alpha = 0.05)
#Plot results of pc algorithm
plot(pc.data, main=paste("PC Estimated network with alpha: ", 0.05))
current_amat <- as(pc.data, "amat")
g_graph <- graph_from_adjacency_matrix(current_amat)
plot.igraph(g_graph,vertex.size=25 ,main=paste("PC Estimated network with alpha 0.05"))

####BN#### Need matrix?
copy_Month_data = subset(copy_Month_data, select = -c(RECPROUSM156N) )
#which(is.na(copy_Month_data), arr.ind=TRUE)
res = gs(copy_Month_data)
plot(res)
#dataset <- BNDataset(data = Monthly_data,  discreteness = rep(F, ncol(Monthly_data)), variables = names(Monthly_data), node.sizes = rep(3, ncol(Monthly_data)))
#BN(Monthly_data)
#learn.network(Monthly_data)

####Convert Weekly to Monthly####
#weekly_unemp_ins_data = Weekly_data$ICNSA
#monthly_unemp_ins_data = weekToMonth(weekly_unemp_ins_data, year = 1968, wkIndex = 1, wkMethod = "ISO")
#monthly_unemp_ins_data = transform(monthly_unemp_ins_data, value = value/4)

####Add Data to Monthly####
Monthly_data['Consumer_Expend']= monthly_consumer_expend_interpol_df['monthly_consumer_expend_interpol']
Monthly_data['Gross_Dom_Income_PC1'] = monthly_income_interpol_df['monthly_income_interpol']
Monthly_data['Natural_Rate_Unemp'] = monthly_nat_rate_unemp_interpol_df['monthly_nat_rate_unemp_interpol']
Monthly_data['SP_Close'] = GSPC['SP_Close']

#Clean Environment
rm(GSPC, monthly_consumer_expend, monthly_consumer_expend_interpol_df, monthly_income, monthly_income_interpol_df, monthly_nat_rate_unemp, monthly_nat_rate_unemp_interpol_df)

####Seasonal Data Processing####
ts_UnEmpClaims = ts(Monthly_data$M08297USM548NNBR, start = as.yearmon(Monthly_data$DATE[1]), freq = 12)
seas_unempclaims = seas(ts_UnEmpClaims)
plot(ts_UnEmpClaims)
p1 = periodogram(ts_UnEmpClaims)
ts_UnEmpClaims_Adj = predict(seas_unempclaims, newdata = ts_UnEmpClaims)
p2 = periodogram(ts_UnEmpClaims_Adj)
plot(ts_UnEmpClaims_Adj)

#Unemp Gap#
Monthly_data['Unemp_Gap'] = Monthly_data['Natural_Rate_Unemp'] - Monthly_data['UNRATE']

####PC####
copy_Month_data = Monthly_data
copy_Month_data = subset(copy_Month_data, select = -c(DATE) )
suffStat <- list(C = cor(copy_Month_data), n = nrow(copy_Month_data))
pc.data <- pc(suffStat, indepTest = gaussCItest, labels = names(copy_Month_data), alpha = 0.05)
#Plot results of pc algorithm
plot(pc.data, main=paste("PC Estimated network with alpha: ", 0.05))
current_amat <- as(pc.data, "amat")
g_graph <- graph_from_adjacency_matrix(current_amat)
plot.igraph(g_graph,vertex.size=25 ,main=paste("PC Estimated network with alpha 0.05"))

####BN#### Need matrix?
copy_Month_data = subset(copy_Month_data, select = -c(RECPROUSM156N) )
#which(is.na(copy_Month_data), arr.ind=TRUE)
res = gs(copy_Month_data)
plot(res)
#dataset <- BNDataset(data = Monthly_data,  discreteness = rep(F, ncol(Monthly_data)), variables = names(Monthly_data), node.sizes = rep(3, ncol(Monthly_data)))
#BN(Monthly_data)
#learn.network(Monthly_data)