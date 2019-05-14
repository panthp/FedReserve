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


#Clear Data
rm(list=ls())

#Import Data
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

####Convert Weekly to Monthly####
#weekly_unemp_ins_data = Weekly_data$ICNSA
#monthly_unemp_ins_data = weekToMonth(weekly_unemp_ins_data, year = 1968, wkIndex = 1, wkMethod = "ISO")
#monthly_unemp_ins_data = transform(monthly_unemp_ins_data, value = value/4)

####Add Converted Data to Monthly Dataset####
Monthly_data['Consumer_Expend']= monthly_consumer_expend_interpol_df['monthly_consumer_expend_interpol']
Monthly_data['Gross_Dom_Income_PC1'] = monthly_income_interpol_df['monthly_income_interpol']
Monthly_data['Natural_Rate_Unemp'] = monthly_nat_rate_unemp_interpol_df['monthly_nat_rate_unemp_interpol']
Monthly_data['SP_Close'] = GSPC['SP_Close']

#Clean Environment of temporary variables
rm(GSPC, monthly_consumer_expend, monthly_consumer_expend_interpol_df, monthly_income, monthly_income_interpol_df, monthly_nat_rate_unemp, monthly_nat_rate_unemp_interpol_df)

####Seasonal Data Processing (Adjust for Seasonansality####
#Convert to time series
ts_UnEmpClaims = ts(Monthly_data$M08297USM548NNBR, start = as.yearmon(Monthly_data$DATE[1]), freq = 12)
#Use seas to determine seasonality for subsequent prediction
seas_unempclaims = seas(ts_UnEmpClaims)
plot(ts_UnEmpClaims)
p1 = periodogram(ts_UnEmpClaims) #Determine frequency of seasonality
#Adjust for seasona
ts_UnEmpClaims_Adj = predict(seas_unempclaims, newdata = ts_UnEmpClaims)
p2 = periodogram(ts_UnEmpClaims_Adj)
plot(ts_UnEmpClaims_Adj) #Confirm Adjustment by Plotting

####Calcylatw Unemp Gap####
Monthly_data['Unemp_Gap'] = Monthly_data['Natural_Rate_Unemp'] - Monthly_data['UNRATE']
