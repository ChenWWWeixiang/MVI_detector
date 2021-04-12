library(pROC)
rt <- read.table("/mnt/data9/deep_R/pytorch-3dunet/s.txt",header = TRUE,sep=' ')
jpeg(file = "/mnt/data9/deep_R/pytorch-3dunet/radiomics_work/roc.jpg")
rocobj <- plot.roc(rt$Gt, rt$Pred,
                   main="Confidence intervals", percent=TRUE,
                   
                   ci=TRUE, # compute AUC (of AUC by default)
                   
                   print.auc=TRUE) # print the AUC (will contain the CI)
ciobj <- ci.se(rocobj, # CI of sensitivity
               
               specificities=seq(0, 100, 5)) # over a select set of specificities


plot(ciobj, type="shape", col="#1c61b6AA") # plot as a blue shape
plot(ci(rocobj, of="thresholds", thresholds="best")) # add one threshold

#plot(iris[,1],col="red")  ## 画图程序
dev.off();