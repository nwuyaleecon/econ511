library(mFilter)

this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)
nips_data <- read.csv(file = "./ps1_data.csv")
log_data <- log(nips_data)
filtered_data <- hpfilter(log_data, freq=1600)
