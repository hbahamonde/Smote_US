############################## 
# Cleaning
##############################


################
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
