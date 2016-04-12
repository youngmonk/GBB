##1. Install R
##2. Install R Studio
##3. Run the code with the comments mentioned in the code.

#-------------------------------------------------------#

install.packages("e1071") ###ONLY TO BE RUN FIRST TIME TO INSTALL THE SVM LIBRARY

options(stringsAsFactors=FALSE)
library("e1071") # calling the SVM LIBRARY

	 ## Put all your files in this directory, by default 'My Documents'


txn=read.csv("TransactionDataCompiledFinal.csv",header=T) # compiled transaction data

#converts various columns to UPPER and trim whitespaces
txn$Model=toupper(trimws(txn$Model))
txn$Variant=toupper(trimws(txn$Variant))
txn$City=toupper(txn$City)

txn$key=paste0(txn$Model,"$",txn$Variant,"$",txn$City) # concatenate to form the key

val=read.csv("sourcelist.csv") # read the data where we need the predictions to be made

#converts various columns to UPPER and trim whitespaces
val$Model=toupper(trimws(val$Model))
val$Variant=toupper(trimws(val$Variant))
val$City=toupper(val$City)

val$key=paste0(val$Model,"$",val$Variant,"$",val$City) # concatenate to form the key
val$Age=2016-val$Year #Edit to add month

val$predPrice=0 # adds a column with all 0's

for (i in 1:nrow(val))
{
  i1=val$key[i]
  d=subset(txn,key==i1) # subsetting the Transaction data according to the key
  
  # check for atleast 15 transaction values, if not then skip that particular row
  if(nrow(d)<15)
    next
  
  mod=svm(Sold.Price ~ Year + Ownership + Out.Kms + Age, data=d) # calling the SVM model
  pred=predict(mod,val[i,]) #predicting the price
  val$predPrice[i]=round(pred) # merging it to the file
  
}

write.csv(val,"result.csv",row.names = F) # output the csv to the local drive