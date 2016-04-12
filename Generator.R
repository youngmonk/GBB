##### given a raw data for make-model-variant-ownership-city, this generates mileage / year buckets
####
###

#Make	Model	Variant	Ownership	City
#HYUNDAI	I20	Magna	1	BANGALORE
#HYUNDAI	I20	Era	1	BANGALORE

d=read.csv("bangalore-keys.csv")
l=list()
x=data.frame()

for(k in 1:nrow(d))
{
  for(i in 1:13)
  {
    for(j in 1:16)
    {
      if(j==1)
      {
        x=cbind(d[k,],(i+2003),1000)
        #colnames(x)=c("Make","Model","variant","city","own","y","odo")
      }
      else
        x=cbind(d[k,],(i+2003),10000*(j-1))
      colnames(x)=c("Make","Model","Variant","City","Ownership","Year","Out.Kms")
      l=rbind(l,x)
    }
  }  
  print(k)
}
a = as.data.frame(l)
write.csv(a,"sourcelist.csv",row.names=F)
