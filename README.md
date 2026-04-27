# ![Less-ls-More-Framework](./IMG/Framework.jpg)
  Evaluate the changes in human judgment consistency between the SOTA model on the MVTec AD benchmark dataset and real data
## 1. ![R](./IMG/Images_Compare.jpg)
   Constructed a stress testing set R that is of the same category as MVTec AD, with a larger quantity and richer content
## 2. Quantify ![differences](./IMG/MAE_Distance.jpg)
### 2.1. Download and install the models in the Models folder to obtain the predicted scores for each image
### 2.2. Obtain pairwise model differences on R using MAXIMUM DISCREPANCY COMPETITION (MDC)
### 2.3. Select the top 10 maximum differences from each category to form the stress test set D, and then calculate the MAE of D and MVTec AD
## 3. Calculate global ranking
### 3.1. obtain the paired result matrix P
### 3.2. calculate the paired performance matrix F
### 3.3. use Perron rank to calculate the global ranking of the model



    
