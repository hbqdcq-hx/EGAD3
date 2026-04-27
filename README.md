# ![Less-ls-More-Framework](./IMG/Framework.jpg)
  Evaluating the Consistency of Human Judgments Between State-of-the-Art Models on the MVTec AD Benchmark and Real-World Data
## 1. ![Stress Test Set Construction](./IMG/Images_Compare.jpg)
   Construct a stress testing set **R** that shares the same categories as MVTec AD, with a larger scale and richer content diversity.
## 2. Quantify Model ![Discrepancy](./IMG/MAE_Distance.jpg)
### 2.1. Download and deploy the models in the Models folder to obtain prediction scores for each image.
### 2.2. Calculate pairwise model discrepancies on **R** using the Maximum Discrepancy Competition (MDC).
### 2.3. Select the top 10 samples with the largest differences from each category to form the stress test set \(\mathcal{D}\), then compute the MAE between \(\mathcal{D}\) and MVTec AD.
## 3. Compute Global Ranking
### 3.1. Obtain the pairwise result matrix **P**.
### 3.2. Calculate the pairwise performance matrix **F**.
### 3.3. Use the Perron rank method to compute the global ranking of the models.
## 4. Verify Framework Stability
   Compare with random selection strategies and validate the stability of the framework using the Spearman correlation coefficient.



    
