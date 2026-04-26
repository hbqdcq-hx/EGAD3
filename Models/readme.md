## Download and install the models
1. Download [models](https://pan.baidu.com/s/1jD-PZu3gvb8zIxJVzQ2aJg?pwd=xahj) code
2. Set up the running environment for each model
   ```
   conda env create -f environment.yml
   ```
4. Download the model [CheckPoints](https://pan.baidu.com/s/1vIzeGLIzwnYkR9O2_d7Zxw?pwd=a199)
5. Configure the CheckPoints path
   Modify the CheckPoints address in the original model code to the actual path where you placed the downloaded CheckPoints.
6. Download the experimental datasets
   - [MVTec-AD](https://pan.baidu.com/s/1lapl_AVc1S-weQrl1FLFIA?pwd=p3ej)
   - [dtd](https://pan.baidu.com/s/1gHLDGFaNM0OjZg7xhApXjw?pwd=paw5)
   - [R](https://pan.baidu.com/s/1SrwmsSpimfUKbGaw6DlYjw?pwd=g3ft)
7. Configure the dataset path
   Modify the data address in the original model code to the actual path where you placed the downloaded datasets.
8. Run the experiment to reproduce results
   ```
   bash test.sh
   ```
