Download and install the models code
  1. Download models code([Models.zip](https://pan.baidu.com/s/1jD-PZu3gvb8zIxJVzQ2aJg?pwd=xahj)
  2. Set up the running environment for each model
     Run the command: conda env create -f environment.yml
  3. Download the model [CheckPoints]（https://pan.baidu.com/s/1vIzeGLIzwnYkR9O2_d7Zxw?pwd=a199）
  4. Configure the CheckPoints path
     Modify the CheckPoints address in the original model code to the actual path where you placed the downloaded CheckPoints.
  5. Download the experimental datasets
     [MVTec-AD](https://pan.baidu.com/s/1lapl_AVc1S-weQrl1FLFIA?pwd=p3ej)
     [dtd](https://pan.baidu.com/s/1gHLDGFaNM0OjZg7xhApXjw?pwd=paw5)
     [R](https://pan.baidu.com/s/1SrwmsSpimfUKbGaw6DlYjw?pwd=g3ft)
  6. Configure the dataset path
     Modify the data address in the original model code to the actual path where you placed the downloaded datasets.
  7. Run the experiment to reproduce results
     Execute the command: bash test.sh
