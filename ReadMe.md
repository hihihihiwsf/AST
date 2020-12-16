## 1. Download data
electricity: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams 20112014
traffic: https://archive.ics.uci.edu/ml/datasets/PEMS-SF
solar data： https://www.nrel.gov/grid/solar-power-data.html
wind: https://www.kaggle.com/sohier/30-years-of-european-wind-generation 
## 2. Preprocess data
` python preprocess_data.py `

## 3. Train and evaluate the model
` python train_gan.py --dataset elect --model test `
--dataset The name of the preprocessed data
--model the position of the specific params.json file under the folder of "experiments".