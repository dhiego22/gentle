![alt text](gentle_icon.jpeg)

GENTLE is a user friendly Streamlit application that can generate features from TCR repertoire data, use feature selection algorithms to identify features with high predictive power and also create fast machine learning models. GENTLE also allows you to download dataframes, networks and classifier models for further analyses.

The GENTLE web application can be accessed on https://share.streamlit.io/dhiego22/gentle/main/gentle.py

## Running with Virtualenv

Inside the main folder, type the following commands:
 
  `pip install virtualenv`
  
  `virtualenv -p python3 venv_python3`
  
  `source venv_python3/bin/activate`
  
  `pip install -U -r requirements.txt`
  
  `streamlit run gentle`
  
  Then, open a web browser and type `localhost:8501`
   
## Running with Docker

Install Docker engine and Docker compose. Inside the main folder, type:

  `docker-compose up`
  
Then, open a web browser and type `localhost:8501`

## Quick guide

1. We provide two scripts to pre-process TCR data as GENTLE's input. The AIRR_script.py transforms AIRR data into the input data format for the data_preprocess.py script. Th data_preprocess.py script also can use data directly from the TCRdb website to produce a csv file to be used as GENTLE's input. The TRegs.csv and the TConvs.csv files in the test_data folder are the examples of the input data.
2. Upload a dataframe where the columns should be the TCR sequences and the rows should be the samples. The file should be in csv format or if it is too big, you can zip it. Observation: you must clear the cache (press 'c' or on the top-right options) when you upload a new dataframe.
3. Choose the feature that you want to analyse at the sidebar. 
4. Choose the normalization method to be used (optional).
5. Check the box at the sidebar to start feature selection process.
6. Choose the features generated for analyses. The feature selection methods will show a rank of the most predictive features. You can sort a column by clicking on it, once for ascending order, twice for descending order. If you choose two features, a 2D scatter plot will appear. If you choose three features, a 3D scatter plot will appear. 
7. Chose the number of splits and repeats for the stratified validation.
8. Check the box at the sidebar to train the model. You can choose four different classification methods. A radar plot with five scoring methods will appear for each classifier. You can download the created model as a pkl format.
9. Upload a second dataframe to validade the created model. A radar plot with the scoring methods will appear along with a confusion matrix. 

## Flowchart
![alt text](figs/flowchart.png)
  
## Bug report

- Please report bugs to dhiego@systemsbiomed.org








