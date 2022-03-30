![alt text](gentle_icon.jpeg)

GENTLE is a user friendly Streamlit application that can generate features from TCR repertoire data, use feature selection algorithms to identify features with high predictive power and also create fast machine learning models. Gentle also allows you to download dataframes, networks and classifier models for further analyses.

## Running with Virtualenv

- Inside the main folder, type the following commands:
- 
  `pip install virtualenv`
  
  `virtualenv -p python3 venv_python3`
  
  `source venv_python3/bin/activate`
  
  `pip install -U -r requirements.txt`
  
  `streamlit run gentle`
  
  Then, open a web browser and type `localhost:8501`
   
## Running with Docker

- Install Docker engine and Docker compose. Inside the main folder, type:

  `docker-compose up`
  
Then, open a web browser and type `localhost:8501`

## Quick guide

1. Upload a dataframe where the columns should be the TCR sequences and the rows should be the samples. The file should be in csv format or if it is too big, you can zip it.
2. Choose the feature that you want to analyse at the sidebar. Depending on the feature, a parameter option regarding a specificity of the feature will appear. You can also choose multiple features. Due to code optimization, if you uncheck the boxes you have already checked you must clean th cache by pressing 'c' or on options in the top-right of the page.
3. Choose the normalization method to be used (optional).
4. Check the box at the sidebar to start feature selection process.
5. Choose the features that you want from the dataframe created with the features selected by the feature selection methods. You can sort a column by clicking on it, once for ascending order, twice for descending order. If you choose three features, a 3D scatter plot will appear. 
6. Check the box to validade the features with six classification methods. You can choose the number of splits and the number of repeats for the stratified k-fold method.
7. Finally, you can choose and download the created model.

## GENTLE flowchart
![alt text](flowchart.png)
  
## Bug report

- Please report bugs to dsouto49@gmail.com









