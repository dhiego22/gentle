![alt text](gentle_icon.jpeg)

GENTLE is a user friendly Streamlit application that can generate features from TCR repertoire data, use feature selection algorithms to identify features with high predictive power and also create fast machine learning models. Gentle also allows you to download dataframes, networks and classifier models for further analyses.

## Running with a virtualenv

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
  
## Bug report

- Please report bugs to dsouto49@gmail.com
