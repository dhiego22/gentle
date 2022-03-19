#####################
#       IMPORTS
#####################

import streamlit as st

import numpy as np
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config
import math
import networkx as nx
from Levenshtein import *
import scipy.stats as ss
import dg3 
import time
import pickle
from datetime import datetime 


# Plots
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import seaborn as sns


# Machine Learning
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb




def page_title():
    """
        This function displays the title of the application
    """
    st.markdown(f'<h1 style="color:green;font-size:36px;">{"GENTLE: GENerator of T cell receptor repertoire features for machine LEarning algorithms classification"}</h1>',
                unsafe_allow_html=True)
    image = Image.open('gentle_icon_v2.1.jpeg')
    st.image(image, width=None, caption=' ')



def data_loading():
    """
        This function loads the dataframe to be analized
    """
    st.markdown(f'<h1 style="color:black;font-size:24px;">{"Upload dataset"}</h1>', unsafe_allow_html=True)

    delimiter = st.radio("Specify the delimiter used in the file", [",", ";", "tab", "space"])
    if delimiter == 'tab':
        delimiter = "   "
    elif delimiter == 'space':
        delimiter = " "

    file = st.file_uploader("The dataframe to be uploaded must have the rows as the samples (TCR repertoire) \
                            and the columns as the TCR sequences (amino acids) plus the target column. \
                            Please set the name of the target column as 'label'.\
                            In case your csv file exceed the 200MB size, you can load it as zip.")

    if file:
        if file.name.endswith('zip'):
            df = pd.read_csv(file, compression='zip', header=0, sep=delimiter, quotechar='"')
            st.dataframe(df)
            st.write('Number of columns (features): ', len(df.columns))
            st.write('Number of rows (samples): ', len(df))
            st.session_state['dataframe'] = df
            st.session_state['features_initializer'] = 1
                  
        elif file.name.endswith('csv'):
            df = pd.read_csv(file, sep=delimiter)
            st.dataframe(df)
            st.write('Number of columns (features): ', len(df.columns))
            st.write('Number of rows (samples): ', len(df))
            st.session_state['dataframe'] = df
            st.session_state['features_initializer'] = 1

        else:
            st.write('Specify your file format: .csv or .zip')
            pass



def biological_significance():
    """
        This function filters the the sequences according to a range size
    """
    st.markdown(f'<h1 style="color:blue;font-size:30px;">{"Biological significance filter"}</h1>', unsafe_allow_html=True)
    st.write('Some sequences may be too large or too small, becoming non significant in a biological point of view.\
        They may contain "*" and "_" as well. These last 2 is filtered automatically.\
              This option allow you to filter these sequences. We suggest as standards, sequences above 8 and below 20 amino acids.')
    sequences_range = st.slider('Select a range for the size of the sequences.', 4, 50, (8, 20))

    label = st.session_state['dataframe']['label']
    tcrs_list = st.session_state['dataframe'].columns.to_list()
    bio_sig_tcrs = []
    for s in tcrs_list:
      if len(s) >= sequences_range[0] and len(s) <= sequences_range[1] and '*' not in s and '_' not in s:
        bio_sig_tcrs.append(s)
    st.session_state['chosen_feature_dataframe'] = st.session_state['dataframe'][bio_sig_tcrs]

    st.dataframe(st.session_state['chosen_feature_dataframe'])
    st.write('Number of columns: ', len(st.session_state['chosen_feature_dataframe'].columns))
    st.write('Number of rows: ', len(st.session_state['chosen_feature_dataframe']))

#TODO
def abundunt_tcrs():
    pass

def clones_as_features():
    """
        This function keeps the TCR clones as features
    """
    st.markdown(f'<h1 style="color:black;font-size:20px;">{"Use the clones counts as features."}</h1>', unsafe_allow_html=True)
    st.session_state['chosen_feature_dataframe'] =  st.session_state['dataframe']
    st.session_state['chosen_feature_dataframe']['sample'] = st.session_state['dataframe'].index

    st.bar_chart(st.session_state['chosen_feature_dataframe'].drop(['sample', 'label'], axis=1))



def diversity_features():
    """
        This function creates a dataframe with diversity features
    """
    def shannon_index(tcrs_df):
        return -sum(n*math.log(2,n) for n in tcrs_df.iloc[:,0]  if n is not 0)

    def simpson_index(tcrs_df):
        return -sum(n**2 for n in tcrs_df.iloc[:,0] if n is not 0)  

    def pielou_index(tcrs_df):
        return simpson_index(tcrs_df) / math.log(len(tcrs_df))

    def hillnumbers_index(tcrs_df, alpha):
        return sum(n**alpha for n in tcrs_df.iloc[:,0] if n is not 0)**(1/1-alpha)

    def gini_index(tcrs_df):
        diff = 0
        for x in tcrs_df.iloc[:,0]:
            for y in tcrs_df.iloc[:,0]:
                diff += abs(x-y)
        return diff/2*(len(tcrs_df)**2)*np.mean(tcrs_df.to_numpy())


    st.header("Diversity features")
    st.write("-----------------------------------------------------------Richness Index----------------------------------------------------", color='r')
    st.latex(r''' H(X) = n ''')
    st.write("----------------------------------------------------------Shannon Index-------------------------------------------------------")
    st.latex(r''' H(X) = -\sum_{i=1}^{n} p(i)\log_2 p(i) ''')
    st.write("----------------------------------------------------------Simpson Index-------------------------------------------------------")
    st.latex(r''' H(X) = -\sum_{i=1}^{n} p(i)^2 ''')
    st.write("----------------------------------------------------------Pielou Index--------------------------------------------------------")
    st.latex(r''' H(X) = -\sum_{i=1}^{n} p(i)^2 / \log_2(n) ''')
    st.write("---------------------------------------------------------Hillnumbers Index----------------------------------------------------")
    st.latex(r''' H(X) = (\sum_{i=1}^{n} p(i)\alpha)^{1/1-\alpha} ''')
    st.write("---------------------------------------------------------Gini Index------------------------------------------------------------")
    st.latex(r''' H(X) = \sum_{i=1}^{n}\sum_{j=1}^{n} |p(i)-p(j)| / 2n^2\bar{p} ''')

    dfs = []
    name = []
    df = st.session_state['dataframe'].drop('label', axis=1).T
    label = [int(x) for x in st.session_state['dataframe']['label']]
    
    for c in df:
      name.append(c) 
      df_aux = pd.DataFrame(df[c])
      df_aux = df_aux[(df_aux.T > 0).any()]
      dfs.append(df_aux)

    richness = []
    shannon = []
    simpson = []
    pielou = []
    hillnumbers = []
    alpha = 1 # one by default, create option to choose this parameter****
    gini = []

    st.write('Calculating diversity features')
    my_bar = st.progress(0)
    for d, percent_complete in zip(dfs, range(0,100, int(100/len(dfs)))):
        richness.append(len(d))   
        shannon.append(shannon_index(d))
        simpson.append(simpson_index(d))
        pielou.append(pielou_index(d))
        hillnumbers.append(hillnumbers_index(d, alpha))
        gini.append(gini_index(d))
        my_bar.progress(percent_complete + 1)
    my_bar.progress(100)

    div_df = pd.DataFrame() 
    div_df['sample'] = name
    div_df['richness'] = richness
    div_df['shannon'] = shannon
    div_df['simpson'] = simpson
    div_df['pielou'] = pielou
    div_df['hillnumbers'] = hillnumbers
    div_df['gini'] = gini
    div_df['label'] = label

    # Show dataframe
    st.markdown(f'<h1 style="color:blue;font-size:24px;">{"DATAFRAME WITH DIVERSITY FEATURES"}</h1>', unsafe_allow_html=True)
    st.dataframe(div_df)
    st.write('Number of columns: ', len(div_df.columns))
    st.write('Number of rows: ', len(div_df))
    st.download_button("Press to Download DataFrame", div_df.to_csv().encode('utf-8'), "file.csv", "text/csv", key='download-csv')

    st.session_state['chosen_feature_dataframe'] = div_df



def network_features():
    """
        This function creates features based on network modeling
    """
    levenshtein_distance = 1
    distances = ['1','2','3','4']
    st.write('To create the graphs the Levenshtein distance is used to create the edges between the nodes')
    levenshtein_distance = int(st.selectbox("Select Levenshtein distance (default = 1):", distances))

    dfs = []
    name = []
    df = st.session_state['dataframe'].drop('label', axis=1).T
    label = [int(x) for x in st.session_state['dataframe']['label']]
    dfs = []
    for c in df:
      name.append(c) 
      df_aux = pd.DataFrame(df[c])
      df_aux = df_aux[(df_aux.T > 0).any()]
      dfs.append(df_aux)

    # Build network
    st.write('Building networks')
    my_bar = st.progress(0)    
    graphs=[]
    dic_names = {}
    dic_graphs = {}
    samples_names = []
    dic_nodes = {}
    dic_edges = {}
    percent_complete = 0
    for d,percent_complete in zip(dfs,range(0,100, int(100/len(dfs)))):
        my_bar.progress(percent_complete + 1)        
        samples_names.append(d.columns.to_list()[0][:-3])
        nodes = []    
        nodes_aux = []
        nodes_names = []
        edges = []
      
        seqs = d.index.to_list()
        G = nx.Graph()

        cont = 0  
        for s1 in seqs:
            cont +=1
            for s2 in seqs[cont:]:
                if distance(s1,s2) == levenshtein_distance:
                    if s1 not in nodes_aux:
                        nodes.append( Node(id=s1, size=100))
                        nodes_names.append(s1)
                    if s2 not in nodes_aux:
                        nodes.append( Node(id=s2, size=100))
                        nodes_names.append(s2)

                    nodes_aux.append(s1)
                    nodes_aux.append(s2)
                    list(set(nodes_aux))

                    G.add_edge(s1, s2)
                    edges.append( Edge(source=s1, target=s2))

        dic_nodes[samples_names[-1]] = nodes
        dic_edges[samples_names[-1]] = edges
        dic_graphs[samples_names[-1]] = pd.DataFrame(nx.to_numpy_array(G), columns= nodes_names, index= nodes_names)
        graphs.append(G)

    my_bar.progress(100)
    st.session_state['graphs'] = graphs

    st.markdown(f'<h1 style="color:blue;font-size:30px;">{"Interactive graph from one of the samples. Choose the sample you want to see"}</h1>', unsafe_allow_html=True)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
    st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)

    chosen_feature = st.radio(" ", sorted(samples_names))
    config = Config(width=1500, 
                height=1000, 
                directed=False,
                nodeHighlightBehavior=True, 
                highlightColor="#blue", # or "blue"
                collapsible=True,
                node={'labelProperty':'label'},
                link={'labelProperty': 'label', 'renderLabel': True},

                # **kwargs e.g. node_size=1000 or node_color="blue"
                ) 

    return_value = agraph(nodes=dic_nodes[chosen_feature], edges=dic_edges[chosen_feature], config=config)
    

    st.download_button("Press to Download Network", dic_graphs[chosen_feature].to_csv().encode('utf-8'), "network.txt", "text/csv", key='download-text')         

    st.write('Calculating network features')
    my_bar = st.progress(0)  
    clustering_coeficient = []
    arrows = []
    density = []
    eccentricity = []
    network_size = []
    trans = []
    percent_complete = 0
    for g, percent_complete in zip(graphs,range(0,100, int(100/len(graphs)))):
      my_bar.progress(percent_complete + 1)          
      arrows.append(len(g.edges))
      density.append(nx.density(g))
      clustering_coeficient.append(nx.average_clustering(g))
      network_size.append(len(g))
      trans.append(nx.transitivity(g))
    my_bar.progress(100)
     
    net_df = pd.DataFrame([])
    net_df['label'] = label
    net_df['sample'] = samples_names
    net_df['arrows'] = arrows
    net_df['density'] = density
    net_df['clustering_coeficient'] = clustering_coeficient  
    net_df['network_size'] = network_size  
    net_df['transitivity'] = trans 

    st.markdown(f'<h1 style="color:black;font-size:24px;">{"Dataframe with network features"}</h1>', unsafe_allow_html=True)
    st.dataframe(net_df.head())
    st.write('Number of columns: ', len(net_df.columns))
    st.write('Number of rows: ', len(net_df))
    st.download_button("Press to Download DataFrame", net_df.to_csv().encode('utf-8'), "file.csv", "text/csv", key='download-csv')

    st.session_state['chosen_feature_dataframe'] = net_df



def motifs_features():
    """
        This function creates new features based on the frequency of motifs
    """
    st.markdown(f'<h1 style="color:black;font-size:20px;">{"Window size = 1"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:red;font-size:20px;">{"D"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:red;font-size:20px;">{"D?F"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:black;font-size:20px;">{"Window size = 2"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:red;font-size:20px;">{"DF"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:red;font-size:20px;">{"D??F"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:black;font-size:20px;">{"Window size = 3"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:red;font-size:20px;">{"DFG"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:red;font-size:20px;">{"D???F"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:black;font-size:20px;">{"Window size = 4"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:red;font-size:20px;">{"DFGK"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:red;font-size:20px;">{"D????F"}</h1>', unsafe_allow_html=True)
    w_s = 1# window_size
    sizes = ['1','2','3','4']
    st.write('Choose the window size for the motifs calculation.\
              The window size refers to the number of contiguous amino acids and the number of amino acids between 2 target amino acids.')
    w_s = int(st.selectbox("Select window size (default = 1):", sizes))

    dfs = []
    name = []
    df = st.session_state['dataframe'].drop('label', axis=1).T
    label = [int(x) for x in st.session_state['dataframe']['label']]
    dfs = []
    for c in df:
      name.append(c) 
      df_aux = pd.DataFrame(df[c])
      df_aux = df_aux[(df_aux.T > 0).any()]
      dfs.append(df_aux)

    dfs2 = []
    for d in dfs:
        motif_dic = {}
        for window_size in range(1,w_s+1):
            for s in d.index:
                seq = s[3:-3]
                aux_list = []
                for i in range(len(seq) - window_size + 1):
                    aux_list.append(seq[i: i + window_size])
                for a in aux_list:
                    if a in motif_dic:
                        motif_dic[a] += 1
                    else:
                        motif_dic[a] = 1
            for s in d.index: 
                seq = s[3:-3]
                j = 0
                aux_list = []
                while (j + window_size) < len(seq)-1:
                    aux_list.append(seq[j] + str(window_size) + seq[j+window_size+1])
                    j += 1
                for a in aux_list:
                    if a in motif_dic:
                        motif_dic[a] += 1
                    else:
                        motif_dic[a] = 1

   
        aux = pd.DataFrame.from_dict(motif_dic, orient='index')
        aux = aux.rename({0: d.columns[0]}, axis=1)
        dfs2.append(aux.T)

    motif_df = pd.concat(dfs2)
    motif_df = motif_df.fillna(0)
    motif_df['label'] = label
    motif_df['sample'] = motif_df.index

    st.markdown(f'<h1 style="color:black;font-size:24px;">{"Dataframe with motif features"}</h1>', unsafe_allow_html=True)
    st.dataframe(motif_df.head())
    st.write('Number of columns: ', len(motif_df.columns))
    st.write('Number of rows: ', len(motif_df))
    st.download_button("Press to Download DataFrame", motif_df.to_csv().encode('utf-8'), "file.csv", "text/csv", key='download-csv')

    st.session_state['chosen_feature_dataframe'] = motif_df



def normalization():
    """
        This function performs normalization methods 
    """

    df = st.session_state['chosen_feature_dataframe']

    st.markdown(f'<h1 style="color:blue;font-size:30px;">{"Choose the normalization method you want to use"}</h1>', unsafe_allow_html=True)
    st.write('Usually, machine learning algorithms perform better with normalized data')
    norm = st.radio(" ", ["No normalization","Standard Scaler", "Min-Max Scaler", "Robust Scaler"])

    if norm == 'No normalization':
        shuffled_df = df.sample(frac=1)
        X = shuffled_df.drop(['label', 'sample'],axis=1)
        y = shuffled_df['label'].to_list()

    elif norm == 'Standard Scaler': 
        shuffled_df = df.sample(frac=1)
        aux = shuffled_df.drop(['label', 'sample'],axis=1)
        standarized_data = StandardScaler().fit_transform(aux)
        standarized_data = pd.DataFrame(standarized_data)
        standarized_data.index = aux.index
        standarized_data.columns = aux.columns
        X = standarized_data
        y = shuffled_df['label'].to_list()
        st.markdown(f'<h1 style="color:black;font-size:24px;">{"Dataframe normalized with Standard Scaler"}</h1>', unsafe_allow_html=True)
        st.dataframe(X.head())
        st.write('Number of columns: ', len(X.columns))
        st.write('Number of rows: ', len(X))
        st.download_button("Press to Download DataFrame", X.to_csv().encode('utf-8'), "file.csv", "text/csv", key='download-csv')

    elif norm == 'Min-Max Scaler': 
        shuffled_df = df.sample(frac=1)
        aux = shuffled_df.drop(['label', 'sample'],axis=1)
        standarized_data = MinMaxScaler().fit_transform(aux)
        standarized_data = pd.DataFrame(standarized_data)
        standarized_data.index = aux.index
        standarized_data.columns = aux.columns
        X = standarized_data
        y = shuffled_df['label'].to_list()
        st.markdown(f'<h1 style="color:black;font-size:24px;">{"Dataframe normalized with Min-Max Scaler"}</h1>', unsafe_allow_html=True)
        st.dataframe(X.head())
        st.write('Number of columns: ', len(X.columns))
        st.write('Number of rows: ', len(X))
        st.download_button("Press to Download DataFrame", X.to_csv().encode('utf-8'), "file.csv", "text/csv", key='download-csv')

    elif norm == 'Robust Scaler': 
        shuffled_df = df.sample(frac=1)
        aux = shuffled_df.drop(['label', 'sample'],axis=1)
        standarized_data = RobustScaler().fit_transform(aux)
        standarized_data = pd.DataFrame(standarized_data)
        standarized_data.index = aux.index
        standarized_data.columns = aux.columns
        X = standarized_data
        y = shuffled_df['label'].to_list()
        st.markdown(f'<h1 style="color:black;font-size:24px;">{"Dataframe normalized with Robust Scaler"}</h1>', unsafe_allow_html=True)
        st.dataframe(X.head())
        st.write('Number of columns: ', len(X.columns))
        st.write('Number of rows: ', len(X))
        st.download_button("Press to Download DataFrame", X.to_csv().encode('utf-8'), "file.csv", "text/csv", key='download-csv')

    return X, y



def feature_selection(X, y):
    """
        This function performs feature selection methods
    """
    st.markdown(f'<h1 style="color:blue;font-size:30px;">{"Choose the feature selection algorithm to be used"}</h1>', unsafe_allow_html=True)
    st.write('Some features may confuse the learning algorithm, thus the ML algorithm may learn better with specific features')
    f_s = 2
    n_f = []
    for i in range(2, int(len(X.columns)/2)+1):
        n_f.append(str(i))

    #n_f = ['2','3','4', '5', '6']
    st.write('Choose the number of features that you want to use. You can choose until half of the size of the features')
    f_s = int(st.selectbox("Select number of features (default = 2):", n_f))
    feat_select_method = st.radio(" ", ["No feature selection","Pearson's correlation", "Ridge", "XGBoost"])

          
    if feat_select_method == 'No feature selection':
        return X

    elif feat_select_method == "Pearson's correlation":  
        start_time = datetime.now()
        def cor_selector(X, y, num_feats):
            cor_list = []
            feature_name = X.columns.tolist()
            # calculate the correlation with y for each feature
            for i in X.columns.tolist():
                cor = np.corrcoef(X[i], y)[0, 1]
                cor_list.append(cor)
            # replace NaN with 0
            cor_list = [0 if np.isnan(i) else i for i in cor_list]
            # feature name
            cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
            # feature selection? 0 for not select, 1 for select
            cor_support = [True if i in cor_feature else False for i in feature_name]
            return cor_support, cor_feature

        cor_support, embeded_feature = cor_selector(X, y, f_s)

        st.markdown(f'<h1 style="color:black;font-size:24px;">{"Dataframe after using Person correlation for feature selection"}</h1>', unsafe_allow_html=True)
        st.dataframe(X[embeded_feature])
        st.write('Number of columns: ', len(X[embeded_feature].columns))
        st.write('Number of rows: ', len(X[embeded_feature]))
        time_elapsed = datetime.now() - start_time 
        st.write('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")
        st.download_button("Press to Download DataFrame", X[embeded_feature].to_csv().encode('utf-8'), "file.csv", "text/csv", key='DF_featureSelection-csv')

        return X[embeded_feature]

    elif feat_select_method == "Ridge":  
        start_time = datetime.now()
        embeded_selector = SelectFromModel(LogisticRegression(C=1, penalty='l2'), max_features=f_s)
        embeded_selector.fit(X, y)
        embeded_support = embeded_selector.get_support()
        embeded_feature = X.loc[:,embeded_support].columns.tolist()

        st.markdown(f'<h1 style="color:black;font-size:24px;">{"Dataframe after using LogisticRegression with l2 penalty for feature selection"}</h1>', unsafe_allow_html=True)
        st.dataframe(X[embeded_feature])
        st.write('Number of columns: ', len(X[embeded_feature].columns))
        st.write('Number of rows: ', len(X[embeded_feature]))
        time_elapsed = datetime.now() - start_time 
        st.write('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")
        st.download_button("Press to Download DataFrame", X[embeded_feature].to_csv().encode('utf-8'), "file.csv", "text/csv", key='DF_featureSelection-csv')

        return X[embeded_feature]


    elif feat_select_method == "XGBoost":  
        start_time = datetime.now() 
        embeded_selector = SelectFromModel(xgb.XGBClassifier(), max_features=f_s)
        embeded_selector.fit(X, y)
        embeded_support = embeded_selector.get_support()
        embeded_feature = X.loc[:,embeded_support].columns.tolist()
  
        st.markdown(f'<h1 style="color:black;font-size:24px;">{"Dataframe after using XGBClassifier for feature selection"}</h1>', unsafe_allow_html=True)
        st.dataframe(X[embeded_feature])
        st.write('Number of columns: ', len(X[embeded_feature].columns))
        st.write('Number of rows: ', len(X[embeded_feature]))
        time_elapsed = datetime.now() - start_time 
        st.write('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")
        st.download_button("Press to Download DataFrame", X[embeded_feature].to_csv().encode('utf-8'), "file.csv", "text/csv", key='DF_featureSelection-csv')

        return X[embeded_feature]



def machine_learning_algorithms(X, y):
    """
        This function performs classification methods
    """
   
    st.markdown(f'<h1 style="color:blue;font-size:30px;">{"Choose the classifier you want to run"}</h1>', unsafe_allow_html=True)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
    st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)

    ml_model = st.radio(" ", ['Gaussian Nayve Bayes', 'Linear Discriminant Analysis', 'K Nearest Neighbor', 'Logistic Regression'])

    if ml_model == 'Gaussian Nayve Bayes':      
        st.header('Gaussian Nayve Bayes')
        clf = GaussianNB()
        model_score(clf, 'GNB', X, y)

    elif ml_model == 'Linear Discriminant Analysis': 

        st.header('Linear Discriminant Analysis')
        clf = LinearDiscriminantAnalysis()
        model_score(clf, 'LDA', X, y)

    elif ml_model == 'K Nearest Neighbor': 

        st.header('K Nearest Neighbor')
        clf = KNeighborsClassifier()
        model_score(clf, 'KNN', X, y)

    elif ml_model == 'Logistic Regression': 

        st.header('Logistic Regression')
        clf = LogisticRegression()
        model_score(clf, 'LRE', X, y)

    st.write('Click the button to save the resulting classifier as in pickle file format')
    if st.button('Save Classifier'):
        file_name = "Classifier.pkl"
        with open(file_name, "wb") as open_file:
            pickle.dump(clf, open_file)
        st.write('File saved!')
                            


def model_score(classifier, classifier_name, X, y):
    """
        This function scores the built classifier
    """
    st.write('Choose the RepeatedStratifiedKFold parameters')
    n_spl = 2
    n_rep = 2
    n_spl = st.slider("Choose the parameter n_splits", 2, 5)
    n_rep = st.slider("Choose the parameter n_repeats", 2, 100)
    cv = RepeatedStratifiedKFold(n_splits=n_spl, n_repeats=n_rep)

    results_skfold_acc = cross_val_score(classifier, X, y, cv=cv,scoring='accuracy')  
    results_skfold_pre = cross_val_score(classifier, X, y, cv=cv,scoring='precision') 
    results_skfold_rec = cross_val_score(classifier, X, y, cv=cv,scoring='recall')      
    results_skfold_f1 = cross_val_score(classifier, X, y, cv=cv,scoring='f1')          
    results_skfold_auc = cross_val_score(classifier, X, y, cv=cv,scoring='roc_auc')

    sp = pd.DataFrame({
                    'group': ['Accuracy','Precision','Recall','F1', 'ROC AUC'],
                    classifier_name: [results_skfold_acc.mean(), 
                            results_skfold_pre.mean(),
                            results_skfold_rec.mean(), 
                            results_skfold_f1.mean(), 
                            results_skfold_auc.mean()] })


    col1, col2 = st.columns(2)
    with col1:
        st.write("Accuracy: ", results_skfold_acc.mean())
        st.write("SD: ", results_skfold_acc.std())
        st.write("Precision: ", results_skfold_pre.mean())
        st.write("SD: ", results_skfold_pre.std())
        st.write("Recall: ", results_skfold_rec.mean())
        st.write("SD: ", results_skfold_rec.std())
        st.write("F1: ", results_skfold_f1.mean())
        st.write("SD: ", results_skfold_f1.std())
        st.write('ROC AUC: ', results_skfold_auc.mean())
        st.write("SD: ", results_skfold_auc.std())        
    with col2:
        fig = px.line_polar(sp, r=classifier_name, theta='group', line_close=True)
        st.write(fig)



def GENTLE_Main():
    """
        Main function of the application
    """

    #st.set_page_config(layout="wide")
    st.set_page_config(layout="centered")

    st.session_state['features_initializer'] = 0
    if 'ml_initializer' not in st.session_state:
        st.session_state.ml_initializer = 0
   
    page_title()

    data_loading()
    
    if st.session_state['features_initializer']:
        
        biological_significance()

        st.markdown(f'<h1 style="color:blue;font-size:30px;">{"Choose the features you want to be created"}</h1>', unsafe_allow_html=True)

        st.chosen_feature = st.radio(" ", ["Clones", "Diversity", "Network", "Motif"])
        if st.chosen_feature == 'Clones':
            clones_as_features()
        elif st.chosen_feature == 'Diversity':
            diversity_features()
        elif st.chosen_feature == 'Network':
            network_features()
        elif st.chosen_feature == 'Motif':
            motifs_features()
    
        X, y = normalization()
        
        main_features = feature_selection(X, y)
 
        st.markdown(f'<h1 style="color:blue;font-size:30px;">{"Create a classifier based on the features you have chosen"}</h1>', unsafe_allow_html=True)
        if st.button('Create Classifier'):
            st.session_state.ml_initializer = 1

        if st.session_state.ml_initializer == 1:
            machine_learning_algorithms(main_features, y)



# Run the GENTLE
if __name__ == '__main__':

    GENTLE_Main()
