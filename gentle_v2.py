#####################
#       IMPORTS
#####################



# General
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import numpy as np
import pandas as pd
import math
import networkx as nx
import networkx.algorithms.community as nx_comm
from Levenshtein import *
import scipy.stats as ss
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
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import umap
import mrmr



def page_title():
    """
        This function displays the title of the application
    """
    st.markdown(f'<h1 style="color:green;font-size:36px;">{"GENTLE: GENerator of T cell receptor repertoire features for machine LEarning algorithms selection and classification"}</h1>',
                unsafe_allow_html=True)
    image = Image.open('gentle_icon_v2.1.jpeg')
    st.image(image, width=None, caption=' ')



def data_loading():
    """
        This function loads the dataframe to be analized
    """

    start_time = datetime.now()
    st.markdown(f'<h1 style="color:blue;font-size:24px;">{"Upload dataset"}</h1>', unsafe_allow_html=True)

    delimiter = st.radio("Specify the delimiter used in the file", [",", ";", "tab", "space"])
    if delimiter == 'tab':
        delimiter = "   "
    elif delimiter == 'space':
        delimiter = " "

    file = st.file_uploader("The dataframe to be uploaded must have the rows as the samples (TCR repertoire) \
                            and the columns as the TCR sequences (amino acids) plus the target column. \
                            Please set the name of the target column as 'label'.\
                            In case your csv file exceed the 200MB size, you can load it as zip.")

    start_time = datetime.now()
    if file:
        @st.cache(allow_output_mutation=True)
        def get_data(file, delimiter, extension):
            """
            Auxiliar function to avoid reloading dataframe when parameters are changed
            """
            if extension == 'csv':
                return pd.read_csv(file, sep=delimiter)
            elif extension == 'zip':
                return pd.read_csv(file, compression='zip', header=0, sep=delimiter, quotechar='"')

        if file.name.endswith('zip'):
            st.session_state['input_dataframe'] = get_data(file, delimiter, 'zip')
            le = LabelEncoder()
            st.session_state['input_dataframe']['label_transformed'] = le.fit_transform(st.session_state['input_dataframe']['label'])
                 
        elif file.name.endswith('csv'):
            st.session_state['input_dataframe'] = get_data(file, delimiter, 'csv')
            le = LabelEncoder()
            st.session_state['input_dataframe']['label_transformed'] = le.fit_transform(st.session_state['input_dataframe']['label'])

        else:
            st.write('Specify your file format: .csv or .zip')
            pass
    
        st.dataframe(st.session_state['input_dataframe'])
        st.session_state['features_initializer'] = 1
        st.write('Uploaded dataframe has ', len(st.session_state['input_dataframe'].columns), 'columns (features) and ', len(st.session_state['input_dataframe']), ' rows (samples)')
        time_elapsed = datetime.now() - start_time 
        st.write('Time elapsed for file upload (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")


        side_bar()



def clones_features(clones):
    """
        This function keeps the main dataframe as features
    """
    st.markdown(f'<h1 style="color:black;font-size:24px;">{"Using uploaded dataframe with the raw clones as features"}</h1>', unsafe_allow_html=True)
    st.session_state['clones'] = st.session_state['input_dataframe'].drop(['label', 'label_transformed'], axis=1)
    st.session_state['clones']['sample'] = st.session_state['input_dataframe'].index
    st.session_state['clones']['label'] = st.session_state['input_dataframe']['label_transformed']



def diversity_features(diversity):
    """
        This function creates a dataframe with diversity features
    """
    st.markdown(f'<h1 style="color:red;font-size:30px;">{"Diversity Features"}</h1>', unsafe_allow_html=True)
    start_time = datetime.now()
    @st.cache
    def shannon_index(tcrs_df):
        return -sum(n*math.log(2,n) for n in tcrs_df.iloc[:,0]  if n is not 0)

    @st.cache
    def simpson_index(tcrs_df):
        return -sum(n**2 for n in tcrs_df.iloc[:,0] if n is not 0)  

    @st.cache
    def inverse_simpson_index(tcrs_df):
        return 1/sum(n**2 for n in tcrs_df.iloc[:,0] if n is not 0) 

    @st.cache
    def pielou_index(tcrs_df):
        return simpson_index(tcrs_df) / math.log(len(tcrs_df))

    @st.cache
    def one_minus_pielou_index(tcrs_df):
        return simpson_index(tcrs_df) / math.log(len(tcrs_df))

    @st.cache
    def hillnumbers_index(tcrs_df, alpha):
        return sum(n**alpha for n in tcrs_df.iloc[:,0] if n is not 0)**(1/1-alpha)

    @st.cache
    def gini_index(tcrs_df):
        diff = 0
        for x in tcrs_df.iloc[:,0]:
            for y in tcrs_df.iloc[:,0]:
                diff += abs(x-y)
        return diff/2*(len(tcrs_df)**2)*np.mean(tcrs_df.to_numpy())


    dfs = []
    name = []
    df = st.session_state['input_dataframe'].drop(['label', 'label_transformed'], axis=1).T
        
    for c in df:
      name.append(c) 
      df_aux = pd.DataFrame(df[c])
      df_aux = df_aux[(df_aux.T > 0).any()]
      dfs.append(df_aux)

    st.markdown(f'<h1 style="color:blue;font-size:24px;">{"Calculating diversity features"}</h1>', unsafe_allow_html=True)
    @st.cache(suppress_st_warning=True)
    def diversity_df():
        richness = []
        shannon = []
        simpson = []
        inverse_simpson = []
        pielou = []
        one_minus_pielou = []
        hillnumbers = []
        alpha = 1 # one by default, create option to choose this parameter****
        gini = []
        my_bar = st.progress(0)
        for d, percent_complete in zip(dfs, range(0,100, int(100/len(dfs)))):
            richness.append(len(d))   
            shannon.append(shannon_index(d))
            simpson.append(simpson_index(d))
            inverse_simpson.append(inverse_simpson_index(d))
            pielou.append(pielou_index(d))
            one_minus_pielou.append(one_minus_pielou_index(d))
            hillnumbers.append(hillnumbers_index(d, alpha))
            gini.append(gini_index(d))
            my_bar.progress(percent_complete + 1)
        my_bar.progress(100)

        return richness, shannon, simpson, inverse_simpson, pielou, one_minus_pielou, hillnumbers, gini

    richness, shannon, simpson, inverse_simpson, pielou, one_minus_pielou, hillnumbers, gini = diversity_df()

    st.session_state['diversity'] = pd.DataFrame() 
    st.session_state['diversity']['sample'] = st.session_state['input_dataframe'].index
    st.session_state['diversity']['richness'] = richness
    st.session_state['diversity']['shannon'] = shannon
    st.session_state['diversity']['simpson'] = simpson
    st.session_state['diversity']['inverse_simpson'] = inverse_simpson
    st.session_state['diversity']['pielou'] = pielou
    st.session_state['diversity']['one_minus_pielou'] = one_minus_pielou    
    st.session_state['diversity']['hillnumbers'] = hillnumbers
    st.session_state['diversity']['gini'] = gini
    st.session_state['diversity']['label'] = st.session_state['input_dataframe']['label_transformed']

    st.markdown(f'<h1 style="color:black;font-size:24px;">{"Dataframe with diversity features"}</h1>', unsafe_allow_html=True)
    st.dataframe(st.session_state['diversity'])
    st.write('Uploaded dataframe has ', len(st.session_state['diversity'].columns), 'columns (features) and ', len(st.session_state['diversity']), ' rows (samples)')
    st.download_button("Press the button to download dataframe with diversity features", st.session_state['diversity'].to_csv().encode('utf-8'), "file.csv", "text/csv", key='download-csv')
    st.markdown(f'<h1 style="color:green;font-size:20px;">{"Features created!"}</h1>', unsafe_allow_html=True)
    time_elapsed = datetime.now() - start_time 
    st.write('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")


def network_features(network):
    """
        This function creates features based on network modeling
    """
    st.markdown(f'<h1 style="color:red;font-size:30px;">{"Network Features"}</h1>', unsafe_allow_html=True)
    start_time = datetime.now()
    distances = ['1','2','3','4','5']
    st.markdown(f'<h1 style="color:blue;font-size:24px;">{"The Levenshtein distance is used to create the edges between the nodesof the graph"}</h1>', unsafe_allow_html=True)
    levenshtein_distance = int(st.selectbox("Select Levenshtein distance (default = 1):", distances))

    df = st.session_state['input_dataframe'].drop(['label', 'label_transformed'], axis=1).T
    dfs = []
    for c in df:
      df_aux = pd.DataFrame(df[c])
      df_aux = df_aux[(df_aux.T > 0).any()]
      dfs.append(df_aux)

    st.markdown(f'<h1 style="color:blue;font-size:24px;">{"Building networks"}</h1>', unsafe_allow_html=True)
    @st.cache(suppress_st_warning=True)
    def build_networks(levenshtein_distance):
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
                    if distance(s1,s2) <= levenshtein_distance:
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

        st.session_state['graphs'] = pd.DataFrame() 
        st.session_state['graphs'] = graphs
        st.session_state['dic_nodes'] = pd.DataFrame() 
        st.session_state['dic_nodes'] = dic_nodes
        st.session_state['dic_edges'] = pd.DataFrame() 
        st.session_state['dic_edges'] = dic_edges
        st.session_state['dic_graphs'] = pd.DataFrame() 
        st.session_state['dic_graphs'] = dic_graphs

        
        st.markdown(f'<h1 style="color:blue;font-size:24px;">{"Calculating network features"}</h1>', unsafe_allow_html=True)
        my_bar = st.progress(0)  
        clustering_coeficient = []
        arrows = []
        density_ = []
        eccentricity = []
        network_size = []
        transitivity_ = []
        voterank = [] 
        connected_comp = []
        wiener = []
        triad = []
        tree = []
        forest = []

        percent_complete = 0
        #start_time = datetime.now()
        for g, percent_complete in zip(st.session_state['graphs'], range(0,100, int(100/len(st.session_state['graphs'])))):
            my_bar.progress(percent_complete + 1)          
            arrows.append(len(g.edges))
            density_.append(nx.density(g))
            if len(g) == 0:
                clustering_coeficient.append(0) 
                tree.append(0)
                forest.append(0)
     
            else:
                clustering_coeficient.append(nx.average_clustering(g))
                tree.append(int(nx.is_tree(g)))
                forest.append(int(nx.is_forest(g)))

            network_size.append(len(g))
            transitivity_.append(nx.transitivity(g))
            voterank.append(len(nx.voterank(g)))
            connected_comp.append(nx.number_connected_components(g))
            triad.append(int(nx.is_triad(g)))



            
        my_bar.progress(100)
        #time_elapsed = datetime.now() - start_time 
        #st.write('Graph features (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")

        return samples_names, arrows, density_, clustering_coeficient, network_size, transitivity_, voterank, connected_comp, triad, tree, forest
     
    samples_names, arrows, density_, clustering_coeficient, network_size, transitivity_, voterank, connected_comp, triad, tree, forest = build_networks(levenshtein_distance)   


    st.session_state['networks'] = pd.DataFrame() 
    st.session_state['networks']['label'] = st.session_state['input_dataframe']['label_transformed']
    st.session_state['networks']['sample'] = st.session_state['input_dataframe'].index
    st.session_state['networks']['arrows'] = arrows
    st.session_state['networks']['density'] = density_
    st.session_state['networks']['clustering_coeficient'] = clustering_coeficient  
    st.session_state['networks']['network_size'] = network_size  
    st.session_state['networks']['transitivity'] = transitivity_ 
    st.session_state['networks']['voterank'] = voterank 
    st.session_state['networks']['connected_comp'] = connected_comp 
    st.session_state['networks']['triad'] = triad
    st.session_state['networks']['tree'] = tree
    st.session_state['networks']['forest'] = forest

    st.markdown(f'<h1 style="color:black;font-size:24px;">{"Dataframe with network features"}</h1>', unsafe_allow_html=True)
    st.dataframe(st.session_state['networks'])
    st.write('Uploaded dataframe has ', len(st.session_state['networks'].columns), 'columns (features) and ', len(st.session_state['networks']), ' rows (samples)')
    st.download_button("Press the button to download dataframe with network features", st.session_state['networks'].to_csv().encode('utf-8'), "file.csv", "text/csv", key='download-csv')
    st.markdown(f'<h1 style="color:green;font-size:20px;">{"Features created!"}</h1>', unsafe_allow_html=True)
    time_elapsed = datetime.now() - start_time 
    st.write('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")

    # Graphs visualization
    if st.checkbox('Check the box to visualize interactive graphs'):
        st.markdown(f'<h1 style="color:blue;font-size:24px;">{"Interactive graph from one of the samples. Choose the sample you want to see"}</h1>', unsafe_allow_html=True)
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
                    ) 

        return_value = agraph(nodes=st.session_state['dic_nodes'][chosen_feature], edges=st.session_state['dic_edges'][chosen_feature], config=config)
    
        st.download_button("Press to Download Network", st.session_state['dic_graphs'][chosen_feature].to_csv().encode('utf-8'), "network.txt", "text/csv", key='download-text')         




def motif_features(motif):
    """
        This function creates new features based on the frequency of motifs
    """
    st.markdown(f'<h1 style="color:red;font-size:30px;">{"Motif Features"}</h1>', unsafe_allow_html=True)
    start_time = datetime.now()
    sizes = ['1', '2', '3', '4']
    st.markdown(f'<h1 style="color:blue;font-size:24px;">{"Choose the window size for the motifs calculation. The window size refers to the number of contiguous amino acids and the number of amino acids between 2 target amino acids."}</h1>', unsafe_allow_html=True)
    w_s = int(st.selectbox("Select window size (default = 1):", sizes))
    
    st.markdown(f'<h1 style="color:blue;font-size:24px;">{"Calculating motif features"}</h1>', unsafe_allow_html=True)
    @st.cache(suppress_st_warning=True)
    def motif_df(w_s): 
        my_bar = st.progress(0)  
        name = []
        df = st.session_state['input_dataframe'].drop(['label', 'label_transformed'], axis=1).T
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

        my_bar.progress(100)

        return pd.concat(dfs2), name

    motifs_, name = motif_df(w_s)

    st.session_state['motif'] = pd.DataFrame() 
    st.session_state['motif'] = motifs_
    st.session_state['motif']  = st.session_state['motif'].fillna(0)
    st.session_state['motif']['label'] = st.session_state['input_dataframe']['label_transformed']
    st.session_state['motif']['sample'] = st.session_state['input_dataframe'].index

    st.markdown(f'<h1 style="color:blue;font-size:24px;">{"Dataframe with motif features"}</h1>', unsafe_allow_html=True)
    st.dataframe(st.session_state['motif'])
    st.write('Uploaded dataframe has ', len(st.session_state['motif'].columns), 'columns (features) and ', len(st.session_state['motif']), ' rows (samples)')
    st.download_button("Press the button to download dataframe with motif features", st.session_state['motif'].to_csv().encode('utf-8'), "file.csv", "text/csv", key='download-csv')
    st.markdown(f'<h1 style="color:green;font-size:20px;">{"Features created!"}</h1>', unsafe_allow_html=True)
    time_elapsed = datetime.now() - start_time 
    st.write('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")



def dimensional_reduction_features():
    """
        This function creates new features based on dimensional reduction machine learning algorithms
    """
    start_time = datetime.now()
    st.markdown(f'<h1 style="color:red;font-size:30px;">{"Dimensional reduction features"}</h1>', unsafe_allow_html=True)

    st.markdown(f'<h1 style="color:blue;font-size:24px;">{"Calculating dimensional reduction features"}</h1>', unsafe_allow_html=True)
    st.write('This method considers only the 3 most important features of each dimensional reduction method: PCA, TSNE, UMAP, ICA, SVD, ISOMAP')
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def drf():
        #label = [int(x) for x in st.session_state['input_dataframe']['label']]
        data = st.session_state['input_dataframe'].drop(['label', 'label_transformed'],axis=1)

        #PCA
        pca = PCA(n_components=3).fit_transform(data)
        pca_df = pd.DataFrame(data = pca, columns = ['PC1', 'PC2', 'PC3'])
        pca_df.index = list(st.session_state['input_dataframe'].index)


        # TSNE
        tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=250).fit_transform(data)
        tsne_df = pd.DataFrame(data = tsne, columns = ['tsne1', 'tsne2', 'tsne3'])
        tsne_df.index = list(st.session_state['input_dataframe'].index)
        final_df = pca_df.join(tsne_df)

        # UMAP
        umap_ = umap.UMAP(n_components=3).fit_transform(data)
        umap_df = pd.DataFrame(data = umap_, columns = ['umap1', 'umap2', 'umap3'])
        umap_df.index = list(st.session_state['input_dataframe'].index)
        final_df = final_df.join(umap_df)

        # ICA
        ica = FastICA(n_components=3, random_state=0).fit_transform(data)
        ica_df = pd.DataFrame(data = ica, columns = ['IC1', 'IC2', 'IC3'])
        ica_df.index = list(st.session_state['input_dataframe'].index)
        final_df = final_df.join(ica_df)

        # SVD
        svd = TruncatedSVD(n_components=3, random_state=42).fit_transform(data)
        svd_df = pd.DataFrame(data = svd, columns = ['SVD1', 'SVD2', 'SVD3'])
        svd_df.index = list(st.session_state['input_dataframe'].index)
        final_df = final_df.join(svd_df)

        # ISOMAP
        isomap = Isomap(n_neighbors=5, n_components=3, n_jobs=-1).fit_transform(data)
        isomap_df = pd.DataFrame(data = isomap, columns = ['isomap1', 'isomap2', 'isomap3'])
        isomap_df.index = list(st.session_state['input_dataframe'].index)
        final_df = final_df.join(isomap_df)

        final_df['label'] = st.session_state['input_dataframe']['label_transformed']
        
        return final_df
        
        
    drf_ = drf()
        
    st.session_state['dimensional_reduction'] = drf_
    st.session_state['dimensional_reduction']['sample'] = st.session_state['input_dataframe'].index

    st.markdown(f'<h1 style="color:blue;font-size:24px;">{"Dataframe with dimensional reduction features"}</h1>', unsafe_allow_html=True)
    st.dataframe(st.session_state['dimensional_reduction'])
    st.write('Uploaded dataframe has ', len(st.session_state['dimensional_reduction'].columns), 'columns (features) and ', len(st.session_state['dimensional_reduction']), ' rows (samples)')
    st.download_button("Press the button to download dataframe with dimensional reduction features", st.session_state['dimensional_reduction'].to_csv().encode('utf-8'), "file.csv", "text/csv", key='download-csv')
    st.markdown(f'<h1 style="color:green;font-size:20px;">{"Features created!"}</h1>', unsafe_allow_html=True)
    time_elapsed = datetime.now() - start_time 
    st.write('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")



def side_bar():
    """
        This function shows the option widgets in the sidebar
    """
    st.sidebar.markdown(f'<h1 style="color:red;font-size:20px;">{"Select the features that you want to be created"}</h1>', unsafe_allow_html=True)

    clones = st.sidebar.checkbox('Clones')
    diversity = st.sidebar.checkbox('Diversity')
    network = st.sidebar.checkbox('Network')
    motif = st.sidebar.checkbox('Motif')
    dr = st.sidebar.checkbox('Dimensional Reduction')

    st.session_state.mosaic_count = 0

    if clones:
        clones_features(clones)
        st.session_state.mosaic_count += 1
    if diversity:
        diversity_features(diversity)
        st.session_state.mosaic_count += 1
    if network:
        network_features(network)
        st.session_state.mosaic_count += 1
    if motif:
        motif_features(motif)
        st.session_state.mosaic_count += 1
    if dr:
        dimensional_reduction_features()
        st.session_state.mosaic_count += 1

    if st.session_state.mosaic_count > 0:
        st.sidebar.markdown(f'<h1 style="color:red;font-size:20px;">{"Choose the normalization method you want to apply to the Mosaic dataframe"}</h1>', unsafe_allow_html=True)
        normalizations = ['No Normalization', 'Standard Scaler', 'Min-Max Scaler', 'Robust Scaler']
        norm_method = st.sidebar.selectbox("", normalizations)

        X, y = create_mosaic(norm_method)
        
        st.sidebar.markdown(f'<h1 style="color:red;font-size:20px;">{"Perform feature selection"}</h1>', unsafe_allow_html=True)
        if st.sidebar.checkbox('Check the box to start feature selection process'):
            feature_selection(X, y)


def feature_selection(X, y):
    """
        This function performs the feature selection method
    """
    if 'label' in X:
        X = X.drop('label', axis=1)

    start_time = datetime.now()
    st.markdown(f'<h1 style="color:red;font-size:30px;">{"Feature selection rank"}</h1>', unsafe_allow_html=True)
    # st.write('X')
    # st.dataframe(X)
    # Pearson
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
    cor_support, embeded_feature = cor_selector(X, y, len(X) - 1)
    scores = list(range(len(embeded_feature)-1,-1,-1))
    rank_dataframe1 = pd.DataFrame()
    rank_dataframe1['features'] = embeded_feature
    rank_dataframe1['Pearson scores'] = scores

    # Ridge
    embeded_selector = SelectFromModel(LogisticRegression(C=1, penalty='l2'), max_features=len(X) - 1)
    embeded_selector.fit(X, y)
    embeded_support = embeded_selector.get_support()
    embeded_feature = X.loc[:,embeded_support].columns.tolist()
    scores = list(range(len(embeded_feature)-1,-1,-1))
    rank_dataframe2 = pd.DataFrame()
    rank_dataframe2['features'] = embeded_feature
    rank_dataframe2['Ridge scores'] = scores
    final_rank_df = pd.merge(rank_dataframe1, rank_dataframe2, on = 'features')

    # XGBoost
    embeded_selector = SelectFromModel(xgb.XGBClassifier(), max_features=len(X) - 1)
    embeded_selector.fit(X, y)
    embeded_support = embeded_selector.get_support()
    embeded_feature = X.loc[:,embeded_support].columns.tolist()
    scores = list(range(len(embeded_feature)-1,-1,-1))
    rank_dataframe3 = pd.DataFrame()
    rank_dataframe3['features'] = embeded_feature
    rank_dataframe3['XGBoost scores'] = scores
    final_rank_df = pd.merge(final_rank_df, rank_dataframe3, on = 'features')

    # mRMR
    y = pd.Series(y)
    y.index = X.index
    selected_features = mrmr.mrmr_classif(X = X, y = y, K = len(X) - 1)
    embeded_feature = X.loc[:,selected_features].columns.tolist()
    scores = list(range(len(selected_features)-1,-1,-1))
    rank_dataframe4 = pd.DataFrame()
    rank_dataframe4['features'] = embeded_feature
    rank_dataframe4['mRMR scores'] = scores
    final_rank_df = pd.merge(final_rank_df, rank_dataframe4, on = 'features')

    # Meeged scores
    final_rank_df['sum'] = final_rank_df[list(final_rank_df.columns)].sum(axis=1)
    final_rank_df = final_rank_df.sort_values(by=['sum'], ascending=False)
    if final_rank_df.empty:
        st.markdown(f'<h1 style="color:black;font-size:20px;">{"The methods could not find any feature with predictive power"}</h1>', unsafe_allow_html=True)
    else:
        st.dataframe(final_rank_df)
        st.download_button("Press the button to download dataframe with the scores of the features", final_rank_df.to_csv().encode('utf-8'), "file.csv", "text/csv", key='download-csv')
        fig = px.bar(final_rank_df.sort_values(by=['sum'], ascending=False), x = 'features', y = 'sum', title="Rank of features", height=600)
        fig.update_layout(title_font_color="red", title_font_size=40)
        st.write(fig)

        time_elapsed = datetime.now() - start_time 
        st.write('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")

        st.markdown(f'<h1 style="color:red;font-size:20px;">{"Select the fearures that you want to validate with some classifiers. If you choose 3 features you can them see in a 3D scattern plot "}</h1>', unsafe_allow_html=True)
        options = st.multiselect('', list(st.session_state['mosaic'].drop(['sample', 'label'], axis=1).columns))
        if len(options) == 3:
            X_ = st.session_state['mosaic'].drop(['sample', 'label'], axis=1)
            X_ = X_[options]
            X_['label'] = st.session_state['input_dataframe']['label']
            st.dataframe(X_)
            fig = px.scatter_3d(X_, x=options[0], y=options[1], z=options[2], color='label')
            st.write(fig)

        X = X[options]
        
        ml_classifiers(X, y)


def ml_classifiers(X, y):
    """
        This function performs classification methods
    """
   
    st.write('Choose the RepeatedStratifiedKFold parameters')
    n_spl = st.slider("Choose the parameter n_splits. This number must be at most the size of the number of samples from the class with less samples.", 2, 10)
    n_rep = st.slider("Choose the parameter n_repeats", 10, 100, 10)
    cv = RepeatedStratifiedKFold(n_splits=int(n_spl), n_repeats=n_rep)
    
    st.markdown(f'<h1 style="color:red;font-size:20px;">{"Perform classification"}</h1>', unsafe_allow_html=True)
    if st.checkbox('Check the box to start classification process'):

        start_time = datetime.now()
        classifier_name = 'Gaussian Nayve Bayes'
        st.header('Gaussian Nayve Bayes')
        classifier = GaussianNB()

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
        time_elapsed = datetime.now() - start_time 
        st.write('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")

        start_time = datetime.now()
        st.header('Linear Discriminant Analysis')
        classifier = LinearDiscriminantAnalysis()
        classifier_name = 'Linear Discriminant Analysis'
      
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
        time_elapsed = datetime.now() - start_time 
        st.write('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")

        start_time = datetime.now()
        st.header('K Nearest Neighbor')
        classifier = KNeighborsClassifier()
        classifier_name = 'K Nearest Neighbor'
      
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
        time_elapsed = datetime.now() - start_time 
        st.write('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")

        start_time = datetime.now()
        st.header('Logistic Regression')
        classifier = LogisticRegression()
        classifier_name = 'Logistic Regression'
      
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
        time_elapsed = datetime.now() - start_time 
        st.write('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")

        start_time = datetime.now()
        st.header('Support Vector Machine')
        classifier = SVC(gamma='auto')
        classifier_name = 'Support Vector Machine'
      
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
        time_elapsed = datetime.now() - start_time 
        st.write('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")

        start_time = datetime.now()
        st.header('Decision Tree')
        classifier = DecisionTreeClassifier()
        classifier_name = 'Decision Tree'
      
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
        time_elapsed = datetime.now() - start_time 
        st.write('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")

        ml_model = st.radio("Choose a model to download", ['Gaussian Nayve Bayes', 'Linear Discriminant Analysis', 'K Nearest Neighbor', 'Logistic Regression', 'Support Vector Machine', 'Decision Tree'])
        st.write('Click the button to save the resulting classifier as in pickle file format')
        if st.button('Save Model'):
            file_name = "Classifier.pkl"
            with open(file_name, "wb") as open_file:
                pickle.dump(clf, open_file)
            st.write('File saved!')




def create_mosaic(scaler_name):
    """
        This function join all the features in one dataframe
    """
    start_time = datetime.now()

    mosaic_list = []
    if 'clones' in st.session_state:
        if not st.session_state['clones'].empty:
            mosaic_list.append(st.session_state['clones'])
    if 'diversity' in st.session_state:
        if not st.session_state['diversity'].empty:
            mosaic_list.append(st.session_state['diversity'])
    if 'networks' in st.session_state:
        if not st.session_state['networks'].empty:
            mosaic_list.append(st.session_state['networks'])
    if 'motif' in st.session_state:
        if not st.session_state['motif'].empty:
            mosaic_list.append(st.session_state['motif'])
    if 'dimensional_reduction' in st.session_state:
        if not st.session_state['dimensional_reduction'].empty:
            mosaic_list.append(st.session_state['dimensional_reduction'])

    if len(mosaic_list) > 0:
        st.session_state['mosaic'] = mosaic_list[0]
        for f in mosaic_list[1:]:
            f = f.drop('label', axis=1)
            st.session_state['mosaic'] = pd.merge(st.session_state['mosaic'], f, on="sample")

        st.markdown(f'<h1 style="color:red;font-size:30px;">{"Mosaic dataframe"}</h1>', unsafe_allow_html=True)
        st.write('The Mosaic dataframe can be the combination of 2 or more dataframes with the features that you chose to be created')
        aux = st.session_state['mosaic'].drop(['label', 'sample'],axis=1)
        
        if scaler_name == 'No Normalization':
            standarized_data = aux
        elif scaler_name == 'Standard Scaler':
            sc = StandardScaler()
            standarized_data = sc.fit_transform(aux)
        elif scaler_name == 'Min-Max Scaler':
            mms = MinMaxScaler()
            standarized_data = mms.fit_transform(aux)
        elif scaler_name == 'Robust Scaler':
            rs = RobustScaler()
            standarized_data = rs.fit_transform(aux)

        standarized_data = pd.DataFrame(standarized_data)
        standarized_data.index = aux.index
        standarized_data.columns = aux.columns
        X = standarized_data
        y = st.session_state['mosaic']['label'].to_list()
        st.session_state['scaled_mosaic'] = X
        st.session_state['scaled_mosaic']['label'] = y
        st.markdown(f'<h1 style="color:black;font-size:24px;">{"Dataframe normalized with " + scaler_name}</h1>', unsafe_allow_html=True)
        st.download_button("Press to Download DataFrame", X.to_csv().encode('utf-8'), "file.csv", "text/csv", key='download-csv')
        st.dataframe(st.session_state['scaled_mosaic'])
        st.write('Uploaded dataframe has ', len(st.session_state['scaled_mosaic'].columns), 'columns (features) and ', len(st.session_state['scaled_mosaic']), ' rows (samples)')
        st.download_button("Press the button to download Mosaic dataframe", st.session_state['scaled_mosaic'].to_csv().encode('utf-8'), "file.csv", "text/csv", key='download-csv')
        time_elapsed = datetime.now() - start_time 
        st.write('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + "\n")

        return X, y



def GENTLE_Main():
    """
        Main function of the application
    """

    st.set_page_config(layout="centered")

    page_title()

    data_loading()



# Run the GENTLE
if __name__ == '__main__':

    GENTLE_Main()