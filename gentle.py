#####################
#       IMPORTS
#####################

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from PIL import Image
import math
import plotly.express as px


from Levenshtein import *
import networkx as nx
import math
import scipy.stats as ss
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

import dg3 


def page_title():
    """
        This function displays the title of the application
    """
    st.markdown(f'<h1 style="color:green;font-size:36px;">{"GENTLE: GENerator of T cell repertoire features for machine LEarning algorithms analysis"}</h1>',
                unsafe_allow_html=True)
    #st.title('GENTLE: GENerator of T cell repertoire features for machine LEarning algorithms analysis',)
    image = Image.open('gentle_icon_v2.jpeg')
    st.image(image, width=None, caption=' ')



def data_loading():
    """
        This function loads the dataframe 
    """
    st.markdown(f'<h1 style="color:black;font-size:24px;">{"Upload dataset"}</h1>',
                unsafe_allow_html=True)
    #st.header("Upload dataset")
    delimiter = st.radio("Specify the delimiter used in the file", [",", ";", "tab", "space"])
    if delimiter == 'tab':
        delimiter = "   "
    elif delimiter == 'space':
        delimiter = " "

    file = st.file_uploader("The dataframe to be uploaded must have the rows as the samples (TCR repertoire) and the columns as the TCR sequences plus the target column.")

    if file:
        if file.name.endswith('zip'):
            df = pd.read_csv(file, compression='zip', header=0, sep=delimiter, quotechar='"')
            st.dataframe(df.T.head())
            st.session_state['dataframe'] = df
            st.session_state['features_initializer'] = 1
                  
        elif file.name.endswith('csv'):
            df = pd.read_csv(file, sep=delimiter)
            st.dataframe(df.head())
            st.session_state['dataframe'] = df
            st.session_state['features_initializer'] = 1

        else:
            st.write('Specify your file format: .csv or .zip')
            pass


def pca_calculation():
    """
        This function perform the principal component analysis
    """
    st.header('PCA of the diversity features')
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    pca = PCA(n_components=2)

    #Data Wrangling
    data = st.session_state['diversity_dataframe'].drop(['label', 'sample_id'],axis=1)

    # Fit
    principalComponents = pca.fit_transform(data)

    principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])

    fig, ax = plt.subplots()
    ax.scatter(principalDf['PC1'], principalDf['PC2'], c= st.session_state['diversity_dataframe']['label'])
    for i, txt in enumerate(st.session_state['diversity_dataframe']['sample_id']):
        ax.annotate(txt, (principalDf['PC1'][i], principalDf['PC2'][i]))
    yellow_patch = mpatches.Patch(color='yellow', label='Responders')
    purple_patch = mpatches.Patch(color='purple', label='Non-responders')
    ax.legend(handles=[yellow_patch, purple_patch])
    st.pyplot(fig)

    # if st.button('Show labels'):
    #     for i, txt in enumerate(df['sample_id']):
    #         ax.annotate(txt, (principalDf['PC1'][i], principalDf['PC2'][i]))
    #     st.pyplot(fig)



        
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


    # col1, col2, col3  = st.columns(3)

    # with col1:
    #     st.write("  Richness Index")
    #     st.latex(r''' H(X) = n ''')
    # with col2:
    #     st.write("  Shannon Index")
    #     st.latex(r''' H(X) = -\sum_{i=1}^{n} p(i)\log_2 p(i) ''')
    # with col3 :
    #     st.write("  Simpson Index")
    #     st.latex(r''' H(X) = -\sum_{i=1}^{n} p(i)^2 ''')

    # col4, col5, col6 = st.columns(3)
    # with col4:
    #     st.write("  Pielou Index")
    #     st.latex(r''' H(X) = -\sum_{i=1}^{n} p(i)^2 / \log_2(n) ''')
    # with col5:
    #     st.write("  Hillnumbers Index")
    #     st.latex(r''' H(X) = (\sum_{i=1}^{n} p(i)\alpha)^{1/1-\alpha} ''')
    # with col6 :
    #     st.write("  Gini Index")
    #     st.latex(r''' H(X) = \sum_{i=1}^{n}\sum_{j=1}^{n} |p(i)-p(j)| / 2n^2\bar{p} ''')     


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

    #features = st.multiselect('Diversity features', ['Shannon', 'Simpson', 'Richness'])
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

    for d in dfs:
        richness.append(len(d))   
        shannon.append(shannon_index(d))
        simpson.append(simpson_index(d))
        pielou.append(pielou_index(d))
        hillnumbers.append(hillnumbers_index(d, alpha))
        gini.append(gini_index(d))


    div_df = pd.DataFrame() 
    div_df['sample'] = name
    div_df['richness'] = richness
    div_df['shannon'] = shannon
    div_df['simpson'] = simpson
    div_df['pielou'] = pielou
    div_df['hillnumbers'] = hillnumbers
    div_df['gini'] = gini
    div_df['label'] = label

    
    st.markdown(f'<h1 style="color:blue;font-size:24px;">{"DATAFRAME WITH DIVERSITY FEATURES"}</h1>', unsafe_allow_html=True)
    st.dataframe(div_df.head())
    st.download_button("Press to Download DataFrame", div_df.to_csv().encode('utf-8'), "file.csv", "text/csv", key='download-csv')

    st.session_state['diversity_dataframe'] = div_df

    #pca_calculation()
    machine_learning_algorithms()

def network_features():
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
    graphs=[]
    dic_names = {}
    dic_graphs = {}
    samples_names = []
    dic_nodes = {}
    dic_edges = {}
    for d in dfs:
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

        dic_names[samples_names[-1]] = nodes_names
        dic_nodes[samples_names[-1]] = nodes
        dic_edges[samples_names[-1]] = edges
        dic_graphs[samples_names[-1]] = nx.to_numpy_array(G)
        graphs.append(G)

    st.session_state['graphs'] = graphs


    st.markdown(f'<h1 style="color:red;font-size:30px;">{"Interactive graph from one of the samples. Choose the sample you want to see"}</h1>', unsafe_allow_html=True)
    #st.header('Interactive graph from one of the samples. Choose the sample you want to see.')
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
    if st.button('Press to Download Network'):
        dg3.get_link_list(dic_graphs[chosen_feature], gene_names=dic_names[chosen_feature], file_name="network.txt")
        st.write('The file went straight to the folder of gentle')
    #st.download_button("Press to Download Network", _network, "network.txt", "text/csv", key='download-text')         

    clustering_coeficient = []
    arrows = []
    density = []
    eccentricity = []
    network_size = []
    trans = []
    for g in graphs:
      arrows.append(len(g.edges))
      density.append(nx.density(g))
      clustering_coeficient.append(nx.average_clustering(g))
      network_size.append(len(g))
      trans.append(nx.transitivity(g))
     

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
    st.download_button("Press to Download DataFrame", net_df.to_csv().encode('utf-8'), "file.csv", "text/csv", key='download-csv')

    st.session_state['network_dataframe'] = net_df

    machine_learning_algorithms()


def machine_learning_algorithms():
    
    if st.chosen_feature == 'Diversity':
        df = st.session_state['diversity_dataframe']
    elif st.chosen_feature == 'Network':
        df = st.session_state['network_dataframe']
    
    shuffled_df = df.sample(frac=1)
    X = shuffled_df.drop(['label', 'sample'],axis=1)
    y = shuffled_df['label'].to_list()

    st.markdown(f'<h1 style="color:blue;font-size:30px;">{"Choose the classifier you want to run"}</h1>', unsafe_allow_html=True)
    #st.header('Choose the classifier you want to run.')
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
    st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)

    ml_model = st.radio(" ", ['Gaussian Nayve Bayes', 'Linear Discriminant Analysis'])

    if ml_model == 'Gaussian Nayve Bayes':      
        st.header('Gaussian Nayve Bayes')
        clf = GaussianNB()

        st.write('Choose the RepeatedStratifiedKFold parameters')
        n_spl = 2
        n_rep = 2
        n_spl = st.slider("Choose the parameter n_splits", 2, 10)
        n_rep = st.slider("Choose the parameter n_repeats", 2, 1000)
        cv = RepeatedStratifiedKFold(n_splits=n_spl, n_repeats=n_rep)
        

        results_skfold_acc = cross_val_score(clf, X, y, cv=cv,scoring='accuracy')  
        results_skfold_pre = cross_val_score(clf, X, y, cv=cv,scoring='precision') 
        results_skfold_rec = cross_val_score(clf, X, y, cv=cv,scoring='recall')      
        results_skfold_f1 = cross_val_score(clf, X, y, cv=cv,scoring='f1')          
        results_skfold_auc = cross_val_score(clf, X, y, cv=cv,scoring='roc_auc')

        sp = pd.DataFrame({
                        'group': ['Accuracy','Precision','Recall','F1', 'ROC AUC'],
                        'GaussianNB': [results_skfold_acc.mean(), 
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
            fig = px.line_polar(sp, r='GaussianNB', theta='group', line_close=True)
            st.write(fig)

    if ml_model == 'Linear Discriminant Analysis': 

        st.header('Linear Discriminant Analysis')
        clf = LinearDiscriminantAnalysis()
     
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=100)

        results_skfold_acc = cross_val_score(clf, X, y, cv=cv,scoring='accuracy')  
        results_skfold_pre = cross_val_score(clf, X, y, cv=cv,scoring='precision') 
        results_skfold_rec = cross_val_score(clf, X, y, cv=cv,scoring='recall')      
        results_skfold_f1 = cross_val_score(clf, X, y, cv=cv,scoring='f1')          
        results_skfold_auc = cross_val_score(clf, X, y, cv=cv,scoring='roc_auc')

        sp = pd.DataFrame({
                        'group': ['Accuracy','Precision','Recall','F1', 'ROC AUC'],
                        'LDA': [results_skfold_acc.mean(), 
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
            fig = px.line_polar(sp, r='LDA', theta='group', line_close=True)
            st.write(fig)


def modtif_feature():
    pass

def GENTLE_Main():
    """
        Main function of the application
    """

    #st.set_page_config(layout="wide")
    st.set_page_config(layout="centered")

    st.session_state['features_initializer'] = 0
    page_title()

    data_loading()

    
    if st.session_state['features_initializer']:
        st.markdown(f'<h1 style="color:blue;font-size:30px;">{"Choose the features you want to be created"}</h1>', unsafe_allow_html=True)
        #st.header("Choose the features you want to be created")
        st.chosen_feature = st.radio(" ", ["Diversity", "Network"])
        if st.chosen_feature == 'Diversity':
            diversity_features()
        elif st.chosen_feature == 'Network':
            network_features()

    



 

# Run the GENTLE
if __name__ == '__main__':

    GENTLE_Main()
