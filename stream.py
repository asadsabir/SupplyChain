import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
from wrangling import G,y_train,y_test


st.set_page_config(layout='wide')
st.title('Time Series forecasting with GNNs and LSTM')

gnn_intro,lstm_intro = st.columns(2)

with gnn_intro:
    st.subheader("Graph Neural Networks")
    st.write("Graph Neural Networks excel in discerning the hidden interactions and dependencies inherent in complex systems. By leveraging the power of graph structures, they illuminate the pathways of influence, uncovering insights that remain hidden to conventional methods.")

with lstm_intro:
    st.subheader("Long Short Term Memory Networks")
    st.write("Time is not a static dimension but a dynamic force shaping our reality. Enter the Long Short-Term Memory (LSTM) networks, the temporal guardians of our data. LSTM networks possess the unique ability to capture and retain sequential dependencies, preserving the essence of time within each prediction.")

st.session_state.neighbors = [[],[]]
st.session_state.box0 = 1
st.sidebar.selectbox('Pick Sensor to analyze',range(36),0,key='input_node')
if st.session_state.input_node == st.session_state.box0:
    st.session_state.box0 = 36 - st.session_state.input_node
st.subheader("Fusing them together!")
st.write("In our quest for precision forecasting, we combine these formidable forces — GNNs and LSTMs — in a seamless integration of graph-based insights and temporal understanding. The result is a model that not only predicts the future but unveils the intricate dance of causality and correlation. This type of model has applications in finance, healthcare, energy and beyond!")

DATA_URL = "https://archive.ics.uci.edu/dataset/734/traffic+flow+forecasting-1"
st.subheader("Demystifying the black box")
st.write(f"To understand how these models can work together, lets walk through an implementation I built using Pytorch on a [traffic forecasting dataset]({DATA_URL}).")
st.write("The goal for this dataset is to forecast the spatio-temporal traffic volume based on the historical traffic volume and other features in neighboring locations. Specifically, the traffic volume is measured every 15 minutes at 36 sensor locations along two major highways in Northern Virginia/Washington D.C. capital region.")
st.write("The 49 features include:")
st.write("1) the historical sequence of traffic volume sensed during the 10 most recent sample points (10 features)")
st.write("2) week day (7 features)")
st.write("3) hour of day (24 features)")
st.write("4) road direction (4 features)")
st.write("5) number of lanes (1 feature)")
st.write("6) degree of the sensor or number of edges (1 feature).")
st.write("7) 2 unknown features (2 features)")

st.divider()
edge_weights = np.load('average_edge_weights.npy')
@st.cache_data
def make_edges():
    edges = []
    for i,(u,v) in enumerate(G.edges):
        edge_width = (edge_weights[i]- 0.78)*15
        edges.append( Edge(source=f"{u}",
                        target=f"{v}" ,
                        #label=f"{edge_weight:1f}",
                        **{
                            'width':edge_width
                        }
                        )
                    )
    return edges

config = Config(width=950,
                height=750,
                directed=False, 
                physics=False, 
                hierarchical=False,
                # **kwargs
                )

st.subheader('The road network')

def make_nodes_graph():
    nodes = []
    selected_node = st.session_state.input_node
    
    for n in G.nodes:
        k_neighborhood = min(nx.shortest_path_length(G,n,selected_node),3)
        colors = ['red','orange','#ebcc34','blue']
        if 0 < k_neighborhood < 3:
            st.session_state.neighbors[k_neighborhood-1].append(n)
        nodes.append( Node(id=f"{n}", 
                        label= f"sensor {n}", 
                        size=25,
                        color=colors[k_neighborhood]
                        ) 
                    ) # includes **kwargs
    return nodes

graph2col, choosecol = st.columns(2)

with graph2col:
    node_graph2 = agraph(nodes=make_nodes_graph(),
                                edges=make_edges(), 
                                config=config)
    if node_graph2:
        if int(node_graph2) != st.session_state.input_node:
            st.session_state.box0 = int(node_graph2)

with choosecol:
    st.write("- The nodes represent traffic sensors while the edges represent the roads that connect them. You can drag the nodes around or zoom in and out to get a better look.")
    st.write('- The edge width/thickness shows the weight of those edges as estimated by the model')
    st.write('- the red node is the selected sensor to predict, you can change this sensor in the sidebar')
    st.write('- the orange nodes are 1 edge away from the red node and the gold/yellow nodes are 2 edges away')
    st.write('- you can click on a node to compare its time series to the red sensor')
    
    st.write('Traffic Volume')
    compare = pd.DataFrame({
        'Time in 15 minute intervals':range(1,1262),
        f'sensor {st.session_state.input_node}':y_train[st.session_state.input_node,:],
        f'sensor {st.session_state.box0}':y_train[st.session_state.box0,:],
    })
    st.line_chart(compare,x='Time in 15 minute intervals',color=['#0ccbe8','#3b4ab8'])

st.subheader('The LSTM Gates')
lstm_image,lstm_explain = st.columns(2)
with lstm_image:
    st.image('https://d2l.ai/_images/lstm-3.svg',use_column_width=True)
with lstm_explain:
    st.write("- Input X represents a sensor and its current features.")
    st.write("- The Hidden state(H) represents the LSTM cell output from the last timestep. This is the 'short term memory' part.")
    st.write("- The Cell state(C) represents the 'long term memory' part and accumulates a bit of information from all the previous timesteps.")
    st.write("- The 'forget' and 'input' gates decide what parts and how much of **X** and **H** should be commited to longterm memory in **C** and how much of **C** should be 'forgotten'.")
    st.write("- The 'output' gate combines **X**, **H** and **C** to produce a prediction which will become the new **H** for the next timestep!")
    st.write("- Information from neighboring nodes up to 2 edges away is also convoluted into this diagram as well using their features")
    st.write("- The parameters that determine how much of the neighboring nodes information to use is independent of the nodes themselves and solely depends on their distance to the prediction node. This helps scale the model to large graphs")
    st.write("- This model makes 32 hidden predictions with the Graph Convolutional LSTM layer.")



st.subheader('Cell State: the backbone of LSTM!')
st.write('The Cell state matrix allows the model to learn and differentiate between both long-term and short term patterns.')
st.write('By analyzing what goes into the Cell State from the Input gates and what leaves the cell state from the Forget gates, we can investigate the relationship between different nodes and hidden features.')

st.selectbox('hidden feature',range(32),key='hidden_feature')

LSTM_gates_te = np.load('LSTM_gates_te.npy')
LSTM_gates_tr = np.load('LSTM_gates_tr.npy')

with st.expander('Forget Gate',expanded=True):

    train_fcol, test_fcol = st.columns(2)

    plot_train = LSTM_gates_tr[:,3,st.session_state.input_node,st.session_state.hidden_feature]*100
    plot_test = LSTM_gates_te[:,3,st.session_state.input_node,st.session_state.hidden_feature]*100
    
    with train_fcol:
        fig = plt.figure(figsize=(10, 4))
        sns.lineplot(plot_train)
        
        plt.ylabel("Percentage of Cell state 'forgotten'(%)")
        plt.xlabel('Time in 15 minute intervals')
        plt.title("Training set")
        
        st.pyplot(fig)

    with test_fcol:
        fig = plt.figure(figsize=(10, 4))

        sns.lineplot(plot_test)
        
        plt.ylabel("Percentage of Cell state 'forgotten'(%)")
        plt.xlabel('Time in 15 minute intervals')
        plt.title("Testing set")
        
        st.pyplot(fig)


with st.expander('Input Gate',expanded=True):

    train_icol, test_icol = st.columns(2)

    plot_train = LSTM_gates_tr[:,4,st.session_state.input_node,st.session_state.hidden_feature]*100
    plot_test = LSTM_gates_te[:,4,st.session_state.input_node,st.session_state.hidden_feature]*100

    with train_icol:
        fig = plt.figure(figsize=(10, 4))
        sns.lineplot(plot_train)
        
        plt.ylabel("Input 'remembered'(%)")
        plt.title("Training set")
        st.pyplot(fig)

    with test_icol:
        fig = plt.figure(figsize=(10, 4))
        sns.lineplot(plot_test)
        
        plt.ylabel("Input 'remembered'(%)")
        plt.title("Testing set")
        st.pyplot(fig)

st.subheader('SAGE: SAmple and AGgregatE!')
st.image('https://snap.stanford.edu/graphsage/sample_and_agg.png')
st.write("- The 32 hidden predictions from the GCLSTM layer pass through ReLU and are then put through this layer to get one traffic volume prediction")
st.write("- The SAGE convolutional layer samples neighbor features, transforms them and then aggregates them either by taking the max,sum or mean")
st.write("- In this model I used a **K** of one and used **mean** as the aggregation method")
st.write("- the final prediction of each sensor is a linear transformation of the neighbor features added to a seperate linear transformation of the root node with a bias of **0.0806**")
st.write("- **Note**: The plots below only show the effect of neighboring sensors **in this layer**. Neighboring nodes have a seperate and possibly different effect in the LSTM layer")

neighbor_effects_te = None
neighbor_effects_tr = None
with open('neighbor_effects_te.json') as f:
    neighbor_effects_te = json.load(f)

with open('neighbor_effects_tr.json') as f:
    neighbor_effects_tr = json.load(f)

sageviztr,sagevizte = st.columns(2)

with sageviztr:
    
    effects = []
    for d in neighbor_effects_tr:
        effects.append(d[str(st.session_state.input_node)])
    effects = pd.DataFrame(effects)
    
    fig = plt.figure(figsize=(10, 4))
    sns.histplot(effects.drop(str(st.session_state.input_node),axis=1))
    plt.xlabel(f'Effect on prediction of Sensor {st.session_state.input_node}')
    plt.ylabel('Frequency')
    st.pyplot(fig)

    fig2 = plt.figure(figsize=(10, 4))
    sns.histplot(effects[str(st.session_state.input_node)])
    plt.xlabel('Effect of sensors own features')
    plt.ylabel('Frequency')
    st.pyplot(fig2)

with sagevizte:
    
    effects = []
    for d in neighbor_effects_te:
        effects.append(d[str(st.session_state.input_node)])
    effects = pd.DataFrame(effects)
    
    fig = plt.figure(figsize=(10, 4))
    sns.histplot(effects.drop(str(st.session_state.input_node),axis=1),multiple='fill')
    
    plt.xlabel(f'Effect on prediction of Sensor {st.session_state.input_node}')
    plt.ylabel('Frequency')
    st.pyplot(fig)

    fig2 = plt.figure(figsize=(10, 4))
    sns.histplot(effects[str(st.session_state.input_node)])
    plt.xlabel('Effect of sensors own features')
    plt.ylabel('Frequency')
    st.pyplot(fig2)

st.subheader('the Forecast!')
forecast_tr,forecast_te = st.columns(2)
with forecast_tr:
    st.markdown('<h4> Training Set </h4>',unsafe_allow_html=True)
    train_preds = np.load('train_preds.npy')
    
    cast = pd.DataFrame({
        'Time in 15 minute intervals':range(1,1262),
        'actual':y_train[st.session_state.input_node,:],
        'prediction':train_preds[:,st.session_state.input_node]
    })
    st.line_chart(cast,color=['#db1212','#12dbca'],x='Time in 15 minute intervals')
    st.write(f'Train Root Mean Squared Error for Sensor {st.session_state.input_node} is **{np.mean(np.square(np.subtract(cast['prediction'],cast['actual'])))**0.5:3f}**')

with forecast_te:
    st.markdown('<h4> Testing Set </h4>',unsafe_allow_html=True)
    test_preds = np.load('test_preds.npy')

    cast = pd.DataFrame({
        'Time in 15 minute intervals':range(1,841),
        'actual':y_test[st.session_state.input_node,:],
        'prediction':test_preds[:,st.session_state.input_node]
    })
    st.line_chart(cast,color=['#db1212','#12dbca'],x='Time in 15 minute intervals')
    st.write(f'Test Root Mean Squared Error for Sensor {st.session_state.input_node} is **{np.mean(np.square(np.subtract(cast['prediction'],cast['actual'])))**0.5:3f}**')
st.write('The average Root Mean Squared Error on the training and testing set for all Sensors is **0.0352** and **0.0482** respectively')
st.subheader('Sources')
paper_urls = ['https://arxiv.org/abs/1612.07659','https://arxiv.org/pdf/1706.02216.pdf']

st.write("- [Structured Sequence Modeling with Graph Convolutional Recurrent Networks.]({paper_urls[0]})")
st.write("- [Inductive Representation Learning on Large Graphs.]({paper_urls[1]})")