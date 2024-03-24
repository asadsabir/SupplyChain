#Save model
#Read papers and source code
#Make the damn ting 


import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from wrangling import G,y_train,X_train,X_test,y_test
import pandas as pd
import networkx as nx
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout='wide')
st.title('Time Series forecasting with GNNs and LSTM')

gnn_intro,lstm_intro = st.columns(2)

with gnn_intro:
    st.subheader("Graph Neural Networks")
    st.write("Graph Neural Networks excel in discerning the hidden interactions and dependencies inherent in complex systems. By leveraging the power of graph structures, they illuminate the pathways of influence, uncovering insights that remain hidden to conventional methods.")

with lstm_intro:
    st.subheader("Long Short Term Memory Networks")
    st.write("Yet, time is not a static dimension but a dynamic force shaping our reality. Enter the Long Short-Term Memory (LSTM) networks, the temporal guardians of our data. LSTM networks possess the unique ability to capture and retain sequential dependencies, preserving the essence of time within each prediction.")

st.session_state.neighbors = [[],[]]
st.session_state.box0 = 1
st.session_state.box1 = 2
st.session_state.upgraph = True

st.subheader("Fusing them together!")
st.write("In our quest for precision forecasting, we combine these formidable forces — GNNs and LSTMs — in a seamless integration of graph-based insights and temporal understanding. The result is a model that not only predicts the future but unveils the intricate dance of causality and correlation. This type of model has applications in finance, healthcare, energy and beyond.")

DATA_URL = "https://archive.ics.uci.edu/dataset/734/traffic+flow+forecasting-1"
st.subheader("Demystifying the black box")
st.write(f"To understand how these models can work together, lets walk through an implementation I built using Pytorch on a [traffic forecasting dataset]({DATA_URL}).")
st.write("The goal for this dataset is to forecast the spatio-temporal traffic volume based on the historical traffic volume and other features in neighboring locations. Specifically, the traffic volume is measured every 15 minutes at 36 sensor locations along two major highways in Northern Virginia/Washington D.C. capital region. Below you can visualize the dataset as a graph structure. The nodes represent traffic sensors while the edges represent the roads that connect them. You can move the nodes around and zoom in and out to get a better look. Click on a node to see its Traffic volume on the right")
st.divider()
model_dict = torch.load('final_supply_chain/models/modelat472.pth')

@st.cache_data
def make_edges():
    edges = []
    for u,v in G.edges:
        edges.append( Edge(source=f"{u}",
                        target=f"{v}" 
                        # **kwargs
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

sensors,volume_time = st.columns(2)


with sensors:
    st.select_slider('Pick sensor for',['Upper Graph','Lower Graph'],key='which_graph')
    node_graph1 = agraph(nodes=[ Node(id=f"{n}",label=f"sensor {n}",size=25,color='blue') for n in G.nodes],
                            edges=make_edges(), 
                            config=config)

    if node_graph1:
        if st.session_state.which_graph == 'Upper Graph':
            st.session_state.box1 = int(node_graph1)
        else:
            st.session_state.box0 = int(node_graph1)

with volume_time:
    st.write(f'Traffic Volume for sensor {st.session_state.box1} over time(15min intervals)')
    st.line_chart(pd.DataFrame(y_train[st.session_state.box1,:],columns=[f'Traffic Volume for sensor {st.session_state.box1}']))
   

    st.write(f'Traffic Volume for sensor {st.session_state.box0} over time(15min intervals)')
    st.line_chart(pd.DataFrame(y_train[st.session_state.box0,:],columns=[f'Traffic Volume for sensor {st.session_state.box0}']))
    



st.divider()
st.subheader('Our starting point')

def make_nodes_graph2():
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


with choosecol:
    st.write('To make a prediction for any sensor, the model considers all nodes that are at most 2 edges away')
    st.write('On the left, the red node is the selected sensor, the orange nodes are 1 edge away and the gold/yellow nodes are 2 edges away')
    st.selectbox('Pick a Sensor',range(36),0,key='input_node')

with graph2col:
    node_graph2 = agraph(nodes=make_nodes_graph2(),
                                edges=make_edges(), 
                                config=config)
    
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
    st.write("- Information from neighboring nodes is convoluted into this diagram as well using their respective **X**s, **H**s and **C**s")
    st.write("- The parameters that determine how much of the neighboring nodes information to use is independent of the nodes themselves and solely depends on their distance to the prediction node. This helps scale the model to large graphs")
    st.write("- This model makes 36 hidden predictions with the Graph Convolutional LSTM layer. The 36 outputs are then fed into a linear layer that makes one final prediction for each sensor")

st.subheader('The Longterm Memory')

st.selectbox('distance from the sensor',range(3),key='forget_neighborhood',help='0 refers the sensor itself, 1 refers to sensors 1 edge away and 2 refers to sensors 2 edges away')
st.selectbox('hidden feature',range(36),key='hidden_feature')


xf_nodes = st.session_state.input_node
dims = ([1],[1])
one_slice = True

if st.session_state.forget_neighborhood > 0:
        xf_nodes = st.session_state.neighbors[st.session_state.forget_neighborhood - 1]
        dims = ([2],[1])
        one_slice = False
        legend_labels = [f'Sensor {n}'for n in st.session_state.neighbors[st.session_state.forget_neighborhood-1]]


with st.expander('Forget Gate'):

    train_fcol, test_fcol = st.columns(2)

    Wxf = model_dict[f'recurrent.conv_x_f.lins.{st.session_state.forget_neighborhood}.weight']
    Whf = model_dict[f'recurrent.conv_h_f.lins.{st.session_state.forget_neighborhood}.weight']
    train_xf = (1 - torch.sigmoid(torch.tensordot(torch.tensor(X_train[:,xf_nodes,:],dtype=torch.float),Wxf,dims) + model_dict['recurrent.conv_x_f.bias'] + model_dict['recurrent.conv_h_f.bias'])) * 100
    test_xf = (1 - torch.sigmoid(torch.tensordot(torch.tensor(X_test[:,xf_nodes,:],dtype=torch.float),Wxf,dims) + model_dict['recurrent.conv_x_f.bias'] + model_dict['recurrent.conv_h_f.bias'] + model_dict['recurrent.b_f'])) * 100

    if one_slice:
        plot_train = train_xf[:,st.session_state.hidden_feature].detach().numpy()
        plot_test = test_xf[:,st.session_state.hidden_feature].detach().numpy()
    else:
        plot_train = train_xf[:,:,st.session_state.hidden_feature].detach().numpy()
        plot_test = test_xf[:,:,st.session_state.hidden_feature].detach().numpy()

    with train_fcol:
        fig = plt.figure(figsize=(10, 4))
        sns.histplot(plot_train)
        plt.xlim((0,100))
        plt.xlabel("Percentage of Cell state 'forgotten'(%)")
        plt.title("Training set")
        if not one_slice:
            plt.legend(labels=legend_labels)
        st.pyplot(fig)

    with test_fcol:
        fig = plt.figure(figsize=(10, 4))

        sns.histplot(plot_test)
        plt.xlim((0,100))
        plt.xlabel("Percentage of Cell state 'forgotten'(%)")
        plt.title("Testing set")
        if not one_slice:
            plt.legend(labels=legend_labels)
        st.pyplot(fig)


with st.expander('Input Gate'):

    train_icol, test_icol = st.columns(2)

    Wxi = model_dict[f'recurrent.conv_x_i.lins.{st.session_state.forget_neighborhood}.weight']
    Whi = model_dict[f'recurrent.conv_h_i.lins.{st.session_state.forget_neighborhood}.weight']
    train_xi = (1 - torch.sigmoid(torch.tensordot(torch.tensor(X_train[:,xf_nodes,:],dtype=torch.float),Wxi,dims) + model_dict['recurrent.conv_x_i.bias'] + model_dict['recurrent.conv_x_i.bias'] + model_dict['recurrent.b_i'])) * 100
    test_xi = (1 - torch.sigmoid(torch.tensordot(torch.tensor(X_test[:,xf_nodes,:],dtype=torch.float),Wxi,dims) + model_dict['recurrent.conv_x_i.bias'] + model_dict['recurrent.conv_x_i.bias'] + model_dict['recurrent.b_i'])) * 100

    if one_slice:
        plot_train = train_xi[:,st.session_state.hidden_feature].detach().numpy()
        plot_test = test_xi[:,st.session_state.hidden_feature].detach().numpy()
    else:
        plot_train = train_xi[:,:,st.session_state.hidden_feature].detach().numpy()
        plot_test = test_xi[:,:,st.session_state.hidden_feature].detach().numpy()
    with train_icol:
        fig = plt.figure(figsize=(10, 4))
        sns.histplot(plot_train)
        plt.xlim((0,100))
        plt.xlabel("Input 'remembered'(%)")
        plt.title("Training set")
        if not one_slice:
            plt.legend(labels=legend_labels)
        st.pyplot(fig)

    with test_icol:
        fig = plt.figure(figsize=(10, 4))
        sns.histplot(plot_test)
        plt.xlim((0,100))
        plt.xlabel("Input 'remembered'(%)")
        plt.title("Testing set")
        if not one_slice:
            plt.legend(labels=legend_labels)
        st.pyplot(fig)

st.write("From this you can see how this type of model can learn both short term and long term sequential patterns by deciding which features to retain for a long time and which ones to forget")
st.write("Try to identify which of the hidden features are used for short term memory and which are kept for the longterm!")


st.subheader('Output Gate')

st.subheader('and finally, the Forecast!')
forecast_tr,forecast_te = st.columns(2)
with forecast_tr:
    train_preds = np.load('train_preds.npy')
    print(train_preds[:,st.session_state.input_node].shape)
    cast = pd.DataFrame({
        'actual':y_train[st.session_state.input_node,:],
        'prediction':train_preds[:,st.session_state.input_node]
    })
    st.line_chart(cast,color=['#db1212','#12dbca'])

with forecast_te:
    test_preds = np.load('test_preds.npy')
    print(test_preds[:,st.session_state.input_node].shape)
    cast = pd.DataFrame({
        'actual':y_test[st.session_state.input_node,:],
        'prediction':test_preds[:,st.session_state.input_node]
    })
    st.line_chart(cast,color=['#db1212','#12dbca'])
st.subheader('Sources')