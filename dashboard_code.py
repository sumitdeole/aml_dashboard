import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Configure page
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Improved CSS
st.markdown("""
<style>
    .main {padding: 1rem;}
    .metric-card {
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }
    .metric-label {
        font-size: 0.9rem !important;
        color: #666 !important;
    }
    .quick-access-btn {
        width: 100%;
        margin-bottom: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background: #f0f2f6;
        padding: 4px 8px;
        border-radius: 8px;
    }
    .empty-network {
        text-align: center;
        padding: 2rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Load data and identify example clients
@st.cache_data
def load_data():
    df = pd.read_csv('data/creditcard.csv')
    np.random.seed(42)
    df['ClientID'] = np.random.choice(range(10000, 20000), size=len(df))
    
    # Get top fraud client (the one with highest amount fraud transactions)
    fraud_client = df[df['Class'] == 1].nlargest(1, 'Amount')['ClientID'].iloc[0]
    
    # Get a legitimate client with multiple transactions
    legit_client = df[df['Class'] == 0].groupby('ClientID').filter(lambda x: len(x) > 5)['ClientID'].iloc[0]
    
    return df, fraud_client, legit_client

df, fraud_client, legit_client = load_data()

# Initialize session state
if 'selected_client' not in st.session_state:
    st.session_state.selected_client = fraud_client
    st.session_state.client_changed = True
else:
    st.session_state.client_changed = False

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    # Quick access buttons at the top
    st.subheader("Quick Access")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👤 Fraud Client Example", key="fraud_btn", 
                    help="View example fraud client", use_container_width=True, type="primary"):
            st.session_state.selected_client = fraud_client
            st.session_state.client_changed = True
            st.rerun()
    with col2:
        if st.button("👤 Legitimate Client Example", key="legit_btn", 
                    help="View example legitimate client", use_container_width=True):
            st.session_state.selected_client = legit_client
            st.session_state.client_changed = True
            st.rerun()
    
    st.markdown("---")
    st.subheader("Client Selection")
    
    # Create dropdown with client options
    client_list = [fraud_client, legit_client] + df[~df['ClientID'].isin([fraud_client, legit_client])]['ClientID'].sample(8).tolist()
    selected_client = st.selectbox(
        "Select Client", 
        client_list,
        index=client_list.index(st.session_state.selected_client) if st.session_state.selected_client in client_list else 0,
        key='client_select'
    )
    
    # Update session state when dropdown changes
    if selected_client != st.session_state.selected_client:
        st.session_state.selected_client = selected_client
        st.session_state.client_changed = True
        st.rerun()
    
    st.markdown("---")
    st.subheader("Model Settings")
    model_choice = st.selectbox("Detection Model", ["NN (SMOTE)", "DNN (SMOTE)"], key='model_select')
    apply_smote = st.checkbox("Apply SMOTE Balancing", True, key='smote_checkbox')

# Use the client from session state
client_data = df[df['ClientID'] == st.session_state.selected_client]
fraud_count = client_data['Class'].sum()

# Format timestamp functions
def format_date(seconds):
    return datetime.fromtimestamp(seconds).strftime('%Y-%m-%d')

def format_time(seconds):
    return datetime.fromtimestamp(seconds).strftime('%H:%M:%S')

# Main content
st.header("Fraud Detection Dashboard")

# KPI Cards (4 columns)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Transactions</div>
        <div class="metric-value">{len(client_data):,}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Fraud Cases</div>
        <div class="metric-value" style="color: {'red' if fraud_count > 0 else 'green'}">{fraud_count}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Amount</div>
        <div class="metric-value">${client_data['Amount'].mean():,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    last_date, last_time = format_date(client_data['Time'].max()), format_time(client_data['Time'].max())
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Last Activity</div>
        <div class="metric-value">{last_date}</div>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["💳 Transactions", "🕸️ Network", "🔍 Fraud Details"])

with tab1:
    st.subheader("Recent Transactions")
    
    # Show all transactions for the client, sorted by time
    display_data = client_data.sort_values('Time', ascending=False)
    
    st.dataframe(
        display_data.assign(
            Date=lambda x: x['Time'].apply(format_date),
            Time=lambda x: x['Time'].apply(format_time)
        )[['Date', 'Time', 'Amount', 'Class']].rename(columns={
            'Amount': 'Amount ($)',
            'Class': 'Fraud'
        }).style.applymap(
            lambda x: 'background-color: #ffcccc' if x == 1 else '', 
            subset=['Fraud']
        ),
        height=min(400, 35 * len(display_data)),
        use_container_width=True
    )

with tab2:
    st.subheader("Transaction Network Analysis")
    
    def visualize_fraud_network(client_id, df, num_fraud=15, num_legit=8, seed=42):
        """Professional fraud network visualization"""
        client_data = df[df['ClientID'] == client_id]
        fraud_count = client_data['Class'].sum()
        
        if fraud_count == 0:
            return None, False
        
        try:
            # Prepare data with transaction frequency
            fraud_samples = client_data[client_data['Class']==1].nlargest(num_fraud, 'Amount')
            legit_samples = client_data[client_data['Class']==0].sample(num_legit, random_state=seed)
            
            # Create graph with meaningful attributes
            G = nx.Graph()
            
            # Add client as hub
            G.add_node(f"Client {client_id}",
                      type='hub',
                      amount=client_data['Amount'].mean(),
                      size=40,
                      color='#e84118')
            
            # Add fraud nodes
            for idx, row in fraud_samples.iterrows():
                G.add_node(f"F_{idx}",
                          type='fraud',
                          amount=row['Amount'],
                          time=row['Time'],
                          size=np.log(row['Amount']+100)*30,
                          color='#ff4757')
                
                # Connect to hub
                G.add_edge(f"Client {client_id}", f"F_{idx}",
                          weight=0.9,
                          color='#ff6b6b')
            
            # Add legit nodes
            for idx, row in legit_samples.iterrows():
                G.add_node(f"L_{idx}",
                          type='legit',
                          amount=row['Amount'],
                          time=row['Time'],
                          size=np.log(row['Amount']+100)*20,
                          color='#2ed573')
                
                # Connect to hub
                G.add_edge(f"Client {client_id}", f"L_{idx}",
                          weight=0.4,
                          color='#ffa502')
            
            # Connect fraud nodes to each other
            fraud_nodes = [n for n in G.nodes if 'F_' in n]
            for i in range(len(fraud_nodes)-1):
                G.add_edge(fraud_nodes[i], fraud_nodes[i+1],
                          weight=0.9,
                          color='#ff6b6b')
            
            return G, True
        
        except Exception as e:
            st.error(f"Error generating network: {str(e)}")
            return None, False
    
    # Generate the network visualization
    network_graph, has_fraud = visualize_fraud_network(st.session_state.selected_client, df)
    
    if has_fraud:
        # Create Plotly figure
        pos = nx.spring_layout(network_graph, seed=42)
        
        edge_traces = []
        for edge in network_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=network_graph.edges[edge]['weight']*3, 
                         color=network_graph.edges[edge]['color']),
                hoverinfo='none',
                mode='lines'
            ))
        
        node_x, node_y, colors, sizes, texts = [], [], [], [], []
        for node in network_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            colors.append(network_graph.nodes[node]['color'])
            sizes.append(network_graph.nodes[node]['size'])
            texts.append(f"${network_graph.nodes[node]['amount']:,.0f}" if 'amount' in network_graph.nodes[node] else node)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(color=colors, size=sizes, 
                       line=dict(width=1, color='DarkSlateGrey')),
            text=texts,
            textposition="bottom center"
        )
        
        fig = go.Figure(data=edge_traces + [node_trace],
                     layout=go.Layout(
                        showlegend=False,
                        margin=dict(b=0,l=0,r=0,t=0),
                        height=500,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        st.markdown("""
        <div style="display: flex; justify-content: center; gap: 15px; margin-top: -10px; flex-wrap: wrap;">
            <div style="display: flex; align-items: center;">
                <div style="width:12px; height:12px; background:#e84118; border-radius:50%; margin-right:5px;"></div>
                <span>Client Hub</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width:12px; height:12px; background:#ff4757; border-radius:50%; margin-right:5px;"></div>
                <span>Fraud Transaction</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width:12px; height:12px; background:#2ed573; border-radius:50%; margin-right:5px;"></div>
                <span>Legitimate Transaction</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-network">
            <h3>No Fraud Network Detected</h3>
            <p>This client has no fraudulent transactions to visualize.</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.subheader("Fraud Analysis")
    
    if fraud_count > 0:
        fraud_data = client_data[client_data['Class'] == 1].sort_values('Amount', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Highest Fraud Amount", 
                     f"${fraud_data['Amount'].max():,.0f}")
        with col2:
            st.metric("Average Fraud Amount", 
                     f"${fraud_data['Amount'].mean():,.0f}")
        
        st.markdown("**Fraud Timeline**")
        timeline_data = fraud_data.assign(
            Timestamp=lambda x: x['Time'].apply(lambda t: datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M')))
        timeline = timeline_data.set_index('Timestamp')['Amount']
        st.line_chart(timeline, height=200)
        
        st.markdown("**Fraud Transactions**")
        st.dataframe(
            timeline_data[['Timestamp', 'Amount']].rename(columns={
                'Amount': 'Amount ($)'
            }).style.format({'Amount ($)': '${:,.2f}'}),
            height=min(200, 35 * len(timeline_data)),
            use_container_width=True
        )
    else:
        st.success("No fraudulent transactions detected for this client")