# Credit Card Fraud Detection Dashboard
This project creates a Streamlit-powered interactive dashboard for analyzing credit card fraud patterns through network visualization and transaction analysis. 
![Dashboard Demo](https://github.com/sumitdeole/aml_dashboard/blob/main/dashboard_demo.gif)



## Dataset Information

This dashboard uses the **Credit Card Fraud Detection Dataset** from Kaggle:
- Contains transactions made by European cardholders in September 2013
- 284,807 transactions with 492 frauds (highly imbalanced dataset)
- Features V1-V28 are PCA-transformed for confidentiality
- Only features not transformed are 'Time' and 'Amount'
- 'Class' is the target variable (1 = fraud, 0 = legitimate)

### Obtaining the Data

1. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Place the `creditcard.csv` file in a `data/` directory in your project folder



## Installation & Setup

1. **Clone the repository**:

`git clone https://github.com/yourusername/credit-card-fraud-dashboard.git`

`cd credit-card-fraud-dashboard`

2. Install dependencies:

`pip install streamlit pandas numpy plotly networkx matplotlib scikit-learn imbalanced-learn`

3. Run the dashboard:

`streamlit run app.py`


## Dashboard Features
- Client Selection: Choose between example fraud/legitimate clients or random clients
- Network Visualization: See [Network Graph](https://github.com/sumitdeole/aml_dashboard/blob/main/Fraud_network_analysis.pdf)
- Detailed Transaction View: Explore all transactions with fraud highlighting
- Fraud Analytics: View fraud-specific metrics and timelines

