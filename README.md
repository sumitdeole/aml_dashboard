# Credit Card Fraud Detection Dashboard

![Dashboard Screenshot](https://via.placeholder.com/800x400?text=Fraud+Detection+Dashboard+Screenshot)

A Streamlit-powered interactive dashboard for analyzing credit card fraud patterns through network visualization and transaction analysis. 

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
3. The file structure should look like:
your-project/
├── data/
│ └── creditcard.csv
├── app.py
└── README.md


## Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/credit-card-fraud-dashboard.git
cd credit-card-fraud-dashboard

2. Install dependencies:
```python
pip install streamlit pandas numpy plotly networkx matplotlib scikit-learn imbalanced-learn

3. Run the dashboard:
```python
streamlit run app.py


## Dashboard Features
- Client Selection: Choose between example fraud/legitimate clients or random clients
- Network Visualization: See transaction patterns as interactive network graphs
- Detailed Transaction View: Explore all transactions with fraud highlighting
- Fraud Analytics: View fraud-specific metrics and timelines

