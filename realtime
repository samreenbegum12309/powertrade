import streamlit as st
import pandas as pd
import numpy as np
import pymysql
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="MCV Forecast with LSTM", layout="wide")
st.title("📈 Market Clearing Volume (MCV) Forecast using PyTorch LSTM")

@st.cache_resource
def scrape_and_store():
    conn = pymysql.connect(host='localhost', user='root', password='root', database='power', charset='utf8mb4')
    cursor = conn.cursor()

    chrome_service = Service("C:\\Users\\SAMREEN\\Downloads\\chromedriver.exe")
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')

    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
    url = "https://www.iexindia.com/market-data/real-time-market/market-snapshot"
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    table = soup.find('table')
    data_list = []

    if table:
        rows = table.find_all('tr')[1:]
        current_hour = 1
        session_id = 1
        timeblock_count = 0
        current_date = datetime.now().strftime('%Y-%m-%d')

        for row in rows:
            cells = [cell.get_text(strip=True) for cell in row.find_all('td')]
            while len(cells) < 9:
                if len(cells) == 6:
                    cells.insert(0, current_date)
                    cells.insert(1, session_id)
                elif len(cells) == 7:
                    cells.insert(0, current_date)
                elif len(cells) == 8:
                    cells.insert(1, session_id)

            if timeblock_count % 2 == 0 and timeblock_count > 0:
                session_id += 1
            if timeblock_count % 4 == 0 and timeblock_count > 0:
                current_hour += 1
            timeblock_count += 1

            try:
                data_list.append([
                    current_date, current_hour, session_id, cells[3],
                    float(cells[4]), float(cells[5]), float(cells[6]),
                    float(cells[7]), float(cells[8])
                ])
                cursor.execute("""
                    INSERT INTO trader (date, hour, session_id, timeblock, purchase_bid, sell_bid, mcv, final_scheduled_volume, mcp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (current_date, current_hour, session_id, cells[3], float(cells[4]),
                      float(cells[5]), float(cells[6]), float(cells[7]), float(cells[8])))
            except:
                pass

        conn.commit()
        cursor.close()
        conn.close()

    return pd.DataFrame(data_list, columns=['date', 'hour', 'session_id', 'timeblock',
                                            'purchase_bid', 'sell_bid', 'mcv',
                                            'final_scheduled_volume', 'mcp'])

df = scrape_and_store()
st.subheader("📊 Latest Market Snapshot")
st.dataframe(df.tail(10), use_container_width=True)

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

numeric_cols = ['purchase_bid', 'sell_bid', 'mcv', 'final_scheduled_volume', 'mcp']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df = df.drop_duplicates()

# Outlier Removal (IQR)
for col in numeric_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]

# LSTM Preprocessing
mcv_data = df['mcv'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(mcv_data)

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

seq_len = 10
X, y = create_sequences(scaled_data, seq_len)
X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
with st.spinner("Training LSTM Model..."):
    for epoch in range(10):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor.to(device)).cpu().numpy()
    y_test_tensor_np = y_test_tensor.numpy()

y_pred_inv = scaler.inverse_transform(y_pred_tensor)
y_test_inv = scaler.inverse_transform(y_test_tensor_np)

mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
accuracy = 100 - mape

st.subheader("📉 Evaluation Metrics")
st.metric("MAE", f"{mae:.2f}")
st.metric("RMSE", f"{rmse:.2f}")
st.metric("MAPE", f"{mape:.2f}%")
st.metric("Model Accuracy", f"{accuracy:.2f}%")

# Plotting
st.subheader("📈 MCV: Actual vs Predicted")
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(y_test_inv, label="Actual")
ax1.plot(y_pred_inv, label="Predicted", linestyle="--", color="red")
ax1.set_title("LSTM Model: Actual vs Predicted MCV")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# Forecast
last_seq = torch.tensor(scaled_data[-seq_len:], dtype=torch.float32).reshape(1, seq_len, 1).to(device)
future_preds = []

with torch.no_grad():
    for _ in range(30):
        next_pred = model(last_seq)
        future_preds.append(next_pred.cpu().numpy())
        last_seq = torch.cat((last_seq[:, 1:, :], next_pred.unsqueeze(1)), dim=1)

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)

st.subheader("🔮 Forecast for Next 30 Days")
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(df.index, df['mcv'], label="Historical")
ax2.plot(future_dates, future_preds, label="Forecast", linestyle="--", color="orange")
ax2.set_title("Future MCV Forecast using LSTM")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

st.success("All data successfully inserted into the database and forecast complete!")
