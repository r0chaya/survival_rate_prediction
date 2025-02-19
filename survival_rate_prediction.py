import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# 1. Load Dataset
cycles_df = pd.read_csv("cycles.csv")
harvests_df = pd.read_csv("harvests.csv")

# 2. Data Cleaning & Preprocessing
harvests_df['total_harvested'] = harvests_df['size'] * harvests_df['weight']
harvest_summary = harvests_df.groupby('cycle_id')['total_harvested'].sum().reset_index()
cycles_df = cycles_df.rename(columns={'id': 'cycle_id'})
sr_df = cycles_df[['cycle_id', 'total_seed', 'area', 'target_cultivation_day']].merge(harvest_summary, on='cycle_id', how='left')
sr_df['survival_rate'] = sr_df['total_harvested'] / sr_df['total_seed']
sr_df.dropna(inplace=True)

# 3. Model Training
X = sr_df[['total_seed', 'area', 'target_cultivation_day']]
y = sr_df['survival_rate']

# Balancing data dengan RandomUnderSampler
# rus = RandomUnderSampler(random_state=42)
# X_resampled, y_resampled = rus.fit_resample(X, y)

# Pembagian dataset: train, validasi, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Fungsi untuk menghapus outlier menggunakan IQR
def remove_outliers(data, target_col):
    data_clean = data.copy()
    num_cols = data_clean.select_dtypes(include=['number']).columns.tolist()

    for col in num_cols:
        if col != target_col:
            Q1 = data_clean[col].quantile(0.25)
            Q3 = data_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            data_clean = data_clean[
                (data_clean[col] >= (Q1 - 1.5 * IQR)) &
                (data_clean[col] <= (Q3 + 1.5 * IQR))
            ]
    #data_clean[target_col]=data_clean[target_col].clip(0,1)

    return data_clean

# Gabungkan kembali fitur dan target untuk membersihkan outlier
train_data = pd.concat([X_train, y_train], axis=1)
train_data_cleaned = remove_outliers(train_data, 'survival_rate' )

# Pisahkan kembali fitur dan target setelah pembersihan outlier
X_train_cleaned = train_data_cleaned[['total_seed', 'area', 'target_cultivation_day']]
y_train_cleaned = train_data_cleaned['survival_rate']


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_cleaned, y_train_cleaned)

# Evaluasi model
y_pred_valid = model.predict(X_valid)
y_pred_test = model.predict(X_test)
data = {'MAE' : [mean_absolute_error(y_valid, y_pred_valid),
                 mean_absolute_error(y_test, y_pred_test)],
        'MSE': [mean_squared_error(y_valid, y_pred_valid),
                mean_squared_error(y_test, y_pred_test)],
        'R-squared': [r2_score(y_valid, y_pred_valid), r2_score(y_test, y_pred_test)]}
df = pd.DataFrame(data, index=['validation Performance','Test Performance'])

print("Validation Performance:")
print("MAE:", mean_absolute_error(y_valid, y_pred_valid))
print("MSE:", mean_squared_error(y_valid, y_pred_valid))
print("R-squared:", r2_score(y_valid, y_pred_valid))

print("\nTest Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_test))
print("MSE:", mean_squared_error(y_test, y_pred_test))
print("R-squared:", r2_score(y_test, y_pred_test))

# Visualisasi hasil prediksi vs aktual
#plt.figure(figsize=(8, 6))
# sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.7)
# plt.xlabel("Actual Survival Rate")
# plt.ylabel("Predicted Survival Rate")
# plt.title("Actual vs Predicted Survival Rate")
# plt.show()

# 4. Save Model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# 5. Streamlit App
st.title("Shrimp Survival Rate Prediction")

st.sidebar.header("Input Features", divider=True)
total_seed = st.sidebar.number_input("Total Seed", min_value=1, value=50000)
area = st.sidebar.number_input("Pond Area (m²)", min_value=1.0, value=500.0)
target_cultivation_day = st.sidebar.number_input("Target Cultivation Day", min_value=1, value=90)

if st.sidebar.button("Predict Survival Rate", help="Click this button to predict", icon=":material/calculate:"):
    features = np.array([[total_seed, area, target_cultivation_day]])
    prediction = model.predict(features)
    st.write(f"Predicted Survival Rate: {prediction[0]:.2f}")
    st.table(df)
