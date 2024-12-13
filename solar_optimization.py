import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from datetime import datetime

# Ignore warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('solar_data.csv')

# Function to calculate zenith angle
def cal_zenith_angle(row):
    if row['SZA'] != -999:
        return row['SZA']
    
    day = row['DY']
    hour = row['HR']
    
    delta = 23.45 * math.sin(360 * (day - 80) / 365)  # Solar Declination angle
    omega = 15 * (hour - 12)  # Hour angle
    theta = math.acos((math.sin(23.03) * math.sin(delta)) + (math.cos(23.03) * math.cos(delta) * math.cos(omega)))
    return 90 - theta

# Calculate new SZA and tilt
df['new_SZA'] = df.apply(cal_zenith_angle, axis=1)
df['tilt'] = 90 - df['new_SZA']

# Clean data
df.drop(['SZA', 'new_SZA'], axis=1, inplace=True)
df = df[(df['CLRSKY_SFC_SW_DWN'] != -999) & (df['ALLSKY_SFC_SW_DWN'] != -999) & (df['ALLSKY_KT'] != -999) & (df['ALLSKY_SRF_ALB'] != -999)]
df.reset_index(drop=True, inplace=True)


# Modelling
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Prepare data for modeling
data_df = df.drop(['tilt'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(data_df, df["tilt"], random_state=42, test_size=0.20)

# Scale features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and fit MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler().fit(X_train)

# Train Random Forest model
model = RandomForestRegressor(random_state=12345)
model.fit(X_train_scaled, y_train)

score_type = ['neg_mean_absolute_error', 'neg_mean_squared_error']
model_results = pd.DataFrame()
for m in ['Train_MAE', 'Val_MAE', 'Train_RMSE', 'Val_RMSE', 'test_MAE', 'test_RMSE']:
    model_results[m] = None

# Define models dictionary
models = {
    'Random Forest': RandomForestRegressor(random_state=12345)
}

for model_name, model in models.items():
    score = cross_validate(model, X_train_scaled, y_train, cv=5, scoring=score_type, return_train_score=True)

    train_mae = (-score['train_neg_mean_absolute_error']).mean()
    val_mae = (-score['test_neg_mean_absolute_error']).mean()
    train_rmse = np.sqrt(-score['train_neg_mean_squared_error']).mean()
    val_rmse = np.sqrt(-score['test_neg_mean_squared_error']).mean()

    res = model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    test_rmse = mean_squared_error(y_test, predictions)
    test_mae = mean_absolute_error(y_test, predictions)
    model_results.loc[model_name] = [train_mae, val_mae, train_rmse, val_rmse,test_mae, test_rmse]
model_results.to_csv('model_results.csv')

model_dict = {
    'Random Forest': {'model': RandomForestRegressor(random_state=12345, n_jobs=-1), 
                      'params': {'n_estimators': list(range(5, 50, 5)), 
                                 'min_samples_split': [2, 5, 10]}}
}

def hyperparameter_tuning():
    best_model = None
    best_score = -math.inf

    for model_name, reg_model in model_dict.items():
        hyper_tuning_model = RandomizedSearchCV(reg_model['model'], reg_model['params'], n_iter=10, cv=5, return_train_score=True, verbose=0, scoring=score_type, refit='neg_mean_squared_error')
        hyper_tuning_model.fit(X_train_scaled, y_train)

        model_res = hyper_tuning_model.best_estimator_
        best_model_score = hyper_tuning_model.best_score_

        res = hyper_tuning_model.cv_results_

        if best_model_score > best_score:
            best_score = best_model_score
            best_model = model_res
        
    return best_model

best_model = hyperparameter_tuning()

np.mean(((best_model.predict(X_test_scaled) - y_test) ** 2))

df['date_time_str'] = df.apply(lambda x: f"{int(x['MO'])}/{int(x['DY'])}/{int(x['YEAR'])} {int(x['HR'])}:00:00", axis=1)
df['date_time'] = pd.to_datetime(df['date_time_str'], format="%m/%d/%Y %H:%M:%S")
df = df.drop(['date_time_str'], axis=1)
data = df.drop(['YEAR', 'MO', 'DY', 'HR', 'date_time'], axis=1)
dataset = data.values

training_data_len = int(np.ceil(len(dataset) * .95))
training_data_len

minmax_scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = minmax_scaler.fit_transform(dataset)
train_data = scaled_data[0:int(training_data_len), :]
x_train, y_train = [], []

for i in range(120,  len(train_data)):
    x_train.append(train_data[i-120:i, :-1])
    y_train.append(train_data[i, -1])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 7))


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM


#  LSTM Model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 7)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

checkpoint_cb = keras.callbacks.ModelCheckpoint("LSTM_Model_multi_features.keras", save_best_only=True, monitor='val_loss')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
lr_decay_cb = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=1, factor=0.5, min_lr=1e-8)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

History = model.fit(x_train, y_train, batch_size=32, epochs=32, validation_split=0.15, verbose=1)


# Plotting the model loss
plt.plot(History.history['loss'], 'g')
plt.plot(History.history['val_loss'], 'b')
plt.title('Model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='best')
plt.savefig('Model_loss.png')
plt.show()


# Predictions
test_data = scaled_data[training_data_len - 120:, :]
x_test = []
y_test = scaled_data[training_data_len:, -1]
for i in range(120, len(test_data)):
    x_test.append(test_data[i-120:i,:-1])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 7))
predictions = model.predict(x_test)
pred = predictions.reshape((predictions.shape[0]))
rmse = np.sqrt(np.mean((pred - y_test) ** 2))

tmp = pd.DataFrame(scaled_data[training_data_len:,:-1])
tmp[7] = predictions

res = minmax_scaler.inverse_transform(tmp.values)
tmp = pd.DataFrame(res)
tmp2 = df.iloc[training_data_len:, :]
tmp2['pred'] = tmp[7].tolist()


# Plotting the Hourly Tilt Angle Predictions
val = tmp2[(tmp2['date_time'] >= pd.Timestamp(2024, 4, 1, 0)) & (tmp2['date_time'] < pd.Timestamp(2024, 12, 2, 0))]
sns.lineplot(data=val, x='HR', y='pred', palette='sky_blue')
plt.xlabel('Hour')
plt.ylabel('Tilt Angle')
plt.title('Hourly tilt angle for 1st April, 2024')
plt.savefig('hourly_tilt_angle.png')
plt.show()



x_total = []
for i in range(120,  len(scaled_data)):
    x_total.append(scaled_data[i-120:i, :-1])
x_total = np.array(x_total)
y_pred = model.predict(x_total)
y_pred = y_pred.reshape((y_pred.shape[0]))
df_full = pd.DataFrame(scaled_data[120:,:-1])
df_full[7] = y_pred.tolist()

tmp_val = minmax_scaler.inverse_transform(df_full.values)
df_full = pd.DataFrame(tmp_val)
df_full['date_time'] = df['date_time'].tolist()[120:]

df_year = df_full[(df_full['date_time'] >= pd.Timestamp(2023,1,1,0)) & (df_full['date_time'] < pd.Timestamp(2024,1,1,0))]
df_year['month'] = df_year['date_time'].apply(lambda x: x.month)
df_year['day'] = df_year['date_time'].apply(lambda x: x.day)


# Plotting the Monthly Tilt Angle Predictions
monthly_tilt = df_year.groupby(['month']).mean()
sns.lineplot(data=monthly_tilt, x='month', y=7, palette='sky_blue')
plt.xlabel('Month')
plt.ylabel('Tilt Angle')
plt.title('Monthly Average Tilt Angle')
plt.savefig('monthly_tilt_angle.png')
plt.show()


# Plotting the Daily Optimal Angle Predictions
_df = df_year.groupby(['month', 'day']).mean()
months = ['Januaray', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
plt.figure(figsize=(10,8))
sns.lineplot(data=_df, y=7, x='day', hue='month', legend='full', palette='flare')
plt.xlabel('Day')
plt.ylabel('Tilt Angle')
plt.title('Daily Optimal angle for each month')
plt.savefig('daily_optimal_angle.png')
plt.show()



