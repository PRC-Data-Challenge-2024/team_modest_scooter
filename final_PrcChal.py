import numpy as np
import pandas as pd
import datetime as dt
import ast
import json
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib
import timeit
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from sklearn.metrics import mean_absolute_error,mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import pytz
from datetime import datetime
import plotly.express as px
import os
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam

#%% LOAD AND PREPROCESS
dff = pd.read_csv("challenge_set.csv")
# new_df = pd.read_csv("submission_set.csv")
new_df = pd.read_csv("final_submission_set.csv")
airports_df = pd.read_csv('airports.csv')

#Combine train of sub data
df = pd.concat([dff, new_df], ignore_index=True)

airports_df = airports_df[['icao', 'time_zone']]
airports_df.columns = ['icao_code', 'timezone']
icao_to_timezone = airports_df.set_index('icao_code')['timezone'].to_dict()

def determine_season(date):
    year = date.year
    last_sunday_march = pd.date_range(start=f'{year}-03-25', end=f'{year}-03-31', freq='W-SUN')[0]
    last_saturday_october = pd.date_range(start=f'{year}-10-25', end=f'{year}-10-31', freq='W-SAT')[0]

    if last_sunday_march <= date <= last_saturday_october:
        return 'S'
    else:
        return 'W'
    
# Function to convert naive time to local time zone
def convert_to_local_time_naive(row, time_col, icao_col, icao_to_timezone):
    naive_time = row[time_col]
    icao_code = row[icao_col]
    timezone_str = icao_to_timezone.get(icao_code, 'UTC')
    timezone = pytz.timezone(timezone_str)
    local_time = naive_time.tz_localize(pytz.utc).astimezone(timezone).replace(tzinfo=None)
    return local_time

# Extract the local hour and adjust if minutes > 30
def get_adjusted_hour(times):
    hour = times.hour
    minute = times.minute
    if minute > 30:
        hour = (hour + 1) % 24  # Ensure the hour wraps around correctly at midnight
    return hour
    
# Convert time columns to datetime
# Convert to naive datetime (remove timezone information)
# df['actual_offblock_time'] = pd.to_datetime(df['actual_offblock_time'])
# df['arrival_time'] = pd.to_datetime(df['arrival_time'])
df['actual_offblock_time'] = pd.to_datetime(df['actual_offblock_time']).dt.tz_localize(None)
df['arrival_time'] = pd.to_datetime(df['arrival_time']).dt.tz_localize(None)
df['date'] = pd.to_datetime(df['date'])
df['day_of_week_num'] = df['date'].dt.dayofweek
# Add DosInt column
df['DosInt'] = np.where(df['country_code_adep'] == df['country_code_ades'], 'D', 'I')
# get the 'Season' column
df['Season'] = df['date'].apply(determine_season)

# new_df['actual_offblock_time'] = pd.to_datetime(new_df['actual_offblock_time'])
# new_df['arrival_time'] = pd.to_datetime(new_df['arrival_time'])
new_df['actual_offblock_time'] = pd.to_datetime(new_df['actual_offblock_time']).dt.tz_localize(None)
new_df['arrival_time'] = pd.to_datetime(new_df['arrival_time']).dt.tz_localize(None)
new_df['date'] = pd.to_datetime(new_df['date'])
new_df['day_of_week_num'] = new_df['date'].dt.dayofweek
new_df['DosInt'] = np.where(new_df['country_code_adep'] == new_df['country_code_ades'], 'D', 'I')
new_df['Season'] = new_df['date'].apply(determine_season)


# Convert offblock and arrival times to local times
df['local_offblock_time'] = df.apply(convert_to_local_time_naive, axis=1, args=('actual_offblock_time', 'adep', icao_to_timezone))
df['local_arrival_time'] = df.apply(convert_to_local_time_naive, axis=1, args=('arrival_time', 'ades', icao_to_timezone))
# Extract the local hour
df['offblock_hour'] = df['local_offblock_time'].apply(get_adjusted_hour)
df['arrival_hour'] = df['local_arrival_time'].apply(get_adjusted_hour)
# df['offblock_hour'] = df['actual_offblock_time'].dt.hour
# df['arrival_hour'] = df['arrival_time'].dt.hour
df['flight_duration_minutes'] = (df['arrival_time'] - df['actual_offblock_time']).dt.total_seconds() / 60.0
df['callsign_num'] = pd.factorize(df['callsign'])[0] #no need for training just for viewing because gonna use One-Hot Encoding for these columns
df['airline_num'] = pd.factorize(df['airline'])[0]

new_df['local_offblock_time'] = new_df.apply(convert_to_local_time_naive, axis=1, args=('actual_offblock_time', 'adep', icao_to_timezone))
new_df['local_arrival_time'] = new_df.apply(convert_to_local_time_naive, axis=1, args=('arrival_time', 'ades', icao_to_timezone))
# Extract the local hour
new_df['offblock_hour'] = new_df['local_offblock_time'].apply(get_adjusted_hour)
new_df['arrival_hour'] = new_df['local_arrival_time'].apply(get_adjusted_hour)
# new_df['offblock_hour'] = new_df['actual_offblock_time'].dt.hour
# new_df['arrival_hour'] = new_df['arrival_time'].dt.hour
new_df['flight_duration_minutes'] = (new_df['arrival_time'] - new_df['actual_offblock_time']).dt.total_seconds() / 60.0
new_df['callsign_num'] = pd.factorize(new_df['callsign'])[0] #no need for training just for viewing because gonna use One-Hot Encoding for these columns
new_df['airline_num'] = pd.factorize(new_df['airline'])[0]

# df.to_pickle('challangedata.pkl')
# new_df.to_pickle('submissiondata.pkl')
df.to_pickle('all_challangedata.pkl')
new_df.to_pickle('final_submissiondata.pkl')
# df = pd.read_pickle('all_challangedata.pkl')
# new_df = pd.read_pickle('final_submissiondata.pkl')

# REMOVE ALL RARE OCCURENCES
# callsign_counts = df['callsign'].value_counts()
# # rare_callsigns = callsign_counts[callsign_counts < 10].index
# # dff = df[df['callsign'].isin(rare_callsigns)]
# # df = df[~df['callsign'].isin(rare_callsigns)]

#%% Graphs

# Group by 'date' and 'aircraft_type' and calculate the mean TOW
mean_tow_per_aircraft_date = df[(df.adep == 67) & (df.ades == 77)].groupby(['date', 'aircraft_type'])['tow'].mean().reset_index()
count_aircraft_date = df[df.adep == 'LTFM'].groupby(['date', 'aircraft_type'])['tow'].count().reset_index()

# Create the plot using Plotly Express
fig = px.line(mean_tow_per_aircraft_date, x='date', y='tow', color='aircraft_type',
              title='Mean TOW per Aircraft Type for Each Date',
              labels={'tow': 'Mean TOW', 'date': 'Date', 'aircraft_type': 'Aircraft Type'})

fig1 = px.line(count_aircraft_date, x='date', y='tow', color='aircraft_type',
              title='Mean TOW per Aircraft Type for Each Date',
              labels={'tow': 'Mean TOW', 'date': 'Date', 'aircraft_type': 'Aircraft Type'})

# Show the plot
# fig.show()
fig.write_html('meantow_figure.html', auto_open=True)
fig1.write_html('first_figure.html', auto_open=True)

#%% TRAIN DATA WITH NN

# LOAD EXTRACTED ADSB data
alladsbdata = pd.read_pickle('F:/OPENSKY/alladsbdata_with_features_new.pkl')
df['descent_duration'] = round((df.arrival_time - df.descent_started_timestamp.dt.tz_localize(None)).dt.total_seconds(),2)
df['month'] = df['date'].dt.month
df['day_of_week_dep'] = df['local_offblock_time'].dt.dayofweek
df['day_of_week_des'] = df['local_arrival_time'].dt.dayofweek
df = pd.merge(df, alladsbdata, on='flight_id', how='left') #features_df
# df.to_pickle('df_with_adsb_new.pkl')
# df = pd.read_pickle('data_with_adsb.pkl')
# dff = df.copy()
df = dff.copy()

#model with no adsb
numeric_columns = ['flight_duration', 'taxiout_time', 'flown_distance']
#model with adsb no take-off
numeric_columns = ['flight_duration', 'taxiout_time', 'flown_distance',
       'avg_tas', 'min_tas', 'max_tas', 'avg_altitude', 'avg_wind_speed', 'max_wind_speed', 'std_wind_speed', 'avg_temperature',
       'avg_specific_humidity', 'std_vertical_rate']

#model with take-off
numeric_columns = ['flight_duration', 'taxiout_time', 'flown_distance',
       'avg_tas', 'min_tas', 'max_tas', 'avg_altitude', 'avg_wind_speed', 'max_wind_speed', 'std_wind_speed', 'avg_temperature',
       'avg_specific_humidity', 'std_vertical_rate_x', 'takeoff_duration',
       'mean_takeoff_tas', 'max_takeoff_tas','mean_takeoff_vertical_rate'] #

numeric_columns = ['flight_duration', 'taxiout_time', 'flown_distance',
       'avg_tas', 'min_tas', 'max_tas', 'avg_altitude', 'avg_wind_speed', 'max_wind_speed', 'std_wind_speed', 'avg_temperature',
       'avg_specific_humidity', 'std_vertical_rate_x', 'takeoff_duration',
       'mean_takeoff_tas', 'max_takeoff_tas','mean_takeoff_vertical_rate',
       'groundspeed_new'] #
cor_matrix = df[['tow'] + numeric_columns].corr() 

categories_columns = ['adep', 'ades', 'aircraft_type', 'wtc', 'airline', 'offblock_hour', 'arrival_hour', 'day_of_week_num', 'country_code_adep', 'country_code_ades', 'month']
df = df.dropna(subset=numeric_columns)

# Factorize categorical columns

categories_map = joblib.load('categories_map2.pkl')
# for key in ['datesection', 'datesectionarr']:
#     categories_map.pop(key, None)

# categories_map = {}
# df2 = pd.read_pickle('all_challangedata.pkl')
# for col in categories_columns:
#     categories_map[col] = df2[col].astype('category').cat.categories
# categories_map['month'] = df['month'].astype('category').cat.categories 
# categories_map['datesection'] = df['datesection'].astype('category').cat.categories
# categories_map['datesectionarr'] = df['datesectionarr'].astype('category').cat.categories
# categories_map['day_of_week_des'] = df['day_of_week_des'].astype('category').cat.categories
# categories_map['day_of_week_dep'] = df['day_of_week_dep'].astype('category').cat.categories

# joblib.dump(categories_map, 'categories_map2.pkl')

def apply_categories_map(data, categories_map):
    for col, categories in categories_map.items():
        data[col] = pd.Categorical(data[col], categories=categories).codes
    return data

df = apply_categories_map(df, categories_map)

# Prepare the features and target variable
X = df[numeric_columns + categories_columns]
y = df['tow']

# Standardize numerical features
scaler = StandardScaler()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Scale numeric columns
X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
X_test_scaled = scaler.transform(X_test[numeric_columns])

# Create embeddings for categorical columns
input_layers = []
embedding_layers = []

max_embedding_dim = 200

for col in categories_columns:
    input_layer = Input(shape=(1,), name=col)
    unique_vals = len(categories_map[col])
    output_dim = min(int(np.sqrt(unique_vals) + 1), max_embedding_dim)  # Apply threshold
    embedding_layer = Embedding(input_dim=unique_vals, output_dim=output_dim)(input_layer)
    embedding_layer = Flatten()(embedding_layer)
    input_layers.append(input_layer)
    embedding_layers.append(embedding_layer)

# Combine embeddings with numeric data
numeric_input = Input(shape=(len(numeric_columns),))
input_layers.append(numeric_input)
all_layers = Concatenate()(embedding_layers + [numeric_input])

# neural network for model with take-off features
dense_1 = Dense(256, activation='relu')(all_layers)
dense_2 = Dense(128, activation='relu')(dense_1)
dense_3 = Dense(64, activation='relu')(dense_2)
dense_4 = Dense(32, activation='relu')(dense_3)
output = Dense(1)(dense_4)
# neural network for model with no take-off features
# dense_1 = Dense(256, activation='relu')(all_layers)
# dense_2 = Dense(128, activation='relu')(dense_1)
# dense_3 = Dense(64, activation='relu')(dense_2)
# output = Dense(1)(dense_3)

model = Model(inputs=input_layers, outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
model.fit([X_train[col].values for col in categories_columns] + [X_train_scaled], y_train, epochs=200, batch_size=128, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the test set
y_pred = model.predict([X_test[col].values for col in categories_columns] + [X_test_scaled])
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

f_y_test = pd.DataFrame(y_test)
f_y_test['Predict'] = [item for sublist in y_pred for item in sublist]
f_y_test['ERROR'] = f_y_test.tow - f_y_test.Predict
f_y_test = f_y_test.merge(df, left_index=True, right_index=True)

model.save('tow_prediction_model_v8_with_takeoff_2607.keras')
# Save the scaler to a file
joblib.dump(scaler, 'scaler_model_v8_with_takeoff_2607.pkl')

#%%  GO!!!! NN PREDICT

new_df = pd.read_pickle('final_submissiondata.pkl')
# new_df = pd.merge(new_df, alladsbdata, on='flight_id', how='left') 
# df1 = new_df.copy()
# new_df = df1.copy()
new_df = new_df.dropna(subset=numeric_columns)
# new_df['month'] = new_df['date'].dt.month
# new_df['datesection'] = new_df['local_offblock_time'].apply(get_datesection)
# new_df['datesectionarr'] = new_df['local_arrival_time'].apply(get_datesection)

# for col in categories_columns:
#     new_df[col] = new_df[col].astype('category').cat.codes

categories_map = joblib.load('categories_map.pkl')

def apply_categories_map(data, categories_map):
    for col, categories in categories_map.items():
        data[col] = pd.Categorical(data[col], categories=categories).codes
    return data

new_df = apply_categories_map(new_df, categories_map)

scaler = joblib.load('scaler_model_v8_with_takeoff_2585.pkl')
new_X_scaled = scaler.transform(new_df[numeric_columns])

# Prepare input data for the model
new_X_input = [new_df[col].values for col in categories_columns] + [new_X_scaled]

# from tensorflow.keras.models import load_model

# # Load the saved Keras model
model = load_model('tow_prediction_model_v8_with_takeoff_2585.keras')

# Make predictions
predictions = model.predict(new_X_input)

# Add the predictions to the new_df
new_df['tow'] = predictions

# new_df.to_pickle('F:/OPENSKY/NN_Predicted_with_no_adsb.pkl')
# new_df.to_pickle('F:/OPENSKY/NN_Predicted_with_take_off_v5.pkl')
# new_df.to_pickle('F:/OPENSKY/NN_Predicted_with_no_take_off_v6.pkl')
# new_df = pd.read_pickle('F:/OPENSKY/NN_Predicted_with_no_adsb.pkl')
# new_df2 = pd.read_pickle('F:/OPENSKY/NN_Predicted_with_no_take_off.pkl')
# new_df = pd.read_pickle('F:/OPENSKY/NN_Predicted_with_no_adsb.pkl')
new_df2 = pd.read_pickle('F:/OPENSKY/NN_Predicted_with_take_off_v5.pkl')
# Merge new_df with new_df2 on flight_id
merged_df = new_df.merge(new_df2[['flight_id', 'tow']], on='flight_id', how='left', suffixes=('', '_new'))

# Update the tow column in new_df2 with values from new_df where flight_id matches
merged_df['tow'] = merged_df['tow_new'].combine_first(merged_df['tow'])

# Drop the additional tow_new column
merged_df = merged_df.drop(columns=['tow_new'])
new_df = merged_df.copy()
# new_df = new_df[~pd.isnull(new_df['tow'])]

submission = new_df[['flight_id','tow']]
submission.to_csv('F:/OPENSKY/team_modest_scooter_v5_abcba6f0-2f7d-46a4-b6f0-df998e2146e2.csv',index = False)
    

