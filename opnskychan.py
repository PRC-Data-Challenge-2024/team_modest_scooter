from pyopensky.config import opensky_config_dir

print(opensky_config_dir)
from pyopensky.s3 import S3Client

s3 = S3Client()

for obj in s3.s3client.list_objects("competition-data", recursive=True):
     print(f"{obj.bucket_name=}, {obj.object_name=}")
     # s3.download_object(obj)
     
#%%
import numpy as np
# from geopy.distance import geodesic
import os
import pandas as pd
import datetime as dt
import ast
import json
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib
import timeit
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from sklearn.metrics import mean_absolute_error,mean_squared_error
from tensorflow.keras.models import load_model
import pytz
from datetime import datetime
import plotly.express as px

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
# Group by 'date', 'aircraft_type', 'adep', and 'ades' and calculate the mean 'tow'
# df['mean_tow'] = df.groupby(['date', 'aircraft_type', 'adep', 'ades'])['tow'].transform('mean')
# df['mean_tow2'] = df.groupby(['callsign', 'aircraft_type'])['tow'].transform('mean')

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

#%%
# numeric_columns = df.select_dtypes(include=[np.number])

numeric_columns = ['flight_duration', 'taxiout_time', 'flown_distance']
cor_matrix = df[numeric_columns + ['tow']].corr()
categories_columns = ['adep', 'ades', 'aircraft_type', 'wtc', 'airline', 'offblock_hour', 'arrival_hour', 'day_of_week_num', 'DosInt', 'Season', 'date', 'callsign']

# Factorize categorical columns
for col in categories_columns:
    df[col] = df[col].astype('category').cat.codes

# Prepare the features and target variable
X = df[numeric_columns + categories_columns]
y = df['tow']

# Standardize numerical features 
scaler = StandardScaler()

# Save the scaler to a file
joblib.dump(scaler, 'scaler_model_Atow.pkl')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric columns
X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
X_test_scaled = scaler.transform(X_test[numeric_columns])

# Create embeddings for categorical columns
input_layers = []
embedding_layers = []

for col in categories_columns:
    input_layer = Input(shape=(1,))
    unique_vals = X[col].nunique()
    embedding_layer = Embedding(input_dim=unique_vals, output_dim=10, input_length=1)(input_layer)
    embedding_layer = Flatten()(embedding_layer)
    input_layers.append(input_layer)
    embedding_layers.append(embedding_layer)

# Combine embeddings with numeric data
numeric_input = Input(shape=(len(numeric_columns),))
input_layers.append(numeric_input)
all_layers = Concatenate()(embedding_layers + [numeric_input])

# Define the neural network
dense_1 = Dense(128, activation='relu')(all_layers)
dense_2 = Dense(64, activation='relu')(dense_1)
output = Dense(1)(dense_2)

model = Model(inputs=input_layers, outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

from tensorflow.keras.callbacks import EarlyStopping
# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
model.fit([X_train[col].values for col in categories_columns] + [X_train_scaled], y_train, epochs=50, batch_size=128, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the test set
y_pred = model.predict([X_test[col].values for col in categories_columns] + [X_test_scaled])
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Transfer prediction values to the original dataframe
f_y_test = pd.DataFrame(y_test)
f_y_test['Predict'] = [item for sublist in y_pred for item in sublist]
f_y_test['ERROR'] = f_y_test.tow - f_y_test.Predict

model.save('tow_prediction_model_v1.keras')


#%% RANNDOM FOREST

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import category_encoders as ce

X = df[numeric_columns + categories_columns]
y = df['tow']

# target_encoder = ce.TargetEncoder(cols=categories_columns) #OneHotEnCoding take too much time to train
# X_encoded = target_encoder.fit_transform(X[categories_columns], y)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categories_columns])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numeric_columns])

X_final = np.concatenate((X_encoded, X_scaled), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predicting on the test set
rf_predictions = rf_model.predict(X_test)

# Evaluate the Random Forest model
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
print(f'Random Forest RMSE: {rf_rmse}')

# Transfer prediction values to the original dataframe
f_y_test = pd.DataFrame({
    'Actual': y_test,
    'Predict': rf_predictions
})

# Merge with the original dataframe to get the corresponding columns
f_y_test = f_y_test.merge(df, left_index=True, right_index=True)

# Calculate the error
f_y_test['ERROR'] = f_y_test['Actual'] - f_y_test['Predict']

# Save the model to disk
joblib.dump(rf_model, 'random_forest_model_v2.pkl')
joblib.dump(scaler, 'scaler_v2.pkl')
joblib.dump(encoder, 'onehot_encoder_v2.pkl')

#%% Evaluate the errors

# Merge with the original dataframe to get the corresponding columns
f_y_test = f_y_test.merge(df, left_index=True, right_index=True)

# Create a DataFrame for all dates in 2022
all_dates = pd.date_range(start='2022-01-01', end='2022-12-31')
all_dates_df = pd.DataFrame(all_dates, columns=['date'])

# Filter the DataFrame for large errors
large_errors = f_y_test[abs(f_y_test['ERROR']) >= 5000]
check  = large_errors.sort_values(by=['ERROR','date'])

# Count total flights and large errors for each date
total_flights_per_date = df['date'].value_counts().sort_index()
large_errors_per_date = large_errors['date'].value_counts().sort_index()

# Merge counts with all dates DataFrame
all_dates_df = all_dates_df.set_index('date')
all_dates_df['total_flights'] = total_flights_per_date
all_dates_df['large_errors'] = large_errors_per_date

# Fill NaNs with 0
all_dates_df = all_dates_df.fillna(0)

# Calculate the ratio
all_dates_df['error_ratio'] = all_dates_df['large_errors'] / all_dates_df['total_flights']

# Reset the index
all_dates_df = all_dates_df.reset_index()

# Create the histogram
fig = px.histogram(
    all_dates_df,
    x='date',
    y='error_ratio',
    title='Ratio of Large Errors (>5000) Over Total Flights by Date in 2022',
    nbins=365  # Ensure we cover all dates
)

# Write the figure to an HTML file
fig.write_html('large_errors_ratio_by_date.html', auto_open=True)

#Error by ROUTES
# Count total flights and large errors for each combination of adep and ades
total_flights_per_route = df.groupby(['adep', 'ades']).size().reset_index(name='total_flights')
large_errors_per_route = large_errors.groupby(['adep', 'ades']).size().reset_index(name='large_errors')

# Merge counts with the total flights DataFrame
route_df = pd.merge(total_flights_per_route, large_errors_per_route, on=['adep', 'ades'], how='left')

# Fill NaNs with 0
route_df = route_df.fillna(0)
route_df = route_df[route_df.total_flights >= 5]

# Calculate the ratio
route_df['error_ratio'] = route_df['large_errors'] / route_df['total_flights']

# Create a combined column for adep and ades
route_df['route'] = route_df['adep'].astype(str) + ' - ' + route_df['ades'].astype(str)

# Create the histogram
fig = px.histogram(
    route_df,
    x='route',
    y='error_ratio',
    title='Ratio of Large Errors (>5000) Over Total Flights by Route (adep - ades)',
    nbins=len(route_df),  # Ensure we cover all routes
    labels={'route': 'Route (adep - ades)', 'error_ratio': 'Error Ratio'}
)

# Rotate x-axis labels for better readability
fig.update_layout(
    xaxis_tickangle=-45
)

# Write the figure to an HTML file
fig.write_html('large_errors_ratio_by_route.html', auto_open=True)

# 
# Count total flights and large errors for each callsign
total_flights_per_callsign = df.groupby('callsign').size().reset_index(name='total_flights')
large_errors_per_callsign = large_errors.groupby('callsign').size().reset_index(name='large_errors')

# Merge counts with the total flights DataFrame
callsign_df = pd.merge(total_flights_per_callsign, large_errors_per_callsign, on='callsign', how='left')

# Fill NaNs with 0
callsign_df = callsign_df.fillna(0)
callsign_df = callsign_df[callsign_df.total_flights >= 5]

# Calculate the ratio
callsign_df['error_ratio'] = callsign_df['large_errors'] / callsign_df['total_flights']

# Create the histogram
fig = px.histogram(
    callsign_df,
    x='callsign',
    y='error_ratio',
    title='Ratio of Large Errors (>5000) Over Total Flights by Callsign',
    nbins=len(callsign_df),  # Ensure we cover all callsigns
    labels={'callsign': 'Callsign', 'error_ratio': 'Error Ratio'}
)

# Rotate x-axis labels for better readability
fig.update_layout(
    xaxis_tickangle=-45
)

# Write the figure to an HTML file
fig.write_html('large_errors_ratio_by_callsign.html', auto_open=True)

# 

total_flights_per_airline = df.groupby('airline').size().reset_index(name='total_flights')
large_errors_per_airline = large_errors.groupby('airline').size().reset_index(name='large_errors')

# Merge counts with the total flights DataFrame
airline_df = pd.merge(total_flights_per_airline, large_errors_per_airline, on='airline', how='left')

# Fill NaNs with 0
airline_df = airline_df.fillna(0)

# Calculate the ratio
airline_df['error_ratio'] = airline_df['large_errors'] / airline_df['total_flights']

# Create the histogram
fig = px.histogram(
    airline_df,
    x='airline',
    y='error_ratio',
    title='Ratio of Large Errors (>5000) Over Total Flights by Airline',
    nbins=len(airline_df),  # Ensure we cover all airlines
    labels={'airline': 'Airline', 'error_ratio': 'Error Ratio'}
)

# Rotate x-axis labels for better readability
fig.update_layout(
    xaxis_tickangle=-45
)

# Write the figure to an HTML file
fig.write_html('large_errors_ratio_by_airline.html', auto_open=True)

# 

total_flights_per_aircraft_type = df.groupby('aircraft_type').size().reset_index(name='total_flights')
large_errors_per_aircraft_type = large_errors.groupby('aircraft_type').size().reset_index(name='large_errors')

# Merge counts with the total flights DataFrame
aircraft_type_df = pd.merge(total_flights_per_aircraft_type, large_errors_per_aircraft_type, on='aircraft_type', how='left')

# Fill NaNs with 0
aircraft_type_df = aircraft_type_df.fillna(0)

# Calculate the ratio
aircraft_type_df['error_ratio'] = aircraft_type_df['large_errors'] / aircraft_type_df['total_flights']

# Create the histogram
fig = px.histogram(
    aircraft_type_df,
    x='aircraft_type',
    y='error_ratio',
    title='Ratio of Large Errors (>5000) Over Total Flights by Aircraft Type',
    nbins=len(aircraft_type_df),  # Ensure we cover all aircraft types
    labels={'aircraft_type': 'Aircraft Type', 'error_ratio': 'Error Ratio'}
)

# Rotate x-axis labels for better readability
fig.update_layout(
    xaxis_tickangle=-45
)

# Write the figure to an HTML file
fig.write_html('large_errors_ratio_by_aircraft_type.html', auto_open=True)
#%% NN PREDICT
# scaler = joblib.load('scaler_model_Atow.pkl')
# model = load_model('tow_prediction_model_v1.keras')

df = pd.read_pickle('challangedata.pkl')
df['mean_tow'] = df.groupby(['date', 'aircraft_type', 'adep', 'ades'])['tow'].transform('mean')
dff = df.drop_duplicates(subset=['date', 'aircraft_type', 'adep', 'ades'], keep='first').reset_index(drop=True)
new_df = pd.read_pickle('submissiondata.pkl')

#match first time
# Merge new_df with df on ['date', 'aircraft_type', 'adep', 'ades']
merged_df = new_df.merge(dff[['date', 'aircraft_type', 'adep', 'ades', 'mean_tow']], 
                         on=['date', 'aircraft_type', 'adep', 'ades'], 
                         how='left', suffixes=('', '_orig'))

merged_df = merged_df.drop_duplicates(subset=['flight_id','callsign','date', 'aircraft_type', 'adep', 'ades'], keep='first').reset_index(drop=True)
# new_df = new_df.drop_duplicates(subset=['flight_id','callsign','date', 'aircraft_type', 'adep', 'ades'], keep='first').reset_index(drop=True)

missing_mean_tow_df = merged_df[merged_df['mean_tow'].isnull()]

merged_missing_df = missing_mean_tow_df.merge(dff[['day_of_week_num', 'aircraft_type', 'adep', 'ades', 'mean_tow', 'date']], 
                                              on=['day_of_week_num', 'aircraft_type', 'adep', 'ades'], 
                                              how='left', suffixes=('', '_2'))

# Get the column of datetimedifference of date and date_2
merged_missing_df['datediff'] = abs((merged_missing_df.date_2 - merged_missing_df.date).dt.total_seconds())
merged_missing_df = merged_missing_df.sort_values(by=['datediff']).drop_duplicates(subset=['flight_id','day_of_week_num', 'aircraft_type', 'adep', 'ades'], keep='first')
merged_missing_df.mean_tow = merged_missing_df.mean_tow_2

missing_mean_tow_df_2 = merged_missing_df[merged_missing_df['mean_tow'].isnull()]
merged_missing_df_2 = missing_mean_tow_df_2.merge(dff[['callsign', 'adep', 'ades', 'mean_tow', 'date']], 
                                              on=['callsign', 'adep', 'ades'], 
                                              how='left', suffixes=('', '_3'))
merged_missing_df_2['datediff'] = abs((merged_missing_df_2.date_2 - merged_missing_df_2.date).dt.total_seconds())
merged_missing_df_2 = merged_missing_df_2.sort_values(by=['datediff']).drop_duplicates(subset=['flight_id','callsign', 'adep', 'ades'], keep='first')
merged_missing_df_2.mean_tow = merged_missing_df_2.mean_tow_3
# Check = merged_missing_df_2[merged_missing_df_2['mean_tow_3'].isnull()]

# Merge to new_df
new_df = new_df.merge(merged_df[['flight_id', 'mean_tow']], on='flight_id', how='left', suffixes=('', '_merged'))
new_df['mean_tow'] = new_df['mean_tow'].combine_first(
    new_df.merge(merged_missing_df[['flight_id', 'mean_tow']], on='flight_id', how='left', suffixes=('', '_merged2'))['mean_tow_merged2']
)
new_df['mean_tow'] = new_df['mean_tow'].combine_first(
    new_df.merge(merged_missing_df_2[['flight_id', 'mean_tow']], on='flight_id', how='left', suffixes=('', '_merged3'))['mean_tow_merged3']
)

mean_tow_by_aircraft_type = df.groupby('aircraft_type')['tow'].mean().to_dict()
new_df['mean_tow'] = new_df.apply(
    lambda row: mean_tow_by_aircraft_type.get(row['aircraft_type'], np.nan) if pd.isnull(row['mean_tow']) else row['mean_tow'],
    axis=1
)
Check = new_df[new_df['mean_tow'].isnull()]
# new_df = new_df[~new_df['mean_tow'].isnull()]

#%%  GO!!!! NN PREDICT

# new_df = pd.read_pickle('final_submissiondata.pkl')
# new_df = pd.merge(new_df, alladsbdata, on='flight_id', how='left') 
# df1 = new_df.copy()
new_df = df1.copy()
new_df = new_df.dropna(subset=numeric_columns)

# for col in categories_columns:
#     new_df[col] = new_df[col].astype('category').cat.codes

categories_map = joblib.load('categories_map.pkl')

def apply_categories_map(data, categories_map):
    for col, categories in categories_map.items():
        data[col] = pd.Categorical(data[col], categories=categories).codes
    return data

new_df = apply_categories_map(new_df, categories_map)

new_X_scaled = scaler.transform(new_df[numeric_columns])

# Prepare input data for the model
new_X_input = [new_df[col].values for col in categories_columns] + [new_X_scaled]


# from tensorflow.keras.models import load_model

# # Load the saved Keras model
# model = load_model('tow_prediction_model_v5.1_with_take_off_2792.keras')

# Make predictions
predictions = model.predict(new_X_input)

# Add the predictions to the new_df
new_df['tow'] = predictions

new_df.to_pickle('F:/OPENSKY/NN_Predicted_with_no_adsb.pkl')
# new_df = pd.read_pickle('F:/OPENSKY/NN_Predicted_with_no_adsb.pkl')
# new_df2 = pd.read_pickle('F:/OPENSKY/NN_Predicted_with_no_take_off.pkl')
new_df2 = pd.read_pickle('F:/OPENSKY/NN_Predicted_with_take_off.pkl')
# Merge new_df with new_df2 on flight_id
merged_df = new_df.merge(new_df2[['flight_id', 'tow']], on='flight_id', how='left', suffixes=('', '_new'))

# Update the tow column in new_df2 with values from new_df where flight_id matches
merged_df['tow'] = merged_df['tow_new'].combine_first(merged_df['tow'])

# Drop the additional tow_new column
merged_df = merged_df.drop(columns=['tow_new'])
new_df = merged_df.copy()
new_df = new_df[~pd.isnull(new_df['tow'])]

submission = new_df[['flight_id','tow']]
submission.to_csv('F:/OPENSKY/team_modest_scooter_v4_abcba6f0-2f7d-46a4-b6f0-df998e2146e2.csv',index = False)

#combine 2 predictions:
new_df_with_tow = new_df[~new_df['mean_tow'].isnull()]
v1 = pd.read_csv('F:/OPENSKY/team_modest_scooter_v1_abcba6f0-2f7d-46a4-b6f0-df998e2146e2.csv')

# Merge v1 with new_df on 'flight_id' to get 'tow' values from new_df
merged_v1 = v1.merge(new_df[['flight_id', 'tow']], on='flight_id', how='left', suffixes=('', '_new'))

# Update 'tow' values in v1 with 'tow' values from new_df where available
merged_v1['tow'] = merged_v1['tow_new'].combine_first(merged_v1['tow'])

submission2 = merged_v1[['flight_id','tow']]
submission2['tow'] = submission2['tow'].round(2)
submission2.to_csv('F:/OPENSKY/team_modest_scooter_v3_abcba6f0-2f7d-46a4-b6f0-df998e2146e2.csv',index = False)

#%% RF PREDICT

# rf_model = joblib.load('random_forest_model.pkl')
# encoder = joblib.load('onehot_encoder.pkl')
# scaler = joblib.load('scaler.pkl')

# One-Hot Encoding for categorical columns in new_df
new_df_encoded = encoder.transform(new_df[categories_columns])

# Scaling numeric features in new_df
new_df_scaled = scaler.transform(new_df[numeric_columns])

# Combine encoded and scaled features
new_df_final = np.concatenate((new_df_encoded, new_df_scaled), axis=1)

# Make predictions using the loaded model
predictions = rf_model.predict(new_df_final)

# Create a DataFrame with the predictions
new_df['tow'] = predictions
submission = new_df[['flight_id','tow']]
submission.to_csv('F:/OPENSKY/team_modest_scooter_v1_abcba6f0-2f7d-46a4-b6f0-df998e2146e2.csv',index = False)

    
