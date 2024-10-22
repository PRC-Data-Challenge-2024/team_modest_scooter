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

df = pd.read_pickle('challangedata.pkl')
# Step 1: Group by 'callsign' and calculate mean 'tow'
mean_tow_df = df.groupby('callsign')['tow'].mean().reset_index()
mean_tow_df.rename(columns={'tow': 'mean_tow'}, inplace=True)

# Step 2: Merge mean 'tow' back to the original dataframe
df = df.merge(mean_tow_df, on='callsign', how='left')

# Step 3: Sort by 'callsign' and keep the first 5 records for each 'callsign'
dff = df.groupby('callsign').head(5).sort_values(by=['callsign','date']).reset_index(drop=True)
check = dff[['callsign','date','tow','mean_tow', 'aircraft_type', 'adep', 'ades']]

#%% Checkdata adsb
adsb = pd.read_parquet('2022-09-01.parquet')
adsb1 = pd.read_parquet('C:/Users/admin/2022-08-30.parquet')
listflights = adsb.flight_id.unique()
temp = adsb[adsb.flight_id == 254939374].sort_values(by='timestamp')
df_id = df[df.flight_id == 254939374]

#%%

# Directory containing the parquet files
directory = 'C:/Users/admin'
# Initialize an empty list to hold the feature dataframes for all files
# alladsbdata = pd.DataFrame()
# alladsbdata2 = pd.read_pickle('F:/OPENSKY/alladsbdata.pkl') 
alladsbdata = pd.read_pickle('F:/OPENSKY/alladsbdata_with_features_new.pkl')

def extract_flight_features(data):
    features = {}
    
    # Ensure the data is sorted by timestamp
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(by='timestamp')
    
    # Flight Performance Metrics
    features['avg_groundspeed'] = data['groundspeed'].mean()
    features['max_groundspeed'] = data['groundspeed'].max()
    features['min_groundspeed'] = data['groundspeed'].min()
    features['std_groundspeed'] = data['groundspeed'].std()
    
    features['avg_vertical_rate'] = data['vertical_rate'].mean()
    features['max_vertical_rate'] = data['vertical_rate'].max()
    features['min_vertical_rate'] = data['vertical_rate'].min()
    features['std_vertical_rate'] = data['vertical_rate'].std()
    
    # Positional Metrics
    features['total_distance'] = data[['latitude', 'longitude']].diff().pow(2).sum(1).pow(0.5).sum()
    
    features['avg_altitude'] = data['altitude'].mean()
    features['max_altitude'] = data['altitude'].max()
    features['min_altitude'] = data['altitude'].min()
    features['std_altitude'] = data['altitude'].std()
    
    # Environmental Metrics
    features['avg_temperature'] = data['temperature'].mean()
    
    wind_speed = np.sqrt(data['u_component_of_wind']**2 + data['v_component_of_wind']**2)
    features['avg_wind_speed'] = wind_speed.mean()
    features['max_wind_speed'] = wind_speed.max()
    features['min_wind_speed'] = wind_speed.min()
    features['std_wind_speed'] = wind_speed.std()
    
    features['avg_specific_humidity'] = data['specific_humidity'].mean()
    
    # Extracting initial climb rate and final descent rate
    features['initial_climb_rate'] = data[data['timestamp'] <= data['timestamp'].iloc[0] + pd.Timedelta(minutes=5)]['vertical_rate'].mean()
    features['final_descent_rate'] = data[data['timestamp'] >= data['timestamp'].iloc[-1] - pd.Timedelta(minutes=5)]['vertical_rate'].mean()
    
    # Detecting climb finished timestamp
    stable_flight = data[(data['vertical_rate'] > -100) & (data['vertical_rate'] < 100)]
    if not stable_flight.empty:
        longest_stable_period = stable_flight['timestamp'].diff().gt(pd.Timedelta(minutes=5)).cumsum().value_counts().idxmax()
        stable_period = stable_flight[stable_flight['timestamp'].diff().gt(pd.Timedelta(minutes=5)).cumsum() == longest_stable_period]
        climb_finished = stable_period['timestamp'].min()
    else:
        climb_finished = np.nan
    
    features['climb_finished_timestamp'] = climb_finished

    # Detecting descent started timestamp
    # Focus on the descent phase closest to landing time
    descent_start_candidates = data[data['vertical_rate'] < -100]
    if not descent_start_candidates.empty:
        descent_started = descent_start_candidates[descent_start_candidates['timestamp'] >= data['timestamp'].iloc[-1] - pd.Timedelta(minutes=30)]
        if not descent_started.empty:
            descent_started_timestamp = descent_started['timestamp'].min()
        else:
            descent_started_timestamp = descent_start_candidates['timestamp'].min()
    else:
        descent_started_timestamp = np.nan
    
    features['descent_started_timestamp'] = descent_started_timestamp
    
    return features

# Process each file and extract features
for filename in tqdm(os.listdir(directory)):
    if filename.endswith(".parquet"):
        file_path = os.path.join(directory, filename)
        
        # Load the parquet file
        adsb = pd.read_parquet(file_path)
        
        # Group by flight_id and apply feature extraction
        grouped = adsb.groupby('flight_id')
        feature_list = []

        for flight_id, group in grouped:
            features = extract_flight_features(group)
            features['flight_id'] = flight_id  # Add flight_id to the features
            feature_list.append(features)
        
        # Create a DataFrame from the features list
        features_df = pd.DataFrame(feature_list)
        
        # Append the features_df to the alladsbdata DataFrame
        alladsbdata = pd.concat([alladsbdata, features_df], ignore_index=True)
        
        # Save the combined features to a pickle file after each file is processed
        alladsbdata.to_pickle('F:/OPENSKY/alladsbdata.pkl')

print("All ADS-B data has been processed and saved to 'F:/OPENSKY/alladsbdata.pkl'")

# Merge the dataframes on 'flight_id'
df = pd.merge(df, alladsbdata, on='flight_id', how='left') #features_df
# df = df[~pd.isnull(df.avg_groundspeed)]
# df = df[~pd.isnull(df.takeoff_duration)]
# df = df.dropna(subset=numeric_columns)
# df['climb_duration'] = round(((df.climb_finished_timestamp.dt.tz_localize(None) - df.actual_offblock_time).dt.total_seconds() - df.taxiout_time*60),2)
df['descent_duration'] = round((df.arrival_time - df.descent_started_timestamp.dt.tz_localize(None)).dt.total_seconds(),2)
# df.to_pickle('data_with_adsb.pkl')
# df = pd.read_pickle('data_with_adsb.pkl')
# dff = df.copy()
df = dff.copy()
# df = df[(df.initial_climb_rate > 500) & (df.climb_duration > 300) & (df.climb_duration <= 2700)]
df = df[(df.climb_duration > 300) & (df.climb_duration <= 2700)]

#%% Updated alladsbdata

alladsbdata = pd.merge(alladsbdata, df, on='flight_id', how='left')
alladsbdata.rename(columns=lambda x: x + '_old' if x != 'flight_id' else x, inplace=True)


#%%
# kiem tra list categories cua ban submission DF co nam trong DF cua TRaining het chua???????

# numeric_columns = ['flight_duration', 'taxiout_time', 'flown_distance', 'avg_groundspeed', 'max_groundspeed', 'std_vertical_rate', 'avg_altitude', 'avg_temperature', 'avg_wind_speed', 'max_wind_speed', 'std_wind_speed', 'avg_specific_humidity', 'climb_duration']
# numeric_columns = [
#        'avg_tas', 'min_tas', 'max_tas', 'std_tas', 'avg_track', 'min_track',
#        'max_track', 'std_track', 'avg_groundspeed', 'max_groundspeed',
#        'min_groundspeed', 'std_groundspeed', 'avg_altitude', 'max_altitude',
#        'min_altitude', 'std_altitude', 'avg_wind_speed', 'max_wind_speed',
#        'min_wind_speed', 'std_wind_speed', 'avg_temperature',
#        'avg_specific_humidity', 'initial_climb_rate', 'final_descent_rate',
#        'avg_vertical_rate', 'max_vertical_rate',
#        'min_vertical_rate', 'std_vertical_rate', 'takeoff_duration',
#        'mean_takeoff_groundspeed', 'min_takeoff_groundspeed',
#        'max_takeoff_groundspeed', 'mean_takeoff_tas', 'min_takeoff_tas',
#        'max_takeoff_tas', 'mean_takeoff_vertical_rate',
#        'min_takeoff_vertical_rate', 'max_takeoff_vertical_rate',
#        'mean_takeoff_temperature', 'mean_takeoff_specific_humidity',
#        'mean_takeoff_wind_speed', 'u_windspeed_take_off',
#        'v_windspeed_take_off', 'descent_duration']

#model with no adsb
numeric_columns = ['flight_duration', 'taxiout_time', 'flown_distance']
#model with adsb no take-off
numeric_columns = ['flight_duration', 'taxiout_time', 'flown_distance',
       'avg_tas', 'min_tas', 'max_tas', 'avg_altitude', 'avg_wind_speed', 'max_wind_speed', 'std_wind_speed', 'avg_temperature',
       'avg_specific_humidity', 'std_vertical_rate']

#model with take-off
numeric_columns = ['flight_duration', 'taxiout_time', 'flown_distance',
       'avg_tas', 'min_tas', 'max_tas', 'avg_altitude', 'avg_wind_speed', 'max_wind_speed', 'std_wind_speed', 'avg_temperature',
       'avg_specific_humidity', 'std_vertical_rate', 'takeoff_duration',
       'mean_takeoff_tas', 'max_takeoff_tas','mean_takeoff_vertical_rate'] #
cor_matrix = df[['tow'] + numeric_columns].corr() #['initial_climb_rate','avg_vertical_rate','max_vertical_rate']
# categories_columns = ['adep', 'ades', 'aircraft_type', 'wtc', 'airline', 'offblock_hour', 'arrival_hour', 'day_of_week_num', 'DosInt', 'Season', 'date', 'country_code_adep', 'country_code_ades']
categories_columns = ['adep', 'ades', 'aircraft_type', 'wtc', 'airline', 'offblock_hour', 'arrival_hour', 'day_of_week_num']
df = df.dropna(subset=numeric_columns)

# Factorize categorical columns
# df2 = pd.read_pickle('all_challangedata.pkl')
# categories_map = {}
categories_map = joblib.load('categories_map.pkl')


# categories_map = {}
# for col in categories_columns:
#     categories_map[col] = df2[col].astype('category').cat.categories
    
# joblib.dump(categories_map, 'categories_map.pkl')

def apply_categories_map(data, categories_map):
    for col, categories in categories_map.items():
        data[col] = pd.Categorical(data[col], categories=categories).codes
    return data

df = apply_categories_map(df, categories_map)

# for col in categories_columns:
#     df[col] = df[col].astype('category').cat.codes
    
# Prepare the features and target variable
X = df[numeric_columns + categories_columns]
y = df['tow']

# Standardize numerical features
scaler = StandardScaler()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric columns
X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
X_test_scaled = scaler.transform(X_test[numeric_columns])

# Create embeddings for categorical columns
input_layers = []
embedding_layers = []

max_embedding_dim = 200

# for col in categories_columns:
#     input_layer = Input(shape=(1,))
#     unique_vals = df[col].nunique()
#     output_dim = min(int(np.sqrt(unique_vals)+1), max_embedding_dim)  # Apply threshold
#     embedding_layer = Embedding(input_dim=unique_vals, output_dim=output_dim)(input_layer)
#     embedding_layer = Flatten()(embedding_layer)
#     input_layers.append(input_layer)
#     embedding_layers.append(embedding_layer)


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

model.save('tow_prediction_model_v6_with_no_adsb_3540.keras')
# Save the scaler to a file
joblib.dump(scaler, 'scaler_model_NN_v6_with_no_adsb_3540.pkl')

#%%

# Prepare features and target
X = df[numeric_columns + categories_columns]
y = df['tow']

# Standardize the numeric columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numeric_columns])

# Define a function to create the model
def create_model():
    input_layers = []
    embedding_layers = []
    max_embedding_dim = 200

    for col in categories_columns:
        input_layer = Input(shape=(1,), name=col)
        unique_vals = len(categories_map[col])
        output_dim = min(int(np.sqrt(unique_vals) + 1), max_embedding_dim)
        embedding_layer = Embedding(input_dim=unique_vals, output_dim=output_dim)(input_layer)
        embedding_layer = Flatten()(embedding_layer)
        input_layers.append(input_layer)
        embedding_layers.append(embedding_layer)

    numeric_input = Input(shape=(len(numeric_columns),))
    input_layers.append(numeric_input)
    all_layers = Concatenate()(embedding_layers + [numeric_input])

    dense_1 = Dense(256, activation='relu')(all_layers)
    dense_2 = Dense(128, activation='relu')(dense_1)
    dense_3 = Dense(64, activation='relu')(dense_2)
    dense_4 = Dense(32, activation='relu')(dense_3)
    output = Dense(1)(dense_4)

    model = Model(inputs=input_layers, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_rmse = []

for train_index, val_index in kf.split(X):
    # Split the data
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Create a new model instance for each fold
    model = create_model()
    
    # Prepare inputs for the model
    model.fit([X.iloc[train_index][col].values for col in categories_columns] + [X_train],
              y_train, 
              epochs=200, 
              batch_size=128, 
              validation_split=0.2, 
              callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    
    # Make predictions and calculate RMSE
    y_pred = model.predict([X.iloc[val_index][col].values for col in categories_columns] + [X_val])
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    fold_rmse.append(rmse)
    print(f'Fold RMSE: {rmse}')

# Report the overall RMSE across folds
print(f'Average RMSE across folds: {np.mean(fold_rmse)}')
print(f'Standard Deviation of RMSE across folds: {np.std(fold_rmse)}')
