import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
# Function to calculate True Airspeed (TAS)
def calculate_tas(groundspeed, u_component_of_wind, v_component_of_wind):
    wind_speed = np.sqrt(u_component_of_wind**2 + v_component_of_wind**2)
    tas = np.sqrt(groundspeed**2 + wind_speed**2)
    return tas

# Function to calculate drift angle
def calculate_drift_angle(groundspeed, u_component_of_wind, v_component_of_wind):
    tas = calculate_tas(groundspeed, u_component_of_wind, v_component_of_wind)
    wind_speed = np.sqrt(u_component_of_wind**2 + v_component_of_wind**2)
    drift_angle = np.degrees(np.arcsin(wind_speed / tas))
    return drift_angle

# Function to extract features from the filtered ADS-B data
def extract_flight_features(data, actual_offblock_time, taxiout_time, parquet):
    features = {}
    
    # Filter out records with no groundspeed
    data = data.dropna(subset=['groundspeed'])
    data = data.sort_values(by='timestamp')

    # Calculate TAS and drift angle
    data['TAS'] = calculate_tas(data['groundspeed'], data['u_component_of_wind'], data['v_component_of_wind'])
    data['drift_angle'] = calculate_drift_angle(data['groundspeed'], data['u_component_of_wind'], data['v_component_of_wind'])
    
    features['parquet'] = parquet
    
    # Mean, min, max, std tas
    features['avg_tas'] = data['TAS'].mean()
    features['min_tas'] = data['TAS'].min()
    features['max_tas'] = data['TAS'].max()
    features['std_tas'] = data['TAS'].std()

    # Mean, min, max, std track
    features['avg_track'] = data['track'].mean()
    features['min_track'] = data['track'].min()
    features['max_track'] = data['track'].max()
    features['std_track'] = data['track'].std()

    # Groundspeed features
    features['avg_groundspeed'] = data['groundspeed'].mean()
    features['max_groundspeed'] = data['groundspeed'].max()
    features['min_groundspeed'] = data['groundspeed'].min()
    features['std_groundspeed'] = data['groundspeed'].std()
    
    # Altitude features
    features['avg_altitude'] = data['altitude'].mean()
    features['max_altitude'] = data['altitude'].max()
    features['min_altitude'] = data['altitude'].min()
    features['std_altitude'] = data['altitude'].std()
    
    # Wind speed features
    data['wind_speed'] = np.sqrt(data['u_component_of_wind']**2 + data['v_component_of_wind']**2)
    features['avg_wind_speed'] = data['wind_speed'].mean()
    features['max_wind_speed'] = data['wind_speed'].max()
    features['min_wind_speed'] = data['wind_speed'].min()
    features['std_wind_speed'] = data['wind_speed'].std()
    
    # Temperature and humidity
    features['avg_temperature'] = data['temperature'].mean()
    features['avg_specific_humidity'] = data['specific_humidity'].mean()
        
    # Initial climb rate and final descent rate
    try:
        features['initial_climb_rate'] = data[data['timestamp'] <= data['timestamp'].iloc[0] + pd.Timedelta(minutes=5)]['vertical_rate'].mean()
    except:
        features['initial_climb_rate'] = np.nan
    try:
        features['final_descent_rate'] = data[data['timestamp'] >= data['timestamp'].iloc[-1] - pd.Timedelta(minutes=5)]['vertical_rate'].mean()
    except:
        features['final_descent_rate'] = np.nan
    
    takeoff_start = actual_offblock_time + taxiout_time
    features['take-off-time'] = takeoff_start
    mean_altitude = data['altitude'].mean()
    altitude_threshold = 0.8 * mean_altitude

    # Calculate climb_finished_timestamp
    try:
        climb_periods = data[(data['vertical_rate'].between(-150, 150)) & 
                             (data['timestamp'] <= takeoff_start + pd.Timedelta(minutes=40)) & (data['timestamp'] >= takeoff_start + pd.Timedelta(minutes=5))]
        
        climb_finished_periods = climb_periods.groupby((climb_periods['vertical_rate'].diff().abs() > 300).cumsum()).filter(lambda x: len(x) > 30)
        climb_finished_periods = climb_finished_periods[climb_finished_periods['altitude'] >= altitude_threshold]
        climb_finished_periods = climb_finished_periods.sort_values(by='timestamp')
        # print(climb_finished_periods)
        
        if not climb_finished_periods.empty:
            features['climb_finished_timestamp'] = climb_finished_periods['timestamp'].iloc[0]
            # print(climb_finished_periods['timestamp'].iloc[0])
        else:
            features['climb_finished_timestamp'] = np.nan
    except:
        features['climb_finished_timestamp'] = np.nan
    
    try:
        # Calculate descent_started_timestamp
        descent_periods = data[(data['vertical_rate'].between(-150, 150)) & 
                               (data['timestamp'] >= data['timestamp'].iloc[-1] - pd.Timedelta(minutes=40))]
        
        descent_started_periods = descent_periods.groupby((descent_periods['vertical_rate'].diff().abs() > 300).cumsum()).filter(lambda x: len(x) >= 30)
        descent_started_periods = descent_started_periods[descent_started_periods['altitude'] >= altitude_threshold]
        descent_started_periods = descent_started_periods.sort_values(by='timestamp')
        
        if not descent_started_periods.empty:
            features['descent_started_timestamp'] = descent_started_periods['timestamp'].iloc[-1]
        else:
            features['descent_started_timestamp'] = np.nan
    except:
        features['descent_started_timestamp'] = np.nan
    
    # Filter out records with wrong vertical_rate
    data = data[(data['vertical_rate'] <= 13000) & (data['vertical_rate'] >= -13000)]
    
    # Vertical rate features
    features['avg_vertical_rate'] = data['vertical_rate'].mean()
    features['max_vertical_rate'] = data['vertical_rate'].max()
    features['min_vertical_rate'] = data['vertical_rate'].min()
    features['std_vertical_rate'] = data['vertical_rate'].std()
    
    # Take-off features
    if not pd.isna(features['climb_finished_timestamp']):
        takeoff_duration = (features['climb_finished_timestamp'] - takeoff_start).total_seconds()
        features['takeoff_duration'] = takeoff_duration
        takeoff_data = data[data['timestamp'] <= features['climb_finished_timestamp']]
        features['mean_takeoff_groundspeed'] = takeoff_data['groundspeed'].mean()
        features['min_takeoff_groundspeed'] = takeoff_data['groundspeed'].min()
        features['max_takeoff_groundspeed'] = takeoff_data['groundspeed'].max()
        features['mean_takeoff_tas'] = takeoff_data['TAS'].mean()
        features['min_takeoff_tas'] = takeoff_data['TAS'].min()
        features['max_takeoff_tas'] = takeoff_data['TAS'].max()
        features['mean_takeoff_vertical_rate'] = takeoff_data['vertical_rate'].mean()
        features['min_takeoff_vertical_rate'] = takeoff_data['vertical_rate'].min()
        features['max_takeoff_vertical_rate'] = takeoff_data['vertical_rate'].max()
        features['mean_takeoff_temperature'] = takeoff_data['temperature'].mean()
        features['mean_takeoff_specific_humidity'] = takeoff_data['specific_humidity'].mean()
        features['mean_takeoff_wind_speed'] = takeoff_data['wind_speed'].mean()
        
        # Wind components during take-off
        features['u_windspeed_take_off'] = takeoff_data['u_component_of_wind'].mean()
        features['v_windspeed_take_off'] = takeoff_data['v_component_of_wind'].mean()
    else:
        features['takeoff_duration'] = np.nan
        features['mean_takeoff_groundspeed'] = np.nan
        features['min_takeoff_groundspeed'] = np.nan
        features['max_takeoff_groundspeed'] = np.nan
        features['mean_takeoff_tas'] = np.nan
        features['min_takeoff_tas'] = np.nan
        features['max_takeoff_tas'] = np.nan
        features['mean_takeoff_vertical_rate'] = np.nan
        features['min_takeoff_vertical_rate'] = np.nan
        features['max_takeoff_vertical_rate'] = np.nan
        features['mean_takeoff_temperature'] = np.nan
        features['mean_takeoff_specific_humidity'] = np.nan
        features['mean_takeoff_wind_speed'] = np.nan
        features['u_windspeed_take_off'] = np.nan
        features['v_windspeed_take_off'] = np.nan
    
    return features

# Load existing alladsbdata DataFrame
alladsbdata = pd.read_pickle('F:/OPENSKY/alladsbdata_with_old_columns.pkl')
alladsbdata = alladsbdata[~pd.isnull(alladsbdata.taxiout_time)]
alladsbdata['taxiout_time'] = alladsbdata['taxiout_time'].astype(int)

# Create an empty DataFrame to store all the extracted features
features_df = pd.DataFrame()

# Iterate through each day from '2022-01-01' to '2022-12-31'
dates = pd.date_range(start='2022-01-01', end='2022-12-31')

for date in tqdm(dates, desc="Processing dates"):
    date_str = date.strftime('%Y-%m-%d')
    adsb_data = pd.read_parquet(f'C:/Users/admin/{date_str}.parquet')
    if not adsb_data.empty:
        try:        
            for flight_id in adsb_data['flight_id'].unique():
                flight_data = adsb_data[adsb_data['flight_id'] == flight_id]
                daily_flights = alladsbdata[alladsbdata.flight_id == flight_id]
                if daily_flights.shape[0] > 0:
                    flight_data['timestamp'] = flight_data['timestamp'].dt.tz_localize(None)
                    flight_info = daily_flights.iloc[0]
                    actual_offblock_time = flight_info['actual_offblock_time']
                    taxiout_time = pd.Timedelta(minutes=flight_info['taxiout_time'])
                    arrival_time = flight_info['arrival_time']
                    flight_period_start = actual_offblock_time + taxiout_time - pd.Timedelta(seconds=60)
                    flight_period_end = arrival_time
                    filtered_data = flight_data[(flight_data['timestamp'] >= flight_period_start) & 
                                                (flight_data['timestamp'] <= flight_period_end)]
                
                    features = extract_flight_features(filtered_data, actual_offblock_time, taxiout_time, date_str)
                    features['flight_id'] = flight_id
                    features_df = pd.concat([features_df, pd.DataFrame([features])], ignore_index=True)
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
        print(f"Finished updating from '{date_str}.parquet'")
        features_df.to_pickle('F:/OPENSKY/alladsbdata_with_features_new.pkl')
    else:
        print(f"Error processing {date_str}:")

# Merge the extracted features with the original alladsbdata
# alladsbdata = pd.merge(alladsbdata, features_df, on='flight_id', how='left')
# alladsbdata.to_pickle('F:/OPENSKY/alladsbdata_with_features_final_new.pkl')
