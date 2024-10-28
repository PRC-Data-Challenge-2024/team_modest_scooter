# PRCDataChal
* I used Extract_ADSB.py to process and extract all feautures for each flight from each parquet file, feature contains:
   1. Min,max,avg,std tas (true air speed cal from groundspeed and win vector)
   2. Min,max,avg,std groundspeed
   3. Min,max,avg,std windspeed
   4. Min,max,avg,std altitude
   5. Min,max,avg,std track
   6. Drift Angle during the flight
   7. Min,max,avg,std teamperature,humidity
   8. Take-off duration: duration from take off to cruising period (using actual off-block time + taxi time as take-off time)
   9. final_descent_rate: mean vertical rate of last 5 min before landing
   10. initial_climb_rate: mean vertical rate of last 5 min after takeoff
From what i read, the aircraft weight affected the take-off the most, heavier aircraft tend to climber slower (slower vertical rate), flight faster (higher tas), take longer to reach cruising speed
So i add some features for take-off duration
  11. Min,max,avg,std tas during take-off
  12. Min,max,avg,std vertical rate during take-off
  13. Min,max,avg,std windspeed during take-off
  14. Min,max,avg,std teamperature,humidity during take-off
  15. Ground speed, altitude, vertical at the moment of take-off
* For main Final_PRCCha.py:
  After extract all the features needed, i store in a dataframe call alladsbdata_with_new_features and merge with the processed training data.
For the training dataset, using several machine learning model (random forest, Neural Network, SVM ....) with different combination of catergory features and got the best result as the following combination:
   categories_columns = ['adep', 'ades', 'aircraft_type', 'wtc', 'airline', 'offblock_hour', 'arrival_hour', 'day_of_week_num', 'country_code_adep', 'country_code_ades', 'month']
   numeric_columns = ['flight_duration', 'taxiout_time', 'flown_distance',
          'avg_tas', 'min_tas', 'max_tas', 'avg_altitude', 'avg_wind_speed', 'max_wind_speed', 'std_wind_speed', 'avg_temperature',
          'avg_specific_humidity', 'std_vertical_rate_x', 'takeoff_duration',
          'mean_takeoff_tas', 'max_takeoff_tas','mean_takeoff_vertical_rate',]
  For the records with no adsb data/no take-off data i trained a different model with different set of numeric/category columns and combine the results.
  * Some observations:
       1. Using date both as category and numeric features did not yeild good results, although its clear that certain days got higher mean tow than others day, especically before big hoildays: For example there is a surge at day 30 April, 30Sep...
       2. Some combination of aircraft_type, airline, adep-ades have 0 std (same value of tow), some airlines had low set of lit values for some specific routes, using classification on these airline would gave beter result, i also think that train different model for different airline would be better but did not have the time to try.
       3. 
 * What i think could improve the model:
      1. Schedule flights data all-flights of year 2022, with flights is canceled, which flight is need to changed aircraft during operation, which aircraft is using for each flights.... Unforternately i did not have access to any of these data.
      2. 
