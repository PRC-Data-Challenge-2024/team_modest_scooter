# PRCDataChal
* I used Extract_ADSB.py to process and extract all feautures from each parquet file, feature contains:
   1. Min,max,avg,std tas (true air speed cal from groundspeed and win vector)
   2. Min,max,avg,std groundspeed
   3. Min,max,avg,std windspeed
   4. Min,max,avg,std altitude
   5. Min,max,avg,std track
   6. Drift Angle during the flight
   7. Min,max,avg,std teamperature,humidity
   8. Take-off duration: duration from take off to cruising period
   9. final_descent_rate: mean vertical rate of last 5 min before landing
   10. initial_climb_rate: mean vertical rate of last 5 min after takeoff
From what i read, the take aircraft 
  
