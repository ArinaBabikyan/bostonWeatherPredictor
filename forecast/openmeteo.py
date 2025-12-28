
import openmeteo_requests

import pandas as pd
import requests_cache
from datetime import datetime, timedelta
from retry_requests import retry
from .preprocess import add_time_features, ensure_feature_order

def fetch_openmeteo_hourly(lat, lon, hours=200, timezone_str="UTC") -> pd.DataFrame:
    # calls https://api.open-meteo.com/v1/forecast (no API key required)
    # parses 'hourly' -> 'temperature_2m' into a DateTime-indexed DataFrame
    # adds time features and enforces FEATURE_NAMES

    # get data on your location
    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    time_before = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%Y-%m-%d")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 42.3601,
        "longitude": -71.0588,
        "start_date": time_before,
        "end_date": time_now,
        "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "precipitation"],
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(2).ValuesAsNumpy()
    hourly_wind_speed_80m = hourly.Variables(3).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(4).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}

    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["surface_pressure"] = hourly_surface_pressure
    hourly_data["wind_speed_10m"] = hourly_wind_speed_80m
    hourly_data["precipitation"] = hourly_precipitation

    df_hourly = pd.DataFrame(data=hourly_data)
    print(df_hourly)
    return df_hourly
