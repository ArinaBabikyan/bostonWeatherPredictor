
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .openmeteo import fetch_openmeteo_hourly
from .inference import predict_next_24h_from_latest_window
from .preprocess import HISTORY_WINDOW

@require_http_methods(["GET"])
def index(request):
    """
    Compute a single t+24h forecast for a fixed location/timezone and render it.
    No user input.
    """
    # Hardcode Boston (adjust if needed)
    lat = 42.3601
    lon = -71.0589
    tz  = "America/New_York"

    try:
        df = fetch_openmeteo_hourly(lat, lon, hours=HISTORY_WINDOW + 48, timezone_str=tz)
        if len(df) < HISTORY_WINDOW:
            return render(request, "forecast/index.html", {
                "error": f"Not enough data: need {HISTORY_WINDOW} hours, got {len(df)}"
            })
        df['year'] = df['date'].apply(lambda x: x.year)
        df['time'] = df['date'].apply(lambda x: x.timestamp())
        df = df.drop('date', axis=1)
        latest_window = df.tail(HISTORY_WINDOW).values

        pred = predict_next_24h_from_latest_window(latest_window)

        # Set your display units here. If your model is in °C, keep "°C".
        units = "°C"

        context = {
            "latitude": lat,
            "longitude": lon,
            "timezone": tz,
            "forecast_horizon_hours": 24,
            "prediction": pred,
            "units": units,
            "error": None,
        }
        return render(request, "forecast/index.html", context)

    except Exception as e:
        return render(request, "forecast/index.html", {
            "error": str(e)
        })

