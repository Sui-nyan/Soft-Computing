import json
import pandas as pd
import numpy as np

# Step 1 & 2：Load JSON and extract hourly weather data.
def load_json(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_hourly_observations(data: dict) -> pd.DataFrame:
    """Extract hourly observations for all stations from the JSON file."""
    locations = (
        data["cwaopendata"]["resources"]["resource"]["data"]
        ["surfaceObs"]["location"]
    )

    rows = []
    for location in locations:
        station = location["station"]
        station_id = station.get("StationID")
        station_name = station.get("StationName")
        station_name_en = station.get("StationNameEN")

        obs_times = location["stationObsTimes"]["stationObsTime"]
        for obs in obs_times:
            weather = obs.get("weatherElements", {})
            rows.append({
                "StationID":        station_id,
                "StationName":      station_name,
                "StationNameEN":    station_name_en,
                "DataTime":         obs.get("DataTime"),
                "AirPressure":      weather.get("AirPressure"),
                "AirTemperature":   weather.get("AirTemperature"),
                "RelativeHumidity": weather.get("RelativeHumidity"),
                "WindSpeed":        weather.get("WindSpeed"),
                "WindDirection":    weather.get("WindDirection"),
                "Precipitation":    weather.get("Precipitation"),
                "SunshineDuration": weather.get("SunshineDuration"),
            })

    df = pd.DataFrame(rows)

    numeric_cols = [
        "AirPressure", "AirTemperature", "RelativeHumidity",
        "WindSpeed", "SunshineDuration",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["DataTime"] = df["DataTime"].astype(str)
    return df


# Step 3: Clean data and filter for the Taipei station

def process_taipei(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the complete DataFrame for the Taipei station,
    format the time column, and fill missing SunshineDuration values during nighttime.
    """
    # format the time column
    df["DataTime"] = pd.to_datetime(df["DataTime"])
    df["Time"] = df["DataTime"].dt.strftime("%Y-%m-%d %H:%M").str.replace(r'^0', '', regex=True)

    # Filter the complete DataFrame for the Taipei station
    df2 = df.loc[
        df["StationName"] == "臺北",
        ["StationName", "Time", "AirTemperature", "RelativeHumidity", "SunshineDuration"],
    ].copy()

    # fill missing SunshineDuration values during nighttime.
    df2["Time"] = pd.to_datetime(df2["Time"])
    hours = df2["Time"].dt.hour
    night_mask = ((hours >= 20) | (hours <= 5)) & df2["SunshineDuration"].isna()
    df2.loc[night_mask, "SunshineDuration"] = 0.0

    return df2


def export_csv_files(df2: pd.DataFrame):
    """Export summerdata.csv."""

    # summerdata: 4 summer weeks (1 week/month from Jun to Sep)
    cond_s1 = (df2["Time"] >= "2025-06-16") & (df2["Time"] < "2025-06-23")
    cond_s2 = (df2["Time"] >= "2025-07-15") & (df2["Time"] < "2025-07-22")
    cond_s3 = (df2["Time"] >= "2025-08-10") & (df2["Time"] < "2025-08-17")
    cond_s4 = (df2["Time"] >= "2025-09-08") & (df2["Time"] < "2025-09-15")
    summer = df2[cond_s1 | cond_s2 | cond_s3 | cond_s4]
    summer.to_csv("summerdata.csv", index=False, encoding="utf-8-sig")

    print(f"Number of records in summerdata.csv：{len(summer)}")


def run_pipeline(json_path: str = "C-B0024-002.json"):
    """
    Run the complete data processing pipeline:
      JSON → hourly DataFrame → filter data for 臺北 station  → Export to CSV
    """
    print(f"Step 1:Reading JSON: {json_path}")
    data = load_json(json_path)

    print("Step 2:Parsing hourly observation data...")
    hourly_df = build_hourly_observations(data)

    print("Step 3:Processing Taipei station data...")
    taipei_df = process_taipei(hourly_df)

    print("Step 4:Exporting CSV files...")
    export_csv_files(taipei_df)

    print("Pipeline completed!")

# Step 5: Function for algorithm.py to call
def get_temperature_data(csv_path: str) -> np.ndarray:
    """
    Read the AirTemperature column from a CSV file and return a numpy array.

    Example (algorithm.ipynb):
        from weather_pipeline import run_pipeline, get_temperature_data

        run_pipeline("C-B0024-002.json")
        outdoor_temps = get_temperature_data("two_weeks.csv")[:24]
    """
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(csv_path, encoding="big5")

    if "AirTemperature" not in df.columns:
        raise ValueError("CSV file does not contain 'AirTemperature' column.")

    return df["AirTemperature"].to_numpy(dtype=float)



if __name__ == "__main__":
    run_pipeline("C-B0024-002.json")
