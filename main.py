"""
FlightOps - Cloud-Native Data Ingestion Service (MySQL Version)
==============================================================
Fetches real-time flight states from the OpenSky Network API, correlates them
with high-altitude atmospheric weather (250hPa flight level) from Open-Meteo,
and stores the merged telemetry into a MySQL database.
"""

import os
import sys
import time
import signal
import datetime
import argparse
import requests
import mysql.connector
from mysql.connector import Error
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# -------------------------------------------------------------------
# Configuration (Environment Variables for Docker/Kubernetes)
# -------------------------------------------------------------------

OPENSKY_URL = os.getenv("OPENSKY_API_URL", "https://opensky-network.org/api/states/all")
WEATHER_URL = os.getenv("OPENMETEO_API_URL", "https://api.open-meteo.com/v1/forecast")

# MySQL Configuration variables matching docker-compose
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "rootpass")
DB_NAME = os.getenv("DB_NAME", "flightops")

DEFAULT_INTERVAL = int(os.getenv("FETCH_INTERVAL_SECONDS", "60"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))

DEFAULT_BBOX = {
    "lamin": float(os.getenv("BBOX_LAMIN", "20.0")),    # South
    "lamax": float(os.getenv("BBOX_LAMAX", "35.0")),    # North
    "lomin": float(os.getenv("BBOX_LOMIN", "20.0")),    # West
    "lomax": float(os.getenv("BBOX_LOMAX", "40.0")),    # East
}

# -------------------------------------------------------------------
# Prometheus Metrics & Graceful Shutdown Setup
# -------------------------------------------------------------------

FLIGHTS_FETCHED = Gauge("flightops_flights_active", "Number of flights fetched in the last cycle")
RECORDS_SAVED = Counter("flightops_records_saved_total", "Total flight-weather records saved to MySQL")
API_ERRORS = Counter("flightops_api_errors_total", "Total external API errors", ["source"])
CYCLE_DURATION = Histogram("flightops_cycle_duration_seconds", "Time spent processing a single ingestion cycle")

shutdown_requested = False

def handle_shutdown_signal(signum, frame):
    global shutdown_requested
    print(f"\n[WARN] Received termination signal ({signum}). Initiating graceful shutdown...")
    shutdown_requested = True

signal.signal(signal.SIGTERM, handle_shutdown_signal)
signal.signal(signal.SIGINT, handle_shutdown_signal)


# -------------------------------------------------------------------
# 1) OpenSky Flight Telemetry Ingestion
# -------------------------------------------------------------------

def fetch_flights(bbox=None):
    """
    Fetches active flights and their coordinates from the OpenSky Network API.
    """
    params = bbox if bbox else {}

    try:
        response = requests.get(OPENSKY_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        API_ERRORS.labels(source="opensky").inc()
        raise e

    raw_states = data.get("states") or []
    fetched_at = data.get("time") or int(time.time())

    flights = []
    for s in raw_states:
        icao24 = s[0]
        callsign = (s[1] or "").strip()
        origin_country = s[2]
        longitude = s[5]
        latitude = s[6]
        altitude = s[7]
        velocity = s[9]
        heading = s[10]

        if latitude is None or longitude is None:
            continue

        flights.append({
            "icao24": icao24,
            "callsign": callsign,
            "origin_country": origin_country,
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "velocity": velocity,
            "heading": heading,
            "fetched_at": fetched_at,
        })

    return flights


# -------------------------------------------------------------------
# 2) Open-Meteo High-Altitude Weather Ingestion (250hPa Flight Level)
# -------------------------------------------------------------------

def fetch_weather(lat, lon, retries=3, backoff=1.5):
    """
    Fetches high-altitude atmospheric weather (250hPa pressure level ~ 10.5 km altitude)
    using Open-Meteo API with exponential backoff retry logic.
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_250hPa,wind_speed_250hPa,wind_direction_250hPa,weather_code",
            }
            response = requests.get(WEATHER_URL, params=params, timeout=10)
            response.raise_for_status()
            current = response.json().get("current", {})

            return {
                "temperature": current.get("temperature_250hPa"),
                "windspeed": current.get("wind_speed_250hPa"),
                "winddirection": current.get("wind_direction_250hPa"),
                "weathercode": current.get("weather_code"),
            }
        except requests.RequestException as e:
            last_error = e
            API_ERRORS.labels(source="openmeteo").inc()
            if attempt < retries:
                wait = backoff ** attempt
                print(f"    [RETRY] Attempt {attempt} failed for ({lat}, {lon}). Retrying in {wait:.1f}s...")
                time.sleep(wait)
    raise last_error


def get_weather_cached(lat, lon, cache):
    """
    Applies 0.5-degree spatial grid batching (~50km) to cache weather requests
    and prevent API rate-limiting.
    """
    key = (round(lat * 2) / 2, round(lon * 2) / 2)
    if key not in cache:
        try:
            cache[key] = fetch_weather(*key)
        except requests.RequestException as e:
            print(f"  [ERROR] Failed to fetch weather for grid {key}: {e}")
            cache[key] = {}
        time.sleep(0.3)
    return cache[key]


# -------------------------------------------------------------------
# 3) MySQL Persistence Layer
# -------------------------------------------------------------------

def get_db_connection():
    """
    Establishes and returns a connection to the MySQL database,
    creating the database and required tables if they don't exist.
    """
    try:
        # Connect first without database name to ensure DB exists
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME};")
        cursor.close()
        conn.close()

        # Connect to the specific database
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flight_weather (
                id INT AUTO_INCREMENT PRIMARY KEY,
                icao24 VARCHAR(255),
                callsign VARCHAR(255),
                origin_country VARCHAR(255),
                latitude FLOAT,
                longitude FLOAT,
                altitude FLOAT,
                velocity FLOAT,
                heading FLOAT,
                temperature FLOAT,
                windspeed FLOAT,
                winddirection FLOAT,
                weathercode INT,
                fetched_at BIGINT,
                inserted_at VARCHAR(255)
            )
        """)
        conn.commit()
        cursor.close()
        return conn
    except Error as e:
        print(f"[ERROR] MySQL connection error: {e}")
        raise e


def save_records(conn, records):
    cursor = conn.cursor()
    query = """
        INSERT INTO flight_weather (
            icao24, callsign, origin_country, latitude, longitude,
            altitude, velocity, heading, temperature, windspeed,
            winddirection, weathercode, fetched_at, inserted_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        cursor.executemany(query, records)
        conn.commit()
    except Error as e:
        print(f"[ERROR] Failed to insert records into MySQL: {e}")
        conn.rollback()
    finally:
        cursor.close()


# -------------------------------------------------------------------
# 4) Main Ingestion Cycle
# -------------------------------------------------------------------

@CYCLE_DURATION.time()
def run_once(conn, bbox, limit=None):
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now_str}] Fetching live flight telemetry...")
    
    try:
        flights = fetch_flights(bbox)
    except Exception as e:
        print(f"  [ERROR] Failed to fetch flights: {e}")
        return

    print(f"  -> Discovered {len(flights)} active flights.")
    FLIGHTS_FETCHED.set(len(flights))

    if limit:
        flights = flights[:limit]

    weather_cache = {}
    records = []
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for f in flights:
        if shutdown_requested:
            print("  [WARN] Aborting weather correlation loop due to pending shutdown signal.")
            break
        weather = get_weather_cached(f["latitude"], f["longitude"], weather_cache)
        records.append((
            f["icao24"],
            f["callsign"],
            f["origin_country"],
            f["latitude"],
            f["longitude"],
            f["altitude"],
            f["velocity"],
            f["heading"],
            weather.get("temperature"),
            weather.get("windspeed"),
            weather.get("winddirection"),
            weather.get("weathercode"),
            f["fetched_at"],
            now_iso,
        ))

    if records:
        try:
            save_records(conn, records)
        except mysql.connector.OperationalError:
            print("  [WARN] MySQL connection lost. Reconnecting...")
            conn.reconnect(attempts=3, delay=2)
            save_records(conn, records)

        RECORDS_SAVED.inc(len(records))
        print(f"  -> Successfully committed {len(records)} records to MySQL ( across {len(weather_cache)} weather grids ).")


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FlightOps Data Ingestion Service")
    parser.add_argument("--once", action="store_true", help="Run a single ingestion cycle and exit")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="Polling interval in seconds (default: 60)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of flights to process per cycle")
    parser.add_argument("--world", action="store_true", help="Fetch global flights instead of the default bounding box")
    args = parser.parse_args()

    bbox = None if args.world else DEFAULT_BBOX
    
    print(f"[INFO] Connecting to MySQL at {DB_HOST}:{DB_PORT}...")
    conn = get_db_connection()

    if args.once:
        run_once(conn, bbox, limit=args.limit)
        conn.close()
        return

    start_http_server(METRICS_PORT)
    print(f"[INFO] Prometheus metrics and health endpoint exposed on port {METRICS_PORT}")
    print(f"[INFO] Connected successfully to MySQL database: {DB_NAME}")
    print(f"[INFO] Ingestion service running continuously (interval: {args.interval}s). Press Ctrl+C to stop.")

    while not shutdown_requested:
        try:
            run_once(conn, bbox, limit=args.limit)
        except requests.RequestException as e:
            print(f"  [ERROR] Network error during ingestion cycle: {e}")

        for _ in range(args.interval):
            if shutdown_requested:
                break
            time.sleep(1)

    if conn.is_connected():
        conn.close()
    print("[INFO] Database connection closed cleanly. Shutdown complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
