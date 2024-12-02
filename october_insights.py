import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from geopy.distance import geodesic
import pydeck as pdk
from typing import Tuple, List, Dict
import numpy as np


# Type aliases
StationStats = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
TripMetrics = Tuple[float, float, float, float, float]

@st.cache_data
def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess the Bay Wheels trip data."""
    df = pd.read_csv(file_path)
    
    # Clean coordinates
    df = df.dropna(subset=["start_lat", "start_lng", "end_lat", "end_lng"])
    df = df[
        (df["start_lat"].between(-90, 90)) & 
        (df["end_lat"].between(-90, 90)) &
        (df["start_lng"].between(-180, 180)) &
        (df["end_lng"].between(-180, 180))
    ]
    
    # Add temporal features
    df["trip_duration_minutes"] = (
        pd.to_datetime(df["ended_at"]) - pd.to_datetime(df["started_at"])
    ).dt.total_seconds() / 60
    
    timestamp_cols = pd.to_datetime(df["started_at"])
    df["date"] = timestamp_cols.dt.date
    df["hour"] = timestamp_cols.dt.hour
    df["day_of_week"] = timestamp_cols.dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18])
    
    # Calculate distances
    df["distance_miles"] = df.apply(
        lambda row: geodesic(
            (row["start_lat"], row["start_lng"]),
            (row["end_lat"], row["end_lng"])
        ).miles,
        axis=1
    )
    
    return df

def analyze_station_patterns(df: pd.DataFrame) -> StationStats:
    """Analyze station usage patterns."""
    # Top stations
    start_stations = (
        df.groupby("start_station_name")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="trip_count")
    )
    
    # Station imbalances
    station_flows = pd.DataFrame({
        "starts": df.groupby("start_station_name").size(),
        "ends": df.groupby("end_station_name").size()
    }).fillna(0)
    
    station_flows["imbalance"] = station_flows["ends"] - station_flows["starts"]
    station_flows = station_flows.sort_values("imbalance", ascending=False)
    
    # Rush hour patterns
    rush_hour_stations = (
        df[df["is_rush_hour"]]
        .groupby("start_station_name")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="trip_count")
    )
    
    return start_stations, station_flows, rush_hour_stations

def calculate_trip_metrics(df: pd.DataFrame) -> TripMetrics:
    """Calculate key trip metrics."""
    return (
        len(df),  # total trips
        df["trip_duration_minutes"].mean(),  # avg duration
        df["trip_duration_minutes"].median(),  # median duration
        df["distance_miles"].mean(),  # avg distance
        df["distance_miles"].max()  # max distance
    )

def create_usage_visualizations(df: pd.DataFrame) -> Dict:
    """Create main usage visualizations."""
    # Hourly patterns
    hourly = df.groupby("hour").size().reset_index(name="trips")
    hourly_chart = alt.Chart(hourly).mark_line().encode(
        x=alt.X("hour:Q", title="Hour of Day"),
        y=alt.Y("trips:Q", title="Number of Trips"),
        tooltip=["hour:Q", "trips:Q"]
    ).properties(title="Trips by Hour")
    
    # User type distribution
    user_dist = df["member_casual"].value_counts(normalize=True).reset_index()
    user_dist.columns = ["type", "percentage"]
    user_chart = px.pie(
        user_dist,
        values="percentage",
        names="type",
        title="User Type Distribution",
        hole=0.4
    )
    
    # Weekend vs weekday patterns
    daily = df.groupby(["day_of_week", "is_weekend", "member_casual"]).size().reset_index(name="trips")
    daily_chart = px.bar(
        daily,
        x="day_of_week",
        y="trips",
        color="member_casual",
        title="Daily Usage Patterns",
        barmode="group"
    )
    
    return {
        "hourly": hourly_chart,
        "users": user_chart,
        "daily": daily_chart
    }

def create_maps(df: pd.DataFrame) -> Dict:
    """Create interactive maps for visualizing bike share patterns."""
    
    # 1. Station Activity Heatmap
    station_activity = pd.DataFrame({
        'latitude': df['start_lat'],
        'longitude': df['start_lng'],
        'weight': 1
    })
    
    heatmap_layer = pdk.Layer(
        'HeatmapLayer',
        data=station_activity,
        opacity=0.6,
        get_position=['longitude', 'latitude'],
        get_weight='weight',
        threshold=0.05,
        radiusPixels=50,
    )
    
    # 2. Trip Flow Lines
    flows = (df.groupby(['start_station_name', 'end_station_name', 'start_lat', 
                        'start_lng', 'end_lat', 'end_lng'])
            .size()
            .reset_index(name='trip_count')
            .sort_values('trip_count', ascending=False)
            .head(100))  # Top 100 routes
            
    flow_layer = pdk.Layer(
        'ArcLayer',
        data=flows,
        get_source_position=['start_lng', 'start_lat'],
        get_target_position=['end_lng', 'end_lat'],
        get_width='trip_count',
        get_source_color=[255, 165, 0, 80],  # Orange start
        get_target_color=[255, 0, 0, 80],    # Red end
        pickable=True,
        auto_highlight=True,
    )
    
    # 3. Station Clustering
    stations = (df.groupby(['start_station_name', 'start_lat', 'start_lng'])
                .size()
                .reset_index(name='station_trips'))
                
    stations['size'] = np.log1p(stations['station_trips']) * 100  # Scale marker size
    
    cluster_layer = pdk.Layer(
        'ScatterplotLayer',
        data=stations,
        get_position=['start_lng', 'start_lat'],
        get_radius='size',
        get_fill_color=[0, 0, 255, 140],
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
    )
    
    # Initial view state centered on Bay Area
    view_state = pdk.ViewState(
        latitude=df['start_lat'].mean(),
        longitude=df['start_lng'].mean(),
        zoom=11,
        pitch=45,
        bearing=0
    )
    
    # Create map styles
    heatmap = pdk.Deck(
        layers=[heatmap_layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v10',
        tooltip={"text": "Concentration of trip starts"}
    )
    
    flow_map = pdk.Deck(
        layers=[flow_layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/light-v10',
        tooltip={"text": "{start_station_name} → {end_station_name}\nTrips: {trip_count}"}
    )
    
    station_map = pdk.Deck(
        layers=[cluster_layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/streets-v11',
        tooltip={"text": "{start_station_name}\nTrips: {station_trips}"}
    )
    
    return {
        "heatmap": heatmap,
        "flows": flow_map,
        "stations": station_map
    }

def main():
    st.title("Bay Wheels Usage Analysis Dashboard")
    
    # Load data
    df = load_and_preprocess_data("bay_wheels_october.csv")
    
    # Calculate metrics
    total_trips, avg_duration, median_duration, avg_distance, max_distance = calculate_trip_metrics(df)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Trips", f"{total_trips:,}")
        st.metric("Average Duration", f"{avg_duration:.1f} min")
    with col2:
        st.metric("Median Duration", f"{median_duration:.1f} min")
        st.metric("Average Distance", f"{avg_distance:.1f} mi")
    with col3:
        st.metric("Longest Trip", f"{max_distance:.1f} mi")
    
    # Map Visualizations
    st.header("Geographic Insights")
    maps = create_maps(df)
    
    # Tabs for different map views
    tab1, tab2, tab3 = st.tabs(["Activity Heatmap", "Trip Flows", "Station Clusters"])
    
    with tab1:
        st.subheader("Trip Activity Heatmap")
        st.write("Areas with higher trip concentration appear brighter")
        st.pydeck_chart(maps["heatmap"])
        
    with tab2:
        st.subheader("Popular Trip Routes")
        st.write("Lines show the most common trip paths, with thickness indicating frequency")
        st.pydeck_chart(maps["flows"])
        
    with tab3:
        st.subheader("Station Usage Clusters")
        st.write("Circles show stations, with size indicating number of trips")
        st.pydeck_chart(maps["stations"])
    
    # Rest of the visualizations
    st.header("Usage Patterns")
    charts = create_usage_visualizations(df)
    
    st.subheader("Hourly Usage Patterns")
    st.altair_chart(charts["hourly"], use_container_width=True)
    
    st.subheader("User Distribution")
    st.plotly_chart(charts["users"], use_container_width=True)
    
    st.subheader("Daily Patterns")
    st.plotly_chart(charts["daily"], use_container_width=True)
    
    # Station analysis
    start_stations, station_flows, rush_hour_stations = analyze_station_patterns(df)
    
    # Display station insights
    st.header("Station Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Start Stations")
        st.dataframe(start_stations)
    with col2:
        st.subheader("Most Imbalanced Stations")
        st.dataframe(station_flows.head(10))

if __name__ == "__main__":
    main()
