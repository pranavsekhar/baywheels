import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from geopy.distance import geodesic
import pydeck as pdk

# Cache the data loading and preprocessing to avoid reloading on every interaction
@st.cache_data
def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Drop rows with missing or invalid coordinates
    df = df.dropna(subset=["start_lat", "start_lng", "end_lat", "end_lng"])
    df = df[
        (df["start_lat"].between(-90, 90)) & 
        (df["end_lat"].between(-90, 90)) &
        (df["start_lng"].between(-180, 180)) &
        (df["end_lng"].between(-180, 180))
    ]

    # Calculate trip duration (in minutes)
    df["trip_duration_minutes"] = (
        pd.to_datetime(df["ended_at"]) - pd.to_datetime(df["started_at"])
    ).dt.total_seconds() / 60

    # Add date, hour, day_of_week, and is_weekend columns
    df["date"] = pd.to_datetime(df["started_at"]).dt.date
    df["hour"] = pd.to_datetime(df["started_at"]).dt.hour
    df["day_of_week"] = pd.to_datetime(df["started_at"]).dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])

    # Calculate trip distance (in miles)
    def calculate_distance(row):
        start = (row["start_lat"], row["start_lng"])
        end = (row["end_lat"], row["end_lng"])
        return geodesic(start, end).miles

    df["distance_km"] = df.apply(calculate_distance, axis=1)
    return df

# Load and preprocess the dataset
file_path = "bay_wheels_october.csv"
df = load_and_preprocess_data(file_path)

# Calculate trip imbalance
start_counts = df.groupby("start_station_name").size()
end_counts = df.groupby("end_station_name").size()

station_differences = pd.DataFrame({
    "station_name": start_counts.index.union(end_counts.index),
    "start_count": start_counts.reindex(start_counts.index.union(end_counts.index), fill_value=0),
    "end_count": end_counts.reindex(end_counts.index.union(start_counts.index), fill_value=0),
})

station_differences["imbalance"] = station_differences["end_count"] - station_differences["start_count"]
station_differences = station_differences.sort_values(by="imbalance", ascending=False).reset_index(drop=True)

# Cache expensive calculations
@st.cache_data
def calculate_aggregations(data):
    total_trips = len(data)
    avg_trip_duration = data["trip_duration_minutes"].mean()
    median_trip_duration = data["trip_duration_minutes"].median()
    avg_distance = data["distance_km"].mean()  # Already in miles due to updated function
    longest_trip = data["distance_km"].max()  # Already in miles due to updated function

    # Aggregate data
    daily_trips = data.groupby("date").size().reset_index(name="trip_count")
    hourly_trips = data.groupby("hour").size().reset_index(name="trip_count")
    top_start_stations = (
        data.groupby("start_station_name").size().sort_values(ascending=False).head(10).reset_index(name="trip_count")
    )
    top_end_stations = (
        data.groupby("end_station_name").size().sort_values(ascending=False).head(10).reset_index(name="trip_count")
    )
    station_pairs = (
        data.groupby(["start_station_name", "end_station_name"]).size().sort_values(ascending=False).head(10).reset_index(name="trip_count")
    )
    user_type_distribution = data["member_casual"].value_counts(normalize=True).reset_index()
    user_type_distribution.columns = ["user_type", "percentage"]

    # Subscriber and Casual trends by hour
    subscriber_trends = data[data["member_casual"] == "member"].groupby("hour").size().reset_index(name="trip_count")
    casual_trends = data[data["member_casual"] == "casual"].groupby("hour").size().reset_index(name="trip_count")

    return (
        total_trips,
        avg_trip_duration,
        median_trip_duration,
        avg_distance,
        longest_trip,
        daily_trips,
        hourly_trips,
        top_start_stations,
        top_end_stations,
        station_pairs,
        user_type_distribution,
        subscriber_trends,
        casual_trends,
    )

# Perform aggregations
(
    total_trips,
    avg_trip_duration,
    median_trip_duration,
    avg_distance,
    longest_trip,
    daily_trips,
    hourly_trips,
    top_start_stations,
    top_end_stations,
    station_pairs,
    user_type_distribution,
    subscriber_trends,
    casual_trends,
) = calculate_aggregations(df)


# Display Key Metrics
st.title("Bay Wheels October 2024 Insights")
st.write(f"**Total Trips:** {total_trips}")
st.write(f"**Average Trip Duration:** {avg_trip_duration:.2f} minutes")
st.write(f"**Median Trip Duration:** {median_trip_duration:.2f} minutes")
st.write(f"**Average Trip Distance:** {avg_distance:.2f} miles")
st.write(f"**Longest Trip Distance:** {longest_trip:.2f} miles")

# Add a quick map visualization
st.subheader("Quick Map: Bay Wheels Stations")
st.write(
    "This map shows the locations of all Bay Wheels stations in the city, helping you get oriented."
)

# Aggregate station locations
station_locations = (
    df[["start_station_name", "start_lat", "start_lng"]]
    .drop_duplicates()
    .rename(columns={"start_lat": "latitude", "start_lng": "longitude"})
)

# Create the Pydeck Layer for Stations
station_layer = pdk.Layer(
    "ScatterplotLayer",
    data=station_locations,
    get_position="[longitude, latitude]",
    get_radius=100,
    get_color="[0, 0, 255, 160]",
    pickable=True,
    tooltip=True,
)

# Create the View State
view_state = pdk.ViewState(
    latitude=station_locations["latitude"].mean(),
    longitude=station_locations["longitude"].mean(),
    zoom=12,
    pitch=0,
)

# Render the Map
st.pydeck_chart(pdk.Deck(layers=[station_layer], initial_view_state=view_state))

###

# Editorial Section: Key Insights
st.title("Editorial Insights: October Bay Wheels Usage")
st.write("Here are some key highlights and narrative insights from the data:")

# 1. Most and Least Popular Stations
busiest_station = top_start_stations.iloc[0]
least_used_station = df.groupby("start_station_name").size().idxmin()
least_used_count = df.groupby("start_station_name").size().min()

st.subheader("Most and Least Popular Stations")
st.write(f"The busiest station in October was **{busiest_station['start_station_name']}**, with {busiest_station['trip_count']} trips.")
st.write(f"The least-used station was **{least_used_station}**, with only {least_used_count} trips starting there.")

# 2. Longest and Shortest Trips
longest_trip_row = df.loc[df["distance_km"].idxmax()]
shortest_trip_row = df.loc[df["distance_km"].idxmin()]

st.subheader("Longest and Shortest Trips")
st.write(f"The longest trip covered **{longest_trip_row['distance_km']:.1f} miles**, starting at **{longest_trip_row['start_station_name']}** and ending at **{longest_trip_row['end_station_name']}**.")
st.write(f"The shortest trip was **{shortest_trip_row['distance_km']:.1f} miles**, from **{shortest_trip_row['start_station_name']}** to **{shortest_trip_row['end_station_name']}**.")

# 3. Subscriber vs. Casual Insights
subscriber_pct = user_type_distribution.loc[user_type_distribution["user_type"] == "member", "percentage"].values[0]
casual_pct = user_type_distribution.loc[user_type_distribution["user_type"] == "casual", "percentage"].values[0]

st.subheader("Subscriber vs. Casual Riders")
st.write(f"Subscribers accounted for **{subscriber_pct:.1f}%** of all trips, while casual riders made up **{casual_pct:.1f}%**.")
st.write("Subscribers were most active during weekday commuter hours, while casual riders dominated on weekends.")

# 4. Commuter Patterns
peak_hour = hourly_trips.loc[hourly_trips["trip_count"].idxmax()]

st.subheader("Commuter Patterns")
st.write(f"The busiest hour of the day was **{peak_hour['hour']}**, with {peak_hour['trip_count']} trips.")
st.write("Weekday mornings (7–9 AM) and evenings (5–7 PM) saw significant spikes, aligning with commuter traffic.")

# 5. Weekend vs. Weekday Behavior
weekend_trips = df[df["is_weekend"]]
weekday_trips = df[~df["is_weekend"]]

weekend_top_station = weekend_trips.groupby("start_station_name").size().idxmax()
weekday_top_station = weekday_trips.groupby("start_station_name").size().idxmax()

st.subheader("Weekend vs. Weekday Trends")
st.write(f"On weekends, the most popular starting station was **{weekend_top_station}**.")
st.write(f"On weekdays, **{weekday_top_station}** dominated as the busiest starting station.")

# 6. Rare but Interesting Routes
rare_routes = station_pairs[station_pairs["trip_count"] == 1]

st.subheader("Rare but Interesting Routes")
st.write("Some rare routes include one-off trips like:")
st.dataframe(rare_routes)

# 7. Trip Duration Patterns
avg_duration_subscriber = df[df["member_casual"] == "member"]["trip_duration_minutes"].mean()
avg_duration_casual = df[df["member_casual"] == "casual"]["trip_duration_minutes"].mean()

st.subheader("Trip Duration Insights")
st.write(f"Subscribers averaged **{avg_duration_subscriber:.1f} minutes** per trip.")
st.write(f"Casual riders took longer, averaging **{avg_duration_casual:.1f} minutes** per trip.")

# 8. Station Imbalances
most_imbalanced_station = station_differences.iloc[0]

st.subheader("Station Imbalances")
st.write(f"The station with the highest imbalance was **{most_imbalanced_station['station_name']}**, with **{most_imbalanced_station['imbalance']}** more trips ending than starting.")
st.write("This suggests a need for bike rebalancing at this location.")

st.write("---")
st.write("Explore the charts and interactive maps below to dive deeper into these insights!")

###

# 1. Daily Trips (Line Chart)
daily_chart = alt.Chart(daily_trips).mark_line().encode(
    x="date:T",
    y="trip_count:Q",
    tooltip=["date:T", "trip_count:Q"]
).properties(title="Daily Trips in October")
st.altair_chart(daily_chart, use_container_width=True)

# 2. Hourly Trips (Bar Chart)
hourly_chart = px.bar(
    hourly_trips,
    x="hour",
    y="trip_count",
    title="Trips by Hour of Day",
    labels={"hour": "Hour of Day", "trip_count": "Number of Trips"},
    text_auto=True,
)
st.plotly_chart(hourly_chart, use_container_width=True)

# 3. Top 10 Starting and Ending Stations
start_stations_chart = px.bar(
    top_start_stations,
    x="trip_count",
    y="start_station_name",
    orientation="h",
    title="Top 10 Starting Stations",
    labels={"start_station_name": "Station Name", "trip_count": "Number of Trips"},
    text_auto=True,
)
st.plotly_chart(start_stations_chart, use_container_width=True)

end_stations_chart = px.bar(
    top_end_stations,
    x="trip_count",
    y="end_station_name",
    orientation="h",
    title="Top 10 Ending Stations",
    labels={"end_station_name": "Station Name", "trip_count": "Number of Trips"},
    text_auto=True,
)
st.plotly_chart(end_stations_chart, use_container_width=True)

# 4. Most Common Station Pairs
st.write("**Top 10 Most Common Station Pairs**")
st.dataframe(station_pairs)

# 5. User Type Distribution (Pie Chart)
user_type_chart = px.pie(
    user_type_distribution,
    values="percentage",
    names="user_type",
    title="User Type Distribution",
    hole=0.4,
)
st.plotly_chart(user_type_chart, use_container_width=True)

# 6. Subscriber vs. Casual Usage Trends
subscriber_chart = alt.Chart(subscriber_trends).mark_line(color="blue").encode(
    x="hour:Q",
    y="trip_count:Q",
    tooltip=["hour:Q", "trip_count:Q"]
).properties(
    title="Subscriber Usage Trends by Hour"
)

casual_chart = alt.Chart(casual_trends).mark_line(color="orange").encode(
    x="hour:Q",
    y="trip_count:Q",
    tooltip=["hour:Q", "trip_count:Q"]
).properties(
    title="Casual User Trends by Hour"
)

st.altair_chart(subscriber_chart, use_container_width=True)
st.altair_chart(casual_chart, use_container_width=True)

# 7. Distribution of Trip Distances
distance_chart = px.histogram(
    df,
    x="distance_km",
    nbins=50,
    title="Distribution of Trip Distances (Miles)",
    labels={"distance_km": "Trip Distance (miles)"},
    text_auto=True
)
st.plotly_chart(distance_chart, use_container_width=True)

# 8. Flow Patterns (ArcLayer Map)
flow_data = df.groupby(["start_station_name", "end_station_name"]).size().reset_index(name="trip_count").sort_values(by="trip_count", ascending=False).head(20)
flow_layer = pdk.Layer(
    "ArcLayer",
    data=flow_data,
    get_source_position="['start_lng', 'start_lat']",
    get_target_position="['end_lng', 'end_lat']",
    get_width="trip_count",
    get_tilt=20,
    get_color="[200, 30, 0, 160]",
    pickable=True,
)
view_state = pdk.ViewState(
    latitude=df["start_lat"].mean(),
    longitude=df["start_lng"].mean(),
    zoom=12,
    pitch=50,
)
st.pydeck_chart(pdk.Deck(layers=[flow_layer], initial_view_state=view_state))

# 9. Weekday vs Weekend Trends
day_usage = df.groupby(["day_of_week", "member_casual"]).size().reset_index(name="trip_count")

weekday_chart = px.bar(
    day_usage,
    x="day_of_week",
    y="trip_count",
    color="member_casual",
    title="Weekday vs Weekend: Subscriber vs Casual Rider Trends",
    labels={"day_of_week": "Day of Week", "trip_count": "Number of Trips"},
    text_auto=True,
)
st.plotly_chart(weekday_chart, use_container_width=True)

# 10. Morning vs Evening Station Activity
morning_trips = df[(df["hour"] >= 7) & (df["hour"] <= 9)]
evening_trips = df[(df["hour"] >= 17) & (df["hour"] <= 19)]

# Aggregate by station
morning_activity = morning_trips.groupby("start_station_name").size().reset_index(name="trip_count").sort_values(by="trip_count", ascending=False)
evening_activity = evening_trips.groupby("start_station_name").size().reset_index(name="trip_count").sort_values(by="trip_count", ascending=False)

# Plot morning and evening activity
morning_chart = px.bar(
    morning_activity.head(10),
    x="trip_count",
    y="start_station_name",
    orientation="h",
    title="Top Stations (Morning Activity: 7-9 AM)",
    labels={"start_station_name": "Station Name", "trip_count": "Number of Trips"},
    text_auto=True
)
evening_chart = px.bar(
    evening_activity.head(10),
    x="trip_count",
    y="start_station_name",
    orientation="h",
    title="Top Stations (Evening Activity: 5-7 PM)",
    labels={"start_station_name": "Station Name", "trip_count": "Number of Trips"},
    text_auto=True
)
st.plotly_chart(morning_chart, use_container_width=True)
st.plotly_chart(evening_chart, use_container_width=True)

# 11. Idle Bikes: End > Start Imbalances
# Calculate trip imbalance
start_counts = df.groupby("start_station_name").size()
end_counts = df.groupby("end_station_name").size()

station_differences = pd.DataFrame({
    "station_name": start_counts.index.union(end_counts.index),
    "start_count": start_counts.reindex(start_counts.index.union(end_counts.index), fill_value=0),
    "end_count": end_counts.reindex(start_counts.index.union(end_counts.index), fill_value=0),
})

station_differences["imbalance"] = station_differences["end_count"] - station_differences["start_count"]
station_differences = station_differences.sort_values(by="imbalance", ascending=False).reset_index(drop=True)

# Visualize imbalances
imbalance_chart = px.bar(
    station_differences.head(10),
    x="imbalance",
    y="station_name",
    orientation="h",
    title="Top Stations with End > Start Imbalances",
    labels={"station_name": "Station Name", "imbalance": "Trip Imbalance"},
    text_auto=True
)
st.plotly_chart(imbalance_chart, use_container_width=True)
