import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Ugunja Intelligence Engine",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. DATA LOADING ---
# Use @st.cache_data to load data once and store in cache
@st.cache_data
def load_data():
    try:
        demand_df = pd.read_csv('./data/demand_forecasting.csv')
        maintenance_df = pd.read_csv('./data/predictive_maintenance.csv')
        route_df = pd.read_csv('./data/route_optimization.csv')
        
        # Convert date columns for proper plotting
        demand_df['date'] = pd.to_datetime(demand_df['date'])
        maintenance_df['date'] = pd.to_datetime(maintenance_df['date'])
        
        return demand_df, maintenance_df, route_df
    except FileNotFoundError:
        st.error("ERROR: Make sure all three CSV files are in the same folder as app.py")
        return None, None, None

demand_df, maintenance_df, route_df = load_data()

# Stop the app if data didn't load
if demand_df is None:
    st.stop()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("Ugunja 🔥")
st.sidebar.subheader("The Intelligence Engine")

page = st.sidebar.radio(
    "Select a Dashboard",
    [
        "📈 Demand Forecasting",
        "🔧 Predictive Maintenance",
        "🚚 Route Optimization",
    ],
    label_visibility="hidden"
)

st.sidebar.info(
    "This dashboard is a live demonstration of the Ugunja AI models, "
    "showcasing insights from our (simulated) data."
)

# --- 4. PAGE: DEMAND FORECASTING ---
if page == "📈 Demand Forecasting":

    st.title("📈 Demand Forecasting Insights")
    st.markdown("Predicting demand to move from a *reactive* to a *predictive* supply chain.")

    # --- Data Filtering ---
    st.subheader("Explore Predicted Demand")
    col1, col2 = st.columns(2)
    with col1:
        neighborhoods = demand_df['neighborhood'].unique()
        selected_neighborhoods = st.multiselect(
            "Filter by Neighborhood",
            options=neighborhoods,
            default=neighborhoods
        )
    with col2:
        days = demand_df['day_of_week'].unique()
        selected_days = st.multiselect(
            "Filter by Day of Week",
            options=days,
            default=days
        )
    
    filtered_demand_df = demand_df[
        (demand_df['neighborhood'].isin(selected_neighborhoods)) &
        (demand_df['day_of_week'].isin(selected_days))
    ]

    # --- Visuals ---
    st.plotly_chart(
        px.line(
            filtered_demand_df.groupby('date')['predicted_demand_cylinders'].sum().reset_index(),
            x='date',
            y='predicted_demand_cylinders',
            title="Predicted Demand Over Time"
        ),
        use_container_width=True
    )

    st.header("Advanced Model Insights")
    st.markdown("Our models don't just predict *what*, they understand *why*.")
    
    col1, col2 = st.columns(2)
    with col1:
        # --- Advanced EDA 1: Time Series Decomposition ---
        st.subheader("Understanding Patterns (Time Series Decomposition)")
        
        # Aggregate data for decomposition
        daily_demand = demand_df.groupby('date')['predicted_demand_cylinders'].sum().reset_index().set_index('date')
        if not daily_demand.empty and len(daily_demand) > 14: # Need enough data to decompose
            decomposition = seasonal_decompose(daily_demand['predicted_demand_cylinders'], model='additive', period=7)
            
            fig_decomp = make_subplots(
                rows=4, cols=1, shared_xaxes=True,
                subplot_titles=("Observed", "Trend", "Seasonal (Weekly)", "Residual")
            )
            fig_decomp.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'), row=1, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='markers', name='Residual', marker=dict(size=2)), row=4, col=1)
            fig_decomp.update_layout(height=600, showlegend=False, margin=dict(t=50))
            st.plotly_chart(fig_decomp, use_container_width=True)
        else:
            st.warning("Not enough data to perform time series decomposition.")

    with col2:
        # --- Advanced EDA 2: Correlation Heatmap ---
        st.subheader("What Drives Demand? (Correlation Matrix)")
        
        df_corr = demand_df.copy()
        day_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
        df_corr['day_of_week_num'] = df_corr['day_of_week'].map(day_map)
        weather_map = {'Rainy': 0, 'Cloudy': 1, 'Sunny': 2}
        df_corr['weather_num'] = df_corr['weather'].map(weather_map)
        
        corr_df = df_corr[['predicted_demand_cylinders', 'day_of_week_num', 'weather_num', 'is_holiday']]
        corr_matrix = corr_df.corr()

        fig_heatmap = px.imshow(
            corr_matrix, text_auto=True, aspect="auto",
            color_continuous_scale='OrRd', labels=dict(color="Correlation")
        )
        fig_heatmap.update_xaxes(side="top")
        fig_heatmap.update_layout(height=600, margin=dict(t=100))
        st.plotly_chart(fig_heatmap, use_container_width=True)

# --- 5. PAGE: PREDICTIVE MAINTENANCE ---
elif page == "🔧 Predictive Maintenance":

    st.title("🔧 Predictive Maintenance Insights")
    st.markdown("Preventing costly breakdowns *before* they happen.")

    # --- KPIs ---
    high_risk_vehicles = maintenance_df[maintenance_df['predicted_risk_score'] > 0.8]['vehicle_id'].nunique()
    st.metric("Vehicles at High Risk (>80%)", f"{high_risk_vehicles}", delta_color="inverse")

    st.header("Advanced Model Insights")
    st.markdown("Our models find the 'needle in the haystack'—spotting anomalies across the entire fleet.")

    col1, col2 = st.columns(2)
    with col1:
        # --- Advanced EDA 3: Anomaly Scatter Plot ---
        st.subheader("Fleet-Wide Anomaly Detection")
        fig_scatter = px.scatter(
            maintenance_df,
            x="avg_fuel_consumption_l_100km",
            y="predicted_risk_score",
            color="vehicle_id",
            hover_data=['date', 'predicted_failure_type'],
            title="Finding the Outlier"
        )
        fig_scatter.update_traces(marker=dict(size=8, opacity=0.7))
        fig_scatter.update_layout(margin=dict(t=50))
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # --- Advanced EDA 4: Risk Distribution Box Plot ---
        st.subheader("Vehicle Risk Score Distribution")
        fig_box = px.box(
            maintenance_df,
            x="vehicle_id",
            y="predicted_risk_score",
            color="vehicle_id",
            title="Comparing Risk Profiles"
        )
        fig_box.update_layout(margin=dict(t=50))
        st.plotly_chart(fig_box, use_container_width=True)

    # --- Drill-down on a specific vehicle ---
    st.header("Vehicle-Specific Anomaly Trend")
    all_vehicles = maintenance_df['vehicle_id'].unique()
    # Default to the anomalous vehicle for the "wow" factor
    anomalous_vehicle = 'KDC 456Y' if 'KDC 456Y' in all_vehicles else all_vehicles[0]
    
    selected_vehicle = st.selectbox(
        "Select a Vehicle to Analyze",
        options=all_vehicles,
        index=list(all_vehicles).index(anomalous_vehicle)
    )

    vehicle_df = maintenance_df[maintenance_df['vehicle_id'] == selected_vehicle]
    
    col1, col2 = st.columns(2)
    with col1:
        fig_fuel_trend = px.line(
            vehicle_df, x='date', y='avg_fuel_consumption_l_100km',
            title=f"Fuel Consumption Trend for {selected_vehicle}"
        )
        st.plotly_chart(fig_fuel_trend, use_container_width=True)
    with col2:
        fig_risk_trend = px.line(
            vehicle_df, x='date', y='predicted_risk_score',
            title=f"Predicted Risk Score Trend for {selected_vehicle}"
        )
        fig_risk_trend.update_yaxes(range=[0, 1]) # Keep Y-axis consistent
        st.plotly_chart(fig_risk_trend, use_container_width=True)


# --- 6. PAGE: ROUTE OPTIMIZATION ---
elif page == "🚚 Route Optimization":

    st.title("🚚 Route Optimization Insights")
    st.markdown("Saving time and money by finding the most efficient path, every time.")
    
    # --- Advanced EDA 5: Interactive ROI Calculator ---
    st.header("💸 Interactive ROI Calculator")
    st.markdown("See for yourself. Use the sliders to estimate your *actual* savings based on our model's performance on this dataset.")
    
    # Calculate total savings from the data
    savings_df = route_df.groupby('route_type').sum(numeric_only=True)
    total_km_saved = savings_df.loc['Standard_Route']['total_distance_km'] - savings_df.loc['Ugunja_AI_Route']['total_distance_km']
    total_hours_saved = savings_df.loc['Standard_Route']['total_time_hours'] - savings_df.loc['Ugunja_AI_Route']['total_time_hours']
    
    col1, col2 = st.columns(2)
    with col1:
        cost_per_km_ksh = st.slider("Select Cost per KM (KSh)", 20, 100, 40)
        cost_per_hour_ksh = st.slider("Select Driver Cost per Hour (KSh)", 300, 1000, 500)
    
    with col2:
        # Calculate total savings
        total_fuel_savings = total_km_saved * cost_per_km_ksh
        total_labor_savings = total_hours_saved * cost_per_hour_ksh
        total_savings = total_fuel_savings + total_labor_savings
        
        # Display the metrics
        st.subheader("Estimated Total Savings")
        st.metric(
            label=f"Total Savings (based on {int(len(route_df)/2)} routes)",
            value=f"KSh {total_savings:,.0f}"
        )
        st.markdown(
            f"From **KSh {total_fuel_savings:,.0f}** in fuel and **KSh {total_labor_savings:,.0f}** in labor."
        )

    st.header("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        # --- Advanced EDA 6: Efficiency Frontier ---
        st.subheader("The Efficiency Frontier")
        fig_frontier = px.scatter(
            route_df,
            x="total_distance_km",
            y="total_time_hours",
            color="route_type",
            title="Ugunja AI vs. Standard Routes",
            color_discrete_map={'Standard_Route': 'grey', 'Ugunja_AI_Route': 'orange'},
            hover_data=['route_id']
        )
        st.plotly_chart(fig_frontier, use_container_width=True)
    
    with col2:
        # --- Basic Bar Charts for Comparison ---
        st.subheader("Average Savings per Route")
        avg_savings_df = route_df.groupby('route_type').mean(numeric_only=True).reset_index()
        fig_dist = px.bar(
            avg_savings_df, x='route_type', y='total_distance_km',
            title="Avg. Distance (KM)", color='route_type',
            color_discrete_map={'Standard_Route': 'grey', 'Ugunja_AI_Route': 'orange'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)