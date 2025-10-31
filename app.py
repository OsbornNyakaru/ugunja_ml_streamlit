import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import calendar  # For calendar heatmap

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Ugunja Intelligence Engine",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. HIDE STREAMLIT BRANDING (Makes it look professional) ---
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- 3. DATA LOADING & CACHING ---
@st.cache_data
def load_data():
    try:
        demand_df = pd.read_csv('./data/demand_forecasting.csv')
        maintenance_df = pd.read_csv('./data/predictive_maintenance.csv')
        route_df = pd.read_csv('./data/route_optimization.csv')
        
        demand_df['date'] = pd.to_datetime(demand_df['date'])
        maintenance_df['date'] = pd.to_datetime(maintenance_df['date'])
        
        return demand_df, maintenance_df, route_df
    except FileNotFoundError:
        st.error("FATAL ERROR: Make sure all three CSV files are in the same folder as app.py")
        return None, None, None

demand_df, maintenance_df, route_df = load_data()

if demand_df is None:
    st.stop()

# --- 4. SIDEBAR NAVIGATION ---
st.sidebar.image("https://i.imgur.com/83ww43k.png", width=200) # Placeholder for Ugunja Logo
st.sidebar.title("Ugunja: The Intelligence Engine")

page = st.sidebar.radio(
    "Select a Model's Insights",
    [
        "📈 Demand Forecasting",
        "🔧 Predictive Maintenance",
        "🚚 Route Optimization",
    ],
    label_visibility="collapsed"
)

st.sidebar.info(
    "This dashboard demonstrates the live insights from our AI models. "
    "The data is a simulation of a real-world Greenwells fleet."
)

# --- 5. PAGE: DEMAND FORECASTING ---
if page == "📈 Demand Forecasting":

    st.title("📈 Demand Forecasting Insights")
    st.markdown("Predicting demand to move from a *reactive* to a *predictive* supply chain.")

    # --- KPIs ---
    total_predicted = demand_df['predicted_demand_cylinders'].sum()
    highest_demand_zone = demand_df.groupby('neighborhood')['predicted_demand_cylinders'].sum().idxmax()
    
    col1, col2 = st.columns(2)
    col1.metric("Total Predicted Demand (Next 6mo)", f"{total_predicted:,} Cylinders")
    col2.metric("Projected Hotspot", highest_demand_zone)

    # --- ADVANCED EDA 1: Calendar Heatmap ---
    st.subheader("Daily Demand Pattern (Calendar Heatmap)")
    st.markdown("This shows daily demand intensity. Our model sees that demand peaks on weekends and holidays.")
    
    try:
        # Requires 'calplot' - if not installed, we'll fall back.
        # Let's do a manual Plotly one to avoid new dependencies.
        daily_demand = demand_df.groupby('date')['predicted_demand_cylinders'].sum().reset_index()
        daily_demand['day_of_week'] = daily_demand['date'].dt.day_name()
        daily_demand['week_of_year'] = daily_demand['date'].dt.isocalendar().week
        daily_demand['month'] = daily_demand['date'].dt.month_name()
        
        fig_cal = px.density_heatmap(
            daily_demand, x="week_of_year", y="day_of_week", z="predicted_demand_cylinders",
            histfunc="avg", title="Average Predicted Demand by Day of Week",
            color_continuous_scale="OrRd",
            facet_col="month", facet_col_wrap=4,
            labels={'predicted_demand_cylinders': 'Avg Demand'}
        )
        fig_cal.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cal, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate calendar heatmap: {e}")

    # --- ADVANCED EDA 2: Time Series Decomposition ---
    st.subheader("Decomposing the 'Why' of Demand")
    st.markdown("We break down demand into its core components to prove our model understands complex, long-term patterns.")
    
    daily_demand_ts = demand_df.groupby('date')['predicted_demand_cylinders'].sum().reset_index().set_index('date')
    if not daily_demand_ts.empty and len(daily_demand_ts) > 14:
        decomposition = seasonal_decompose(daily_demand_ts['predicted_demand_cylinders'], model='additive', period=7)
        fig_decomp = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            subplot_titles=("1. Observed: The actual data", "2. Trend: The long-term direction", "3. Seasonal: The weekly pattern", "4. Residual: The 'noise'")
        )
        fig_decomp.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend', line=dict(color='orange')), row=2, col=1)
        fig_decomp.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal', line=dict(color='green')), row=3, col=1)
        fig_decomp.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='markers', name='Residual', marker=dict(size=2, color='gray')), row=4, col=1)
        fig_decomp.update_layout(height=700, showlegend=False, margin=dict(t=100))
        st.plotly_chart(fig_decomp, use_container_width=True)


# --- 6. PAGE: PREDICTIVE MAINTENANCE ---
elif page == "🔧 Predictive Maintenance":

    st.title("🔧 Predictive Maintenance Insights")
    st.markdown("Preventing costly breakdowns *before* they happen. This is our 'insurance policy' for the fleet.")

    # --- KPIs ---
    high_risk_vehicles = maintenance_df[maintenance_df['predicted_risk_score'] > 0.8]['vehicle_id'].nunique()
    total_fleet_size = maintenance_df['vehicle_id'].nunique()
    
    col1, col2 = st.columns(2)
    col1.metric("Total Fleet Size", f"{total_fleet_size} Vehicles")
    col2.metric("🔴 Vehicles at High Risk (>80%)", f"{high_risk_vehicles}", delta_color="inverse")

    # --- ADVANCED EDA 3: Fleet Risk Treemap ---
    st.subheader("At-a-Glance Fleet Risk Overview (Treemap)")
    st.markdown("This shows your entire fleet in one box. The **size** of the rectangle is the vehicle's mileage (how much it works), and the **color** is its risk score. You can instantly spot the high-risk, high-use vehicles.")
    
    # Get the latest data point for each vehicle
    latest_fleet_data = maintenance_df.sort_values('date').drop_duplicates('vehicle_id', keep='last')
    
    fig_tree = px.treemap(
        latest_fleet_data,
        path=[px.Constant("All Vehicles"), 'vehicle_id'], # Create a hierarchy
        values='mileage',
        color='predicted_risk_score',
        color_continuous_scale='OrRd', # Orange-Red is perfect for risk
        hover_data={'predicted_failure_type': True, 'predicted_risk_score': ':.2f', 'mileage': True}
    )
    fig_tree.update_layout(margin = dict(t=25, l=25, r=25, b=25))
    st.plotly_chart(fig_tree, use_container_width=True)

    # --- ADVANCED EDA 4: Anomaly Detection ---
    st.subheader("Finding the 'Needle in the Haystack'")
    st.markdown("Our model spots the *one* bad truck out of the *entire* healthy fleet.")
    
    col1, col2 = st.columns(2)
    with col1:
        # Anomaly Scatter Plot
        fig_scatter = px.scatter(
            maintenance_df, x="avg_fuel_consumption_l_100km", y="predicted_risk_score",
            color="vehicle_id", hover_data=['date', 'predicted_failure_type'],
            title="Fleet-Wide Anomaly Detection"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        # Risk Distribution Box Plot
        fig_box = px.box(
            maintenance_df, x="vehicle_id", y="predicted_risk_score",
            color="vehicle_id", title="Vehicle Risk Score Distribution"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # --- Drill-down on a specific vehicle ---
    st.header("Vehicle-Specific Anomaly Trend (The 'Proof')")
    all_vehicles = maintenance_df['vehicle_id'].unique()
    anomalous_vehicle = 'KDC 456Y' if 'KDC 456Y' in all_vehicles else all_vehicles[0]
    
    selected_vehicle = st.selectbox(
        "Select a Vehicle to Analyze",
        options=all_vehicles,
        index=list(all_vehicles).index(anomalous_vehicle) # Default to the anomalous one
    )
    
    vehicle_df = maintenance_df[maintenance_df['vehicle_id'] == selected_vehicle]
    
    col1, col2 = st.columns(2)
    with col1:
        fig_fuel_trend = px.line(vehicle_df, x='date', y='avg_fuel_consumption_l_100km', title=f"Fuel Consumption Trend")
        st.plotly_chart(fig_fuel_trend, use_container_width=True)
    with col2:
        fig_risk_trend = px.line(vehicle_df, x='date', y='predicted_risk_score', title=f"Predicted Risk Score Trend")
        fig_risk_trend.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_risk_trend, use_container_width=True)


# --- 7. PAGE: ROUTE OPTIMIZATION ---
elif page == "🚚 Route Optimization":

    st.title("🚚 Route Optimization Insights")
    st.markdown("Saving time and money by finding the most efficient path, every time.")
    
    # --- ADVANCED EDA 5: Interactive ROI Calculator ---
    st.header("💸 Interactive ROI Calculator (The 'Money Shot')")
    st.markdown("This is a direct business proposal. Use the sliders to see how much money Ugunja saves you based on *your* operational costs.")
    
    savings_df = route_df.groupby('route_type').sum(numeric_only=True)
    total_km_saved = savings_df.loc['Standard_Route']['total_distance_km'] - savings_df.loc['Ugunja_AI_Route']['total_distance_km']
    total_hours_saved = savings_df.loc['Standard_Route']['total_time_hours'] - savings_df.loc['Ugunja_AI_Route']['total_time_hours']
    num_routes = int(len(route_df)/2)

    col1, col2 = st.columns(2)
    with col1:
        cost_per_km_ksh = st.slider("Fuel/Maint. Cost per KM (KSh)", 20, 100, 40)
        cost_per_hour_ksh = st.slider("Driver Labor Cost per Hour (KSh)", 300, 1000, 500)
    
    with col2:
        total_fuel_savings = total_km_saved * cost_per_km_ksh
        total_labor_savings = total_hours_saved * cost_per_hour_ksh
        total_savings = total_fuel_savings + total_labor_savings
        
        st.metric(
            label=f"ESTIMATED TOTAL SAVINGS (from {num_routes} routes)",
            value=f"KSh {total_savings:,.0f}"
        )
        st.success(f"From **KSh {total_fuel_savings:,.0f}** in fuel/maint. and **KSh {total_labor_savings:,.0f}** in labor.")
    
    # --- ADVANCED EDA 6: Waterfall Chart ---
    st.subheader("Visualizing Your Savings (Waterfall Chart)")
    st.markdown("This chart breaks down the exact financial impact of our AI.")
    
    fig_waterfall = go.Figure(go.Waterfall(
        name = "Savings", orientation = "v",
        measure = ["relative", "relative", "total"],
        x = ["Fuel/Maint. Savings (from KM)", "Labor Savings (from Hours)", "Total Savings"],
        text = [f"KSh {total_fuel_savings:,.0f}", f"KSh {total_labor_savings:,.0f}", f"KSh {total_savings:,.0f}"],
        y = [total_fuel_savings, total_labor_savings, total_savings],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        increasing = {"marker":{"color":"orange"}},
        totals = {"marker":{"color":"green"}}
    ))
    fig_waterfall.update_layout(title = "Financial Breakdown of Savings", showlegend = False)
    st.plotly_chart(fig_waterfall, use_container_width=True)

    # --- ADVANCED EDA 7: Efficiency Frontier ---
    st.subheader("Proving Our AI is Smarter (The 'Efficiency Frontier')")
    st.markdown("This shows every route. The 'Standard' routes are a scattered cloud. The 'Ugunja AI' routes are a tight, optimized line. Our AI consistently finds the best balance of distance and time.")
    
    fig_frontier = px.scatter(
        route_df,
        x="total_distance_km", y="total_time_hours",
        color="route_type",
        title="Ugunja AI vs. Standard Routes",
        color_discrete_map={'Standard_Route': 'grey', 'Ugunja_AI_Route': 'orange'},
        hover_data=['route_id']
    )
    st.plotly_chart(fig_frontier, use_container_width=True)