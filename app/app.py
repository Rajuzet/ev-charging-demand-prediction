"""
⚡ EV Charging Demand Prediction & Usage Pattern Analytics
GUVI | HCL Machine Learning Capstone Project
Authors: Rajuzet
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from lime.lime_tabular import LimeTabularExplainer

# 1. Page Configuration & Theme
st.set_page_config(
    page_title="VoltPulse | EV Charging Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium Custom Styling
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Top Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F766E 100%);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #F8FAFC;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 6px;
    }
    .badge-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    /* KPI Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border-top: 4px solid #0284C7;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 700;
        color: #64748B;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #10B981;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Container Cards */
    .glass-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }
    
    /* Recommendations Box */
    .insight-box {
        background: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #166534;
        margin-top: 14px;
    }
    .warning-box {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #92400E;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)


# 3. Data & Model Loading
@st.cache_data
def load_clean_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_csv = os.path.join(base_dir, 'data', 'processed', 'cleaned_charging_records.csv')
    df = pd.read_csv(clean_csv)
    df['StartDatetime'] = pd.to_datetime(df['StartDatetime'])
    df['EndDatetime'] = pd.to_datetime(df['EndDatetime'])
    return df


@st.cache_resource
def load_models_and_metrics():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    
    reg_model = joblib.load(os.path.join(models_dir, 'regression_model.pkl'))
    clf_model = joblib.load(os.path.join(models_dir, 'classification_model.pkl'))
    preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
    
    with open(os.path.join(models_dir, 'evaluation_metrics.json'), 'r') as f:
        metrics = json.load(f)

    daily_ts = pd.read_csv(os.path.join(models_dir, 'daily_timeseries.csv'))
    daily_ts['Date'] = pd.to_datetime(daily_ts['Date'])
    shap_sample = pd.read_csv(os.path.join(models_dir, 'shap_sample_data.csv'))

    ts_model_path = os.path.join(models_dir, 'timeseries_model.pkl')
    hw_model = joblib.load(ts_model_path) if os.path.exists(ts_model_path) else None

    # Load precomputed SHAP and decomposition artifacts if available
    shap_imp_file = os.path.join(models_dir, 'shap_importance.json')
    shap_imp = None
    if os.path.exists(shap_imp_file):
        with open(shap_imp_file, 'r') as f:
            shap_imp = json.load(f)

    decomp_file = os.path.join(models_dir, 'timeseries_decomposition.csv')
    decomp_df = pd.read_csv(decomp_file) if os.path.exists(decomp_file) else None
    if decomp_df is not None:
        decomp_df['Date'] = pd.to_datetime(decomp_df['Date'])

    return reg_model, clf_model, preprocessor, metrics, daily_ts, shap_sample, hw_model, shap_imp, decomp_df


def process_custom_uploaded_csv(file_obj, preprocessor, reg_model, clf_model, df_baseline):
    """
    Automated pipeline for uploaded CSV datasets:
    - Cleans duplicates and validates datetimes/durations
    - Engineers temporal and historical features
    - Runs batch regression (Demand in kWh) and classification (High vs Normal)
    """
    df_up = pd.read_csv(file_obj)
    df_up = df_up.drop_duplicates()
    
    # Parse datetimes
    df_up['StartDatetime'] = pd.to_datetime(df_up['StartDatetime'], errors='coerce')
    df_up['EndDatetime'] = pd.to_datetime(df_up['EndDatetime'], errors='coerce')
    df_up = df_up.dropna(subset=['StartDatetime', 'EndDatetime'])
    
    # Calculate duration
    calc_dur = (df_up['EndDatetime'] - df_up['StartDatetime']).dt.total_seconds() / 60.0
    if 'Duration' in df_up.columns:
        df_up['Duration_Mins'] = np.where(df_up['Duration'] > 0, df_up['Duration'], calc_dur)
    else:
        df_up['Duration_Mins'] = calc_dur
    
    valid_mask = (df_up['Duration_Mins'] > 0) & (df_up['Duration_Mins'] <= 1440)
    df_up = df_up[valid_mask].reset_index(drop=True)
    
    # Temporal feature extraction
    df_up['StartHour'] = df_up['StartDatetime'].dt.hour
    df_up['StartDay'] = df_up['StartDatetime'].dt.day
    df_up['StartMonth'] = df_up['StartDatetime'].dt.month
    df_up['DayOfWeek'] = df_up['StartDatetime'].dt.day_of_week
    df_up['DayName'] = df_up['StartDatetime'].dt.day_name()
    df_up['IsWeekend'] = df_up['DayOfWeek'].isin([5, 6]).astype(int)
    
    def assign_tod(h):
        if 5 <= h < 12: return 'Morning'
        elif 12 <= h < 17: return 'Afternoon'
        elif 17 <= h < 22: return 'Evening'
        else: return 'Night'
    df_up['TimeOfDay'] = df_up['StartHour'].apply(assign_tod)
    
    df_up['Hour_Sin'] = np.sin(2 * np.pi * df_up['StartHour'] / 24.0)
    df_up['Hour_Cos'] = np.cos(2 * np.pi * df_up['StartHour'] / 24.0)
    
    # Match historical baselines from training distribution
    loc_map = df_baseline.groupby('Location')['Demand'].mean().to_dict()
    type_map = df_baseline.groupby('ChargerType')['Demand'].mean().to_dict()
    overall_mean = df_baseline['Demand'].mean()
    
    df_up['Loc_Avg_Demand'] = df_up['Location'].map(loc_map).fillna(overall_mean)
    df_up['ChargerType_Avg_Demand'] = df_up['ChargerType'].map(type_map).fillna(overall_mean)
    
    feat_cols = ['Location', 'ChargerCompany', 'ChargerType', 'TimeOfDay', 'IsWeekend',
                 'Duration_Mins', 'StartHour', 'StartDay', 'StartMonth', 'DayOfWeek',
                 'Hour_Sin', 'Hour_Cos', 'Loc_Avg_Demand', 'ChargerType_Avg_Demand']
    
    # Batch ML Predictions
    X_proc = preprocessor.transform(df_up[feat_cols])
    df_up['Predicted_Demand_kWh'] = np.round(np.maximum(0.5, reg_model.predict(X_proc)), 2)
    df_up['Predicted_High_Demand'] = clf_model.predict(X_proc)
    df_up['High_Demand_Prob'] = np.round(clf_model.predict_proba(X_proc)[:, 1], 4)
    df_up['Demand_Tier'] = np.where(df_up['Predicted_High_Demand'] == 1, 'High Demand', 'Normal Demand')
    
    # Standardize target columns
    if 'Demand' not in df_up.columns:
        df_up['Demand'] = df_up['Predicted_Demand_kWh']
        df_up['High_Demand'] = df_up['Predicted_High_Demand']
    else:
        if 'High_Demand' not in df_up.columns:
            df_up['High_Demand'] = (df_up['Demand'] >= 14.2).astype(int)
            
    return df_up


try:
    df_clean = load_clean_data()
    reg_model, clf_model, preprocessor, metrics, daily_ts, shap_sample, hw_model, shap_imp, decomp_df = load_models_and_metrics()
    is_ready = True
except Exception as err:
    st.error(f"Error loading project artifacts: {err}")
    is_ready = False

if is_ready:
    # 4. Sidebar Branding & Interactive Filters
    with st.sidebar:
        # Modern EV Station Vector Logo
        st.markdown("""
        <div style="text-align: center; margin-bottom: 16px;">
            <svg width="68" height="68" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="24" height="24" rx="12" fill="#0284C7"/>
                <path d="M11 4L5 14H11L10 20L19 9H13L15 4H11Z" fill="#F8FAFC" stroke="#38BDF8" stroke-width="0.75" stroke-linejoin="round"/>
            </svg>
            <div style="font-size: 1.35rem; font-weight: 800; color: #0F172A; margin-top: 6px;">VoltPulse Analytics</div>
            <div style="font-size: 0.78rem; color: #64748B; font-weight: 600;">EV Demand & Smart Grid Intelligence</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # File Upload Slot
        st.markdown("#### 📁 Upload Custom CSV Dataset")
        uploaded_file = st.file_uploader(
            "Upload Charging Records (.csv)",
            type=["csv"],
            help="Upload any raw or batch EV session CSV to auto-clean, feature engineer, and predict demand & tiers."
        )
        
        if uploaded_file is not None:
            try:
                df_uploaded = process_custom_uploaded_csv(uploaded_file, preprocessor, reg_model, clf_model, df_clean)
                st.success(f"✓ Processed {len(df_uploaded):,} uploaded sessions!")
                use_uploaded_toggle = st.checkbox("Active: Use Uploaded Data", value=True)
                if use_uploaded_toggle:
                    active_df = df_uploaded
                    is_custom_dataset = True
                else:
                    active_df = df_clean
                    is_custom_dataset = False
            except Exception as e:
                st.error(f"Error processing CSV: {e}")
                active_df = df_clean
                is_custom_dataset = False
        else:
            active_df = df_clean
            is_custom_dataset = False
        
        st.markdown("---")
        
        app_mode = st.radio(
            "📍 Core Navigation",
            [
                "📊 Executive Overview & EDA",
                "⚡ Demand Predictor (Regression)",
                "🎯 High-Demand Classifier",
                "⏳ Time-Series Forecasting",
                "🔍 Explainable AI (SHAP & LIME)",
                "🏆 Model Leaderboard"
            ],
            index=0
        )
        
        st.markdown("---")
        st.markdown("#### ⚙️ Global Station Filters")
        locations_list = sorted(active_df['Location'].unique())
        selected_locs = st.multiselect("Station Location", locations_list, default=locations_list)
        
        charger_types = sorted(active_df['ChargerType'].unique())
        selected_types = st.multiselect("Charger Type", charger_types, default=charger_types, format_func=lambda x: "Level 2 AC" if x == 0 else "DC Fast Charger")
        
        st.markdown("---")
        st.markdown("""
        <div style="background: #F1F5F9; padding: 12px; border-radius: 8px; font-size: 0.75rem; color: #475569;">
            <b>GUVI | HCL Project</b><br>
            • Records: 72,421 sessions<br>
            • Reg Model: Random Forest (R² = 0.80)<br>
            • Clf Model: XGBoost (F1 = 0.90)
        </div>
        """, unsafe_allow_html=True)

    # Filtered View based on active dataset
    filtered_df = active_df[
        (active_df['Location'].isin(selected_locs)) &
        (active_df['ChargerType'].isin(selected_types))
    ]

    # TOP HERO HEADER
    badge_text = f"CUSTOM UPLOAD ({len(active_df):,} SESSIONS)" if is_custom_dataset else "GUVI | HCL CAPSTONE"
    st.markdown(f"""
    <div class="hero-banner">
        <div>
            <div class="hero-title">
                ⚡ EV Charging Demand & Usage Analytics
            </div>
            <div class="hero-subtitle">
                AI-driven operational intelligence for electric vehicle infrastructure, grid load forecasting & user behavior analytics.
            </div>
        </div>
        <div style="text-align: right; margin-top: 8px;">
            <span class="badge-pill">{badge_text}</span>
            <div style="font-size: 0.75rem; color: #CBD5E1; margin-top: 4px;">Active Pool: <b>{len(active_df):,} Sessions</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # If Custom File Uploaded, show Batch Scored Expander & Download Button
    if is_custom_dataset:
        with st.expander("📥 Uploaded Dataset Batch Scoring Summary & Export", expanded=True):
            st.markdown(f"**Uploaded File:** `{uploaded_file.name}` | **Validated Sessions:** `{len(active_df):,}`")
            up_c1, up_c2, up_c3 = st.columns(3)
            with up_c1:
                st.metric("Total Predicted Energy", f"{active_df['Predicted_Demand_kWh'].sum():,.1f} kWh")
            with up_c2:
                st.metric("Avg Predicted Demand", f"{active_df['Predicted_Demand_kWh'].mean():.2f} kWh")
            with up_c3:
                up_high_pct = (active_df['Predicted_High_Demand'].mean()) * 100
                st.metric("High-Demand Ratio", f"{up_high_pct:.1f}%")
            
            # Show preview
            preview_cols = ['Location', 'ChargerType', 'StartDatetime', 'Duration_Mins', 'Predicted_Demand_kWh', 'Demand_Tier', 'High_Demand_Prob']
            st.dataframe(active_df[preview_cols].head(5), use_container_width=True)
            
            # Export CSV Button
            csv_data = active_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Dataset with Predictions (CSV)",
                data=csv_data,
                file_name=f"scored_{uploaded_file.name}",
                mime="text/csv",
                type="primary"
            )

    # =========================================================================
    # TAB 1: EXECUTIVE OVERVIEW & EDA
    # =========================================================================
    if app_mode == "📊 Executive Overview & EDA":
        # 5 High-Impact KPI Cards
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #0284C7;">
                <div class="metric-label">Total Sessions</div>
                <div class="metric-value">{len(filtered_df):,}</div>
                <div class="metric-sub">✓ Validated records</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #10B981;">
                <div class="metric-label">Delivered Energy</div>
                <div class="metric-value">{filtered_df['Demand'].sum()/1000:,.1f} MWh</div>
                <div class="metric-sub">⚡ Clean grid power</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #F59E0B;">
                <div class="metric-label">Avg Session Load</div>
                <div class="metric-value">{filtered_df['Demand'].mean():.2f} kWh</div>
                <div class="metric-sub">Median: {metrics.get('demand_median_threshold', 14.2):.2f} kWh</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #8B5CF6;">
                <div class="metric-label">Avg Duration</div>
                <div class="metric-value">{filtered_df['Duration_Mins'].mean():.1f} min</div>
                <div class="metric-sub">~{(filtered_df['Duration_Mins'].mean()/60):.1f} hours/plug-in</div>
            </div>
            """, unsafe_allow_html=True)
        with k5:
            high_ratio = (filtered_df['High_Demand'].mean()) * 100
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #EF4444;">
                <div class="metric-label">High-Demand Rate</div>
                <div class="metric-value">{high_ratio:.1f}%</div>
                <div class="metric-sub">Sessions ≥ 14.20 kWh</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================================================
        # ROW 1: VENUE GEOGRAPHY (WIDE) vs HARDWARE TECHNOLOGY (COMPACT)
        # =====================================================================
        col_c1, col_c2 = st.columns([1.65, 1.0])
        with col_c1:
            with st.container(border=True):
                st.markdown("""
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                    <span style="font-size: 1.08rem; font-weight: 700; color: #0F172A;">🏢 Cumulative Energy Delivered by Location</span>
                    <span style="font-size: 0.76rem; font-weight: 600; color: #0284C7; background: #E0F2FE; padding: 2px 8px; border-radius: 9999px;">14 Station Categories</span>
                </div>
                <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 10px;">
                    Aggregate energy delivered (kWh) sorted by demand intensity across physical installation venues
                </div>
                """, unsafe_allow_html=True)
                loc_agg = filtered_df.groupby('Location')['Demand'].agg(['sum', 'count', 'mean']).reset_index()
                loc_agg = loc_agg.sort_values(by='sum', ascending=False)
                fig_loc = px.bar(
                    loc_agg, x='Location', y='sum',
                    color='mean',
                    color_continuous_scale='Blues',
                    labels={
                        'Location': 'Station Venue Location',
                        'sum': 'Total Delivered (kWh)',
                        'mean': 'Avg kWh/Session'
                    },
                    template='plotly_white'
                )
                fig_loc.update_traces(
                    hovertemplate="<b>%{x}</b><br>Total Delivered: %{y:,.1f} kWh<br>Avg per Session: %{marker.color:.2f} kWh<extra></extra>"
                )
                fig_loc.update_layout(
                    height=410,
                    xaxis_title="",
                    yaxis_title="Total Energy (kWh)",
                    yaxis=dict(tickformat=",.0f"),
                    xaxis_tickangle=-38,
                    coloraxis_colorbar=dict(title="Avg kWh", thickness=12, len=0.65, y=0.5),
                    margin=dict(t=15, b=85, l=45, r=15)
                )
                st.plotly_chart(fig_loc, use_container_width=True)

        with col_c2:
            with st.container(border=True):
                st.markdown("""
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                    <span style="font-size: 1.08rem; font-weight: 700; color: #0F172A;">🔌 Technology Share & Utilization</span>
                    <span style="font-size: 0.76rem; font-weight: 600; color: #0284C7; background: #E0F2FE; padding: 2px 8px; border-radius: 9999px;">Hardware Split</span>
                </div>
                <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 10px;">
                    Delivered power split: Level 2 AC Standard (Type 0) vs. DC Fast Charging (Type 1)
                </div>
                """, unsafe_allow_html=True)
                pie_data = filtered_df.copy()
                pie_data['Charger_Label'] = pie_data['ChargerType'].map({
                    0: 'Level 2 AC Standard (Type 0)',
                    1: 'DC Fast Charger (Type 1)'
                })
                fig_pie = px.pie(
                    pie_data, names='Charger_Label', values='Demand',
                    color='Charger_Label',
                    color_discrete_map={
                        'Level 2 AC Standard (Type 0)': '#38BDF8',
                        'DC Fast Charger (Type 1)': '#0284C7'
                    },
                    hole=0.52,
                    labels={'Charger_Label': 'Hardware Spec', 'Demand': 'Energy (kWh)'},
                    template='plotly_white'
                )
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate="<b>%{label}</b><br>Total Energy: %{value:,.1f} kWh<br>Share: %{percent}<extra></extra>"
                )
                fig_pie.update_layout(
                    height=410,
                    margin=dict(t=15, b=40, l=15, r=15),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # =====================================================================
        # ROW 2: TEMPORAL PATTERNS (24-HR DIURNAL & WEEKLY CADENCE)
        # =====================================================================
        col_c3, col_c4 = st.columns([1, 1])
        with col_c3:
            with st.container(border=True):
                st.markdown("""
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                    <span style="font-size: 1.08rem; font-weight: 700; color: #0F172A;">⏰ 24-Hour Diurnal Demand Curve</span>
                    <span style="font-size: 0.76rem; font-weight: 600; color: #10B981; background: #D1FAE5; padding: 2px 8px; border-radius: 9999px;">Dual-Axis Profile</span>
                </div>
                <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 10px;">
                    Hourly session arrival volume (bars) overlaid with mean demand load per session (line)
                </div>
                """, unsafe_allow_html=True)
                hour_agg = filtered_df.groupby('StartHour')['Demand'].agg(['count', 'mean']).reset_index()
                fig_hour = go.Figure()
                fig_hour.add_trace(go.Bar(
                    x=hour_agg['StartHour'], y=hour_agg['count'], name='Session Volume (Count)',
                    marker_color='#BAE6FD', opacity=0.8, yaxis='y1',
                    hovertemplate="Hour %{x}:00<br>Sessions: %{y:,}<extra></extra>"
                ))
                fig_hour.add_trace(go.Scatter(
                    x=hour_agg['StartHour'], y=hour_agg['mean'], name='Mean Demand (kWh)',
                    line=dict(color='#0284C7', width=3), mode='lines+markers', yaxis='y2',
                    hovertemplate="Hour %{x}:00<br>Mean Load: %{y:.2f} kWh<extra></extra>"
                ))
                fig_hour.update_layout(
                    height=390,
                    xaxis=dict(title="Hour of Arrival (00:00 - 23:00 hrs)", tickmode='linear', dtick=2),
                    yaxis=dict(title="Active Session Volume", tickformat=","),
                    yaxis2=dict(title="Avg Demand (kWh)", overlaying='y', side='right', tickformat=".1f"),
                    template='plotly_white',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                    margin=dict(t=35, b=45, l=45, r=45)
                )
                st.plotly_chart(fig_hour, use_container_width=True)

        with col_c4:
            with st.container(border=True):
                st.markdown("""
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                    <span style="font-size: 1.08rem; font-weight: 700; color: #0F172A;">📅 Day of Week Behavioral Distribution</span>
                    <span style="font-size: 0.76rem; font-weight: 600; color: #10B981; background: #D1FAE5; padding: 2px 8px; border-radius: 9999px;">Weekly Cadence</span>
                </div>
                <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 10px;">
                    Weekly session traffic and average load intensity across Monday through Sunday
                </div>
                """, unsafe_allow_html=True)
                days_seq = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_agg = filtered_df.groupby('DayName')['Demand'].agg(['count', 'mean']).reindex(days_seq).reset_index()
                fig_day = px.bar(
                    day_agg, x='DayName', y='count',
                    color='mean', color_continuous_scale='Teal',
                    labels={
                        'DayName': 'Day of the Week',
                        'count': 'Total Sessions',
                        'mean': 'Avg Demand (kWh)'
                    },
                    template='plotly_white'
                )
                fig_day.update_traces(
                    hovertemplate="<b>%{x}</b><br>Session Traffic: %{y:,}<br>Mean Demand: %{marker.color:.2f} kWh<extra></extra>"
                )
                fig_day.update_layout(
                    height=390,
                    xaxis_title="",
                    yaxis_title="Total Sessions",
                    yaxis=dict(tickformat=","),
                    coloraxis_colorbar=dict(title="Avg kWh", thickness=12, len=0.65, y=0.5),
                    margin=dict(t=25, b=45, l=45, r=15)
                )
                st.plotly_chart(fig_day, use_container_width=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # =====================================================================
        # ROW 3: DURATION BEHAVIOR (WIDE) vs OPERATOR NETWORK SPLIT (COMPACT)
        # =====================================================================
        col_c5, col_c6 = st.columns([1.65, 1.0])
        with col_c5:
            with st.container(border=True):
                st.markdown("""
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                    <span style="font-size: 1.08rem; font-weight: 700; color: #0F172A;">📈 Duration vs. Energy Demand Correlation</span>
                    <span style="font-size: 0.76rem; font-weight: 600; color: #F59E0B; background: #FEF3C7; padding: 2px 8px; border-radius: 9999px;">2,500 Session Sample</span>
                </div>
                <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 10px;">
                    Plug-in duration vs. energy delivered, segmented by AC Level 2 and DC Fast Charging
                </div>
                """, unsafe_allow_html=True)
                scatter_sample = filtered_df.sample(min(2500, len(filtered_df)), random_state=42).copy()
                scatter_sample['Charger_Label'] = scatter_sample['ChargerType'].map({
                    0: 'Level 2 AC Standard',
                    1: 'DC Fast Charger'
                })
                fig_scatter = px.scatter(
                    scatter_sample,
                    x='Duration_Mins', y='Demand',
                    color='Charger_Label',
                    color_discrete_map={
                        'Level 2 AC Standard': '#0284C7',
                        'DC Fast Charger': '#F59E0B'
                    },
                    opacity=0.65,
                    labels={
                        'Duration_Mins': 'Session Duration (Minutes)',
                        'Demand': 'Energy Consumed (kWh)',
                        'Charger_Label': 'Hardware Tier'
                    },
                    template='plotly_white'
                )
                fig_scatter.update_traces(
                    hovertemplate="Duration: %{x} mins<br>Demand: %{y:.2f} kWh<extra>%{fullData.name}</extra>"
                )
                fig_scatter.update_layout(
                    height=390,
                    xaxis_title="Plug-in Duration (Minutes)",
                    yaxis_title="Energy Consumed (kWh)",
                    xaxis=dict(tickformat=","),
                    yaxis=dict(tickformat=","),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                    margin=dict(t=35, b=45, l=45, r=15)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        with col_c6:
            with st.container(border=True):
                st.markdown("""
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                    <span style="font-size: 1.08rem; font-weight: 700; color: #0F172A;">⚡ Infrastructure Provider Comparison</span>
                    <span style="font-size: 0.76rem; font-weight: 600; color: #F59E0B; background: #FEF3C7; padding: 2px 8px; border-radius: 9999px;">2 Operator Networks</span>
                </div>
                <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 10px;">
                    Total delivered power and average load delivered by network operators
                </div>
                """, unsafe_allow_html=True)
                comp_agg = filtered_df.groupby('ChargerCompany')['Demand'].agg(['sum', 'count', 'mean']).reset_index()
                comp_agg['Company_Name'] = comp_agg['ChargerCompany'].map({
                    0: 'Operator Network A',
                    1: 'Operator Network B'
                })
                fig_comp = px.bar(
                    comp_agg, x='Company_Name', y='sum',
                    color='mean', color_continuous_scale='Viridis',
                    labels={
                        'Company_Name': 'Charging Operator',
                        'sum': 'Total Energy Delivered (kWh)',
                        'mean': 'Avg kWh/Session'
                    },
                    template='plotly_white'
                )
                fig_comp.update_traces(
                    hovertemplate="<b>%{x}</b><br>Delivered: %{y:,.1f} kWh<br>Avg per Session: %{marker.color:.2f} kWh<extra></extra>"
                )
                fig_comp.update_layout(
                    height=390,
                    xaxis_title="",
                    yaxis_title="Total Energy Delivered (kWh)",
                    yaxis=dict(tickformat=",.0f"),
                    coloraxis_colorbar=dict(title="Avg kWh", thickness=12, len=0.65, y=0.5),
                    margin=dict(t=25, b=45, l=45, r=15)
                )
                st.plotly_chart(fig_comp, use_container_width=True)

    # =========================================================================
    # TAB 2: DEMAND PREDICTOR (REGRESSION)
    # =========================================================================
    elif app_mode == "⚡ Demand Predictor (Regression)":
        st.markdown("## ⚡ Session Energy Demand Predictor (Regression)")
        st.markdown(f"Trained with **{metrics.get('best_regression_model', 'Random Forest')}** regressor. Evaluated on test split:")

        # Regression Metrics Proportional Card Strip (MAE, MSE, RMSE, R2)
        best_reg_stats = metrics['regression_metrics'].get(metrics.get('best_regression_model', 'Random Forest'), {})
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 22px;">
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #0284C7; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">MAE (Mean Absolute Error)</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #0284C7; margin: 2px 0;">{best_reg_stats.get('MAE', 3.38):.3f} <span style="font-size: 0.8rem; color: #64748B;">kWh</span></div>
                <div style="font-size: 0.72rem; color: #10B981; font-weight: 600;">Avg. prediction deviation</div>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #6366F1; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">MSE (Mean Squared Error)</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #6366F1; margin: 2px 0;">{best_reg_stats.get('MSE', 36.98):.2f}</div>
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">Variance penalty</div>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #F59E0B; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">RMSE (Root Mean Sq Error)</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #F59E0B; margin: 2px 0;">{best_reg_stats.get('RMSE', 6.08):.3f} <span style="font-size: 0.8rem; color: #64748B;">kWh</span></div>
                <div style="font-size: 0.72rem; color: #10B981; font-weight: 600;">Standard error scale</div>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #10B981; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">R² Score (Explained Variance)</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #10B981; margin: 2px 0;">{best_reg_stats.get('R2', 0.7983):.4f}</div>
                <div style="font-size: 0.72rem; color: #10B981; font-weight: 600;">~80% variance captured</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # State Management for Scenario Presets
        if 'reg_dur' not in st.session_state:
            st.session_state['reg_dur'] = 90
        if 'reg_hour' not in st.session_state:
            st.session_state['reg_hour'] = 14
        if 'reg_type' not in st.session_state:
            st.session_state['reg_type'] = 0

        # Scenario Presets Buttons
        st.markdown("##### ⚡ Quick Scenario Presets")
        q1, q2, q3 = st.columns(3)
        if q1.button("🚗 Workplace Commute (2 hrs, AC)", use_container_width=True):
            st.session_state['reg_dur'] = 120
            st.session_state['reg_hour'] = 9
            st.session_state['reg_type'] = 0
            st.rerun()
        if q2.button("⚡ Highway DC Fast Charge (30 min)", use_container_width=True):
            st.session_state['reg_dur'] = 30
            st.session_state['reg_hour'] = 17
            st.session_state['reg_type'] = 1
            st.rerun()
        if q3.button("🏨 Overnight Hotel Charge (6 hrs)", use_container_width=True):
            st.session_state['reg_dur'] = 360
            st.session_state['reg_hour'] = 21
            st.session_state['reg_type'] = 0
            st.rerun()

        p_col1, p_col2 = st.columns([1, 1])
        with p_col1:
            st.markdown("#### 1. Configure Session Parameters")
            in_loc = st.selectbox("Charging Location Venue", locations_list, index=0)
            in_company = st.selectbox("Charging Station Operator", [0, 1], format_func=lambda x: f"Operator Network {x+1}")
            in_type = st.radio(
                "Hardware Speed", [0, 1],
                index=st.session_state['reg_type'],
                format_func=lambda x: "Standard AC (Type 0)" if x == 0 else "DC Fast Charger (Type 1)",
                horizontal=True
            )
            in_duration = st.slider(
                "Session Duration (minutes)", min_value=10, max_value=720,
                value=st.session_state['reg_dur'], step=5
            )
            
            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                in_hour = st.slider("Arrival Hour", 0, 23, value=st.session_state['reg_hour'])
                in_month = st.selectbox("Month of Year", list(range(1, 13)), index=5)
            with sub_c2:
                in_day = st.slider("Day of Month", 1, 31, 15)
                in_dow = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], index=2, format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])

            is_weekend = 1 if in_dow in [5, 6] else 0
            tod = 'Morning' if 5 <= in_hour < 12 else ('Afternoon' if 12 <= in_hour < 17 else ('Evening' if 17 <= in_hour < 22 else 'Night'))
            loc_avg = df_clean[df_clean['Location'] == in_loc]['Demand'].mean()
            type_avg = df_clean[df_clean['ChargerType'] == in_type]['Demand'].mean()

            run_pred = st.button("Calculate Predicted Energy (kWh)", type="primary", use_container_width=True)

        with p_col2:
            st.markdown("#### 2. Predicted Demand & Energy Intelligence")
            
            # Predict
            input_df = pd.DataFrame([{
                'Location': in_loc,
                'ChargerCompany': in_company,
                'ChargerType': in_type,
                'TimeOfDay': tod,
                'IsWeekend': is_weekend,
                'Duration_Mins': in_duration,
                'StartHour': in_hour,
                'StartDay': in_day,
                'StartMonth': in_month,
                'DayOfWeek': in_dow,
                'Hour_Sin': np.sin(2 * np.pi * in_hour / 24.0),
                'Hour_Cos': np.cos(2 * np.pi * in_hour / 24.0),
                'Loc_Avg_Demand': loc_avg,
                'ChargerType_Avg_Demand': type_avg
            }])

            proc_input = preprocessor.transform(input_df)
            predicted_demand = max(0.5, float(reg_model.predict(proc_input)[0]))
            transfer_rate = predicted_demand / (in_duration / 60.0)
            est_cost_usd = predicted_demand * 0.28

            # Metric Display
            st.markdown(f"""
            <div style="background: #F8FAFC; border: 2px solid #38BDF8; border-radius: 14px; padding: 20px; text-align: center; margin-bottom: 16px;">
                <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Estimated Energy Delivered</div>
                <div style="font-size: 2.8rem; font-weight: 800; color: #0284C7; margin: 4px 0;">{predicted_demand:.2f} <span style="font-size: 1.2rem; color: #64748B;">kWh</span></div>
                <div style="display: flex; justify-content: space-around; margin-top: 10px; font-size: 0.85rem; font-weight: 600;">
                    <span style="color: #0F766E;">⚡ Avg Power: ~{transfer_rate:.1f} kW</span>
                    <span style="color: #6366F1;">💵 Est. Cost: ~${est_cost_usd:.2f}</span>
                    <span style="color: {'#B91C1C' if predicted_demand >= 14.2 else '#15803D'};">🏷️ Tier: {'High Demand' if predicted_demand >= 14.2 else 'Normal Demand'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge Chart
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_demand,
                number={'suffix': " kWh", 'valueformat': ".2f"},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Session Energy vs Fleet Benchmark (kWh)"},
                delta={'reference': df_clean['Demand'].mean(), 'increasing': {'color': '#0284C7'}},
                gauge={
                    'axis': {'range': [0, 80], 'tickformat': ".0f", 'tickwidth': 1},
                    'bar': {'color': "#0284C7"},
                    'steps': [
                        {'range': [0, 14.2], 'color': "#E0F2FE"},
                        {'range': [14.2, 40], 'color': "#BAE6FD"},
                        {'range': [40, 80], 'color': "#7DD3FC"}
                    ],
                    'threshold': {
                        'line': {'color': "#EF4444", 'width': 3},
                        'thickness': 0.8,
                        'value': metrics.get('demand_median_threshold', 14.2)
                    }
                }
            ))
            fig_g.update_layout(height=260, margin=dict(t=30, b=20, l=20, r=20))
            st.plotly_chart(fig_g, use_container_width=True)

            # Operational guidance
            if predicted_demand >= metrics.get('demand_median_threshold', 14.2):
                st.markdown("""
                <div class="warning-box">
                    ⚠️ <b>High Energy Session:</b> This charging cycle exceeds the station median. Advised to prioritize cooling and schedule load-balancing.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="insight-box">
                    ✓ <b>Normal Demand Session:</b> Within regular grid load parameters. Standard charging profile applies.
                </div>
                """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 3: HIGH-DEMAND CLASSIFIER
    # =========================================================================
    elif app_mode == "🎯 High-Demand Classifier":
        st.markdown("## 🎯 High-Demand Session Classifier")
        st.markdown(f"Trained with **{metrics.get('best_classification_model', 'XGBoost')}** classifier to identify peak load sessions ($\ge 14.20$ kWh). Evaluated on test split:")

        # Classification Metrics Proportional Card Strip (Accuracy, Precision, Recall, F1, ROC-AUC)
        best_clf_stats = metrics['classification_metrics'].get(metrics.get('best_classification_model', 'XGBoost'), {})
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 14px;">
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #0284C7; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Accuracy</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #0284C7; margin: 2px 0;">{best_clf_stats.get('Accuracy', 0.9013)*100:.1f}%</div>
                <div style="font-size: 0.72rem; color: #10B981; font-weight: 600;">Overall correct tiers</div>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #6366F1; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Precision</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #6366F1; margin: 2px 0;">{best_clf_stats.get('Precision', 0.8745)*100:.1f}%</div>
                <div style="font-size: 0.72rem; color: #10B981; font-weight: 600;">Low false-alarm rate</div>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #EC4899; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Recall / Sensitivity</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #EC4899; margin: 2px 0;">{best_clf_stats.get('Recall', 0.9364)*100:.1f}%</div>
                <div style="font-size: 0.72rem; color: #10B981; font-weight: 600;">Captures 93.6% peak surges</div>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #F59E0B; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">F1-Score</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #F59E0B; margin: 2px 0;">{best_clf_stats.get('F1', 0.9044):.4f}</div>
                <div style="font-size: 0.72rem; color: #10B981; font-weight: 600;">Harmonic mean balance</div>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #10B981; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">ROC-AUC Score</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #10B981; margin: 2px 0;">{best_clf_stats.get('ROC_AUC', 0.9615):.4f}</div>
                <div style="font-size: 0.72rem; color: #10B981; font-weight: 600;">Outstanding separation</div>
            </div>
        </div>
        <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 10px 14px; margin-bottom: 22px; font-size: 0.8rem; color: #1E40AF; display: flex; align-items: center; justify-content: space-between;">
            <span>⚖️ <b>Class Imbalance Mitigation:</b> ADASYN synthetic oversampling boosted minority peak-demand recall from <b>83.15%</b> to <b>93.74%</b> with zero sample leakage.</span>
            <span style="background: #DBEAFE; color: #1E3A8A; font-weight: 700; padding: 3px 8px; border-radius: 6px; font-size: 0.72rem;">ADASYN + XGBoost Champion</span>
        </div>
        """, unsafe_allow_html=True)

        cl1, cl2 = st.columns([1, 1])
        with cl1:
            st.markdown("#### Configure Booking Profile")
            c_loc = st.selectbox("Location Venue", locations_list, key="clf_loc")
            c_comp = st.selectbox("Operator Network", [0, 1], key="clf_comp")
            c_type = st.radio("Charger Hardware", [0, 1], format_func=lambda x: "AC Standard" if x == 0 else "DC Fast", horizontal=True, key="clf_type")
            c_dur = st.slider("Duration (minutes)", 10, 600, 120, step=5, key="clf_dur")
            c_hour = st.slider("Time of Booking (Hour)", 0, 23, 18, key="clf_hour")
            c_dow = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], index=4, format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x], key="clf_dow")

            c_is_wknd = 1 if c_dow in [5, 6] else 0
            c_tod = 'Morning' if 5 <= c_hour < 12 else ('Afternoon' if 12 <= c_hour < 17 else ('Evening' if 17 <= c_hour < 22 else 'Night'))

            cl_btn = st.button("Classify Demand Tier", type="primary", use_container_width=True)

        with cl2:
            st.markdown("#### Real-Time Classification Result")
            cl_row = pd.DataFrame([{
                'Location': c_loc,
                'ChargerCompany': c_comp,
                'ChargerType': c_type,
                'TimeOfDay': c_tod,
                'IsWeekend': c_is_wknd,
                'Duration_Mins': c_dur,
                'StartHour': c_hour,
                'StartDay': 15,
                'StartMonth': 6,
                'DayOfWeek': c_dow,
                'Hour_Sin': np.sin(2 * np.pi * c_hour / 24.0),
                'Hour_Cos': np.cos(2 * np.pi * c_hour / 24.0),
                'Loc_Avg_Demand': df_clean[df_clean['Location'] == c_loc]['Demand'].mean(),
                'ChargerType_Avg_Demand': df_clean[df_clean['ChargerType'] == c_type]['Demand'].mean()
            }])

            proc_cl = preprocessor.transform(cl_row)
            pred_class = clf_model.predict(proc_cl)[0]
            pred_prob = clf_model.predict_proba(proc_cl)[0][1]

            if pred_class == 1:
                st.markdown(f"""
                <div style="background: #FEF2F2; border: 2px solid #EF4444; border-radius: 14px; padding: 22px; text-align: center; margin-bottom: 16px;">
                    <div style="font-size: 0.85rem; color: #991B1B; font-weight: 700;">PREDICTED CLUSTER</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #B91C1C; margin: 4px 0;">⚡ HIGH DEMAND SESSION</div>
                    <div style="font-size: 0.95rem; color: #7F1D1D; font-weight: 600;">High-Demand Probability: <b>{pred_prob*100:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(pred_prob))
                st.markdown("""
                <div class="warning-box">
                    🚨 <b>Grid Operator Action Plan:</b>
                    <ul>
                        <li>Substation capacity allocation recommended.</li>
                        <li>Consider dynamic tariff surcharge for peak-hour energy mitigation.</li>
                        <li>Recommend EV driver nearby low-demand charging hub if queue forms.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #F0FDF4; border: 2px solid #22C55E; border-radius: 14px; padding: 22px; text-align: center; margin-bottom: 16px;">
                    <div style="font-size: 0.85rem; color: #166534; font-weight: 700;">PREDICTED CLUSTER</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #15803D; margin: 4px 0;">✓ NORMAL DEMAND SESSION</div>
                    <div style="font-size: 0.95rem; color: #14532D; font-weight: 600;">Normal-Demand Probability: <b>{(1 - pred_prob)*100:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(1 - pred_prob))
                st.markdown("""
                <div class="insight-box">
                    ✓ <b>Standard Operating Procedure:</b> Station operating within normal bounds. Off-peak energy credits eligible.
                </div>
                """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 4: TIME SERIES FORECASTING
    # =========================================================================
    elif app_mode == "⏳ Time-Series Forecasting":
        st.markdown("## ⏳ EV Charging Demand Time Series Forecasting")
        st.markdown("Long-term aggregate daily usage patterns, Augmented Dickey-Fuller stationarity test, and multi-step future demand forecasting.")

        ts_stats = metrics.get('timeseries_metrics', {}).get('Stationarity_ADF', {})
        ts_models = metrics.get('timeseries_metrics', {}).get('Models', {})

        # Time Series Proportional Metric Card Strip (ADF, 7D-MA, SES, Holt-Winters)
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-bottom: 20px;">
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #6366F1; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">ADF Stationarity Test</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #6366F1; margin: 2px 0;">p = {ts_stats.get('p_value', 0.3415):.4f}</div>
                <div style="font-size: 0.72rem; color: #F59E0B; font-weight: 600;">Non-Stationary (Stat: {ts_stats.get('ADF_Statistic', -1.88):.2f})</div>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #0284C7; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">7-Day Moving Average</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #0284C7; margin: 2px 0;">{ts_models.get('Moving Average (7-Day)', {}).get('MAE', 581.46):.2f} <span style="font-size: 0.8rem; color: #64748B;">kWh MAE</span></div>
                <div style="font-size: 0.72rem; color: #10B981; font-weight: 600;">RMSE: {ts_models.get('Moving Average (7-Day)', {}).get('RMSE', 892.01):.2f} kWh</div>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #EC4899; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Exponential Smoothing</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #EC4899; margin: 2px 0;">{ts_models.get('Simple Exponential Smoothing', {}).get('MAE', 620.66):.2f} <span style="font-size: 0.8rem; color: #64748B;">kWh MAE</span></div>
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">RMSE: {ts_models.get('Simple Exponential Smoothing', {}).get('RMSE', 925.48):.2f} kWh</div>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #10B981; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Holt-Winters Seasonal</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #10B981; margin: 2px 0;">{ts_models.get('Holt-Winters Exponential Smoothing', {}).get('MAE', 729.00):.2f} <span style="font-size: 0.8rem; color: #64748B;">kWh MAE</span></div>
                <div style="font-size: 0.72rem; color: #10B981; font-weight: 600;">Captures 7-day cyclical oscillations</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Historical Trend Line
        daily_ts['7D_MA'] = daily_ts['Total_Demand_kWh'].rolling(7).mean()
        daily_ts['30D_MA'] = daily_ts['Total_Demand_kWh'].rolling(30).mean()

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=daily_ts['Date'], y=daily_ts['Total_Demand_kWh'],
            name='Daily Actual Energy (kWh)',
            line=dict(color='#CBD5E1', width=1.2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Actual Demand: %{y:,.1f} kWh<extra></extra>"
        ))
        fig_ts.add_trace(go.Scatter(
            x=daily_ts['Date'], y=daily_ts['7D_MA'],
            name='7-Day Moving Average',
            line=dict(color='#0284C7', width=2.5),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>7D Rolling Avg: %{y:,.1f} kWh<extra></extra>"
        ))
        fig_ts.add_trace(go.Scatter(
            x=daily_ts['Date'], y=daily_ts['30D_MA'],
            name='30-Day Moving Average',
            line=dict(color='#F43F5E', width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>30D Rolling Avg: %{y:,.1f} kWh<extra></extra>"
        ))

        fig_ts.update_layout(
            title="366-Day Historical Daily Aggregate EV Energy Demand (kWh)",
            xaxis_title="Calendar Date",
            yaxis_title="Total Daily Energy Delivered (kWh)",
            yaxis=dict(tickformat=",.0f"),
            template='plotly_white', hovermode='x unified',
            margin=dict(t=30, b=40, l=10, r=10)
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Seasonal Decomposition Section (Trend, Seasonality, Residuals)
        if decomp_df is not None:
            st.markdown("### 🔍 Time-Series Decomposition (Trend, Seasonality & Noise)")
            st.markdown("Decomposed using an additive seasonal model with a 7-day cyclical weekly period:")
            
            fig_decomp = go.Figure()
            fig_decomp.add_trace(go.Scatter(x=decomp_df['Date'], y=decomp_df['Trend'], name='Underlying Growth Trend', line=dict(color='#0284C7', width=2.5)))
            fig_decomp.add_trace(go.Scatter(x=decomp_df['Date'], y=decomp_df['Seasonal'], name='Weekly Seasonality (± kWh)', line=dict(color='#10B981', width=1.5)))
            fig_decomp.add_trace(go.Scatter(x=decomp_df['Date'], y=decomp_df['Residual'], name='Irregular Noise / Residuals', line=dict(color='#94A3B8', width=1)))
            
            fig_decomp.update_layout(
                title="Decomposed Trend, Weekly Seasonality & Residual Components",
                xaxis_title="Date",
                yaxis_title="Energy Deviation (kWh)",
                template='plotly_white', hovermode='x unified',
                margin=dict(t=30, b=40, l=10, r=10)
            )
            st.plotly_chart(fig_decomp, use_container_width=True)

        # Future Demand Forecast
        st.markdown("### 🔮 Future Demand Forecasting (Holt-Winters Seasonal Smoothing)")
        horizon = st.slider("Select Forecast Horizon (Days)", min_value=7, max_value=30, value=14, step=1)
        
        if hw_model is not None:
            last_dt = daily_ts['Date'].max()
            future_index = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=horizon)
            future_values = hw_model.forecast(horizon)

            # Proportional Forecast Summary Metric Strip
            tot_proj = float(np.sum(future_values))
            daily_avg_proj = float(np.mean(future_values))
            peak_proj = float(np.max(future_values))
            peak_day_str = future_index[np.argmax(future_values)].strftime('%A, %b %d')

            st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 12px 0 16px 0;">
                <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 10px; padding: 12px 16px; text-align: center;">
                    <div style="font-size: 0.72rem; color: #166534; font-weight: 700; text-transform: uppercase;">Total Projected Demand ({horizon} Days)</div>
                    <div style="font-size: 1.55rem; font-weight: 800; color: #15803D; margin: 2px 0;">{tot_proj:,.0f} <span style="font-size: 0.8rem; color: #166534;">kWh</span></div>
                    <div style="font-size: 0.72rem; color: #166534; font-weight: 600;">Fleet-wide generation budget</div>
                </div>
                <div style="background: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 10px; padding: 12px 16px; text-align: center;">
                    <div style="font-size: 0.72rem; color: #075985; font-weight: 700; text-transform: uppercase;">Average Daily Demand</div>
                    <div style="font-size: 1.55rem; font-weight: 800; color: #0284C7; margin: 2px 0;">{daily_avg_proj:,.1f} <span style="font-size: 0.8rem; color: #075985;">kWh/day</span></div>
                    <div style="font-size: 0.72rem; color: #075985; font-weight: 600;">Expected 24-hr baseload</div>
                </div>
                <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 10px; padding: 12px 16px; text-align: center;">
                    <div style="font-size: 0.72rem; color: #92400E; font-weight: 700; text-transform: uppercase;">Expected Peak Load Day</div>
                    <div style="font-size: 1.55rem; font-weight: 800; color: #D97706; margin: 2px 0;">{peak_proj:,.0f} <span style="font-size: 0.8rem; color: #92400E;">kWh</span></div>
                    <div style="font-size: 0.72rem; color: #92400E; font-weight: 600;">Peak on <b>{peak_day_str}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            recent_slice = daily_ts.tail(45)
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=recent_slice['Date'], y=recent_slice['Total_Demand_kWh'],
                name='Recent Actuals (kWh)',
                line=dict(color='#0284C7', width=2.5),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Historical: %{y:,.1f} kWh<extra></extra>"
            ))
            fig_fc.add_trace(go.Scatter(
                x=future_index, y=future_values,
                name='Holt-Winters Seasonal Forecast',
                line=dict(color='#10B981', width=3, dash='dash'),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Projected Demand: %{y:,.1f} kWh<extra></extra>"
            ))

            fig_fc.update_layout(
                title=f"Next {horizon}-Day Predictive Energy Demand Outlook",
                xaxis_title="Forecast Date",
                yaxis_title="Projected Energy Demand (kWh)",
                yaxis=dict(tickformat=",.0f"),
                template='plotly_white',
                margin=dict(t=30, b=40, l=10, r=10)
            )
            st.plotly_chart(fig_fc, use_container_width=True)

        # Time-Series Model Benchmark Comparison Table
        st.markdown("#### 📊 Time-Series Model Performance Benchmark")
        ts_eval_df = pd.DataFrame(metrics['timeseries_metrics']['Models']).T
        st.dataframe(
            ts_eval_df.style.highlight_min(subset=['MAE', 'RMSE'], color='#BBF7D0').format({'MAE': '{:,.2f} kWh', 'RMSE': '{:,.2f} kWh'}),
            use_container_width=True
        )

    # =========================================================================
    # TAB 5: EXPLAINABLE AI (SHAP & LIME)
    # =========================================================================
    elif app_mode == "🔍 Explainable AI (SHAP & LIME)":
        st.markdown("## 🔍 Explainable AI: SHAP & LIME Transparency")
        st.markdown("Deciphering model decision boundaries globally with SHAP and locally with LIME for station managers and utility auditors.")

        # 1. SHAP Global Feature Importance
        st.markdown("### 1. Global Explainability: SHAP TreeExplainer Impact (kWh)")
        st.markdown("SHAP (SHapley Additive exPlanations) computes the exact contribution of each feature towards the expected energy demand in **real-world kilowatt-hours**:")
        
        if shap_imp is not None:
            shap_plot_df = pd.DataFrame(shap_imp).head(12)
            fig_shap = px.bar(
                shap_plot_df, x='Mean_SHAP_Impact_kWh', y='Feature', orientation='h',
                color='Mean_SHAP_Impact_kWh', color_continuous_scale='Blues',
                title="Top Features Influencing Energy Demand (Mean |SHAP| Impact in kWh)",
                labels={'Mean_SHAP_Impact_kWh': 'Mean Absolute SHAP Value (kWh)', 'Feature': 'Input Feature'},
                template='plotly_white'
            )
            fig_shap.update_traces(
                hovertemplate="<b>%{y}</b><br>SHAP Impact: %{x:.3f} kWh<extra></extra>"
            )
            fig_shap.update_layout(
                xaxis_title="Mean |SHAP Value| (Impact on Model Output in kWh)",
                yaxis_title="Engineered Input Feature",
                xaxis=dict(tickformat=".2f"),
                yaxis=dict(autorange='reversed'),
                margin=dict(t=30, b=40, l=10, r=10)
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(metrics['categorical_columns'])
        feature_names = list(cat_names) + metrics['numerical_columns']

        st.markdown("---")
        # 2. Local Session Explainer (LIME)
        st.markdown("### 2. Local Explainability: LIME Session Instance Breakdown")
        st.write("Inspect how individual session characteristics contributed towards driving demand up (+) or down (-) for a specific driver:")

        sample_id = st.slider("Select Sample Session from Audit Pool", 0, len(shap_sample)-1, 15)
        picked_session = shap_sample.iloc[[sample_id]]
        st.dataframe(picked_session[['Location', 'ChargerType', 'StartDatetime', 'Duration_Mins', 'Demand', 'High_Demand']], use_container_width=True)

        proc_sample = preprocessor.transform(shap_sample[metrics['categorical_columns'] + metrics['numerical_columns']])
        lime_explainer = LimeTabularExplainer(
            training_data=proc_sample,
            feature_names=feature_names,
            mode='regression',
            random_state=42
        )

        test_row = preprocessor.transform(picked_session[metrics['categorical_columns'] + metrics['numerical_columns']])
        exp_res = lime_explainer.explain_instance(test_row[0], reg_model.predict, num_features=7)

        lime_data = pd.DataFrame(exp_res.as_list(), columns=['Feature Condition', 'Effect Weight'])
        lime_data['Impact'] = lime_data['Effect Weight'].apply(lambda w: 'Increases Demand (+)' if w > 0 else 'Decreases Demand (-)')

        fig_l = px.bar(
            lime_data, x='Effect Weight', y='Feature Condition', orientation='h',
            color='Impact',
            color_discrete_map={'Increases Demand (+)': '#10B981', 'Decreases Demand (-)': '#EF4444'},
            title=f"LIME Feature Attribution for Session #{sample_id} (Actual Demand: {picked_session['Demand'].values[0]:.2f} kWh)",
            labels={'Effect Weight': 'Impact on Predicted Energy (kWh)', 'Feature Condition': 'Decision Rule'},
            template='plotly_white'
        )
        fig_l.update_traces(
            hovertemplate="<b>%{y}</b><br>Attribution: %{x:+.2f} kWh<extra></extra>"
        )
        fig_l.update_layout(
            xaxis_title="Attribution Effect on Energy Demand (kWh)",
            yaxis_title="Decision Rule Condition",
            xaxis=dict(tickformat="+.2f"),
            yaxis=dict(autorange='reversed'),
            margin=dict(t=30, b=40, l=10, r=10)
        )
        st.plotly_chart(fig_l, use_container_width=True)

    # =========================================================================
    # TAB 6: MODEL BENCHMARK LEADERBOARD
    # =========================================================================
    elif app_mode == "🏆 Model Leaderboard":
        st.markdown("## 🏆 Machine Learning Model Evaluation Leaderboard")
        st.markdown("Comprehensive performance metrics evaluated on isolated test splits as prescribed in the project rubric.")

        st.markdown("### 1. Regression Models Benchmark (Target: `Demand` in kWh)")
        reg_table = pd.DataFrame(metrics['regression_metrics']).T.sort_values(by='R2', ascending=False)
        st.dataframe(
            reg_table.style.highlight_max(subset=['R2'], color='#BBF7D0').highlight_min(subset=['MAE', 'RMSE'], color='#BBF7D0').format({
                'MAE': '{:.3f} kWh', 'MSE': '{:.2f}', 'RMSE': '{:.3f} kWh', 'R2': '{:.4f}'
            }),
            use_container_width=True
        )

        fig_rb = px.bar(
            reg_table.reset_index().rename(columns={'index': 'Model Name'}),
            x='Model Name', y='R2',
            color='MAE', color_continuous_scale='Blues_r',
            title="Regression Model Variance Explained (R² Score) & Mean Absolute Error (MAE)",
            labels={'R2': 'R² Score (0.0 to 1.0)', 'MAE': 'MAE (kWh)', 'Model Name': 'Algorithm'},
            template='plotly_white'
        )
        fig_rb.update_traces(hovertemplate="<b>%{x}</b><br>R² Score: %{y:.4f}<br>MAE: %{marker.color:.3f} kWh<extra></extra>")
        fig_rb.update_layout(yaxis_title="R² Score (Higher is Better)", yaxis=dict(range=[0, 1.0], tickformat=".2f"), margin=dict(t=30, b=40, l=10, r=10))
        st.plotly_chart(fig_rb, use_container_width=True)

        st.markdown("---")
        st.markdown("### 2. Classification Models Benchmark (Target: `High_Demand` ≥ 14.20 kWh)")
        clf_table = pd.DataFrame(metrics['classification_metrics']).T.sort_values(by='F1', ascending=False)
        st.dataframe(
            clf_table.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC'], color='#BBF7D0').format({
                'Accuracy': '{:.2%}', 'Precision': '{:.2%}', 'Recall': '{:.2%}', 'F1': '{:.4f}', 'ROC_AUC': '{:.4f}'
            }),
            use_container_width=True
        )

        fig_cb = px.bar(
            clf_table.reset_index().rename(columns={'index': 'Model Name'}),
            x='Model Name', y='F1',
            color='ROC_AUC', color_continuous_scale='Teal',
            title="Classification Model Benchmark: F1-Score & ROC-AUC",
            labels={'F1': 'F1 Score (0.0 to 1.0)', 'ROC_AUC': 'ROC-AUC Score', 'Model Name': 'Algorithm'},
            template='plotly_white'
        )
        fig_cb.update_traces(hovertemplate="<b>%{x}</b><br>F1 Score: %{y:.4f}<br>ROC-AUC: %{marker.color:.4f}<extra></extra>")
        fig_cb.update_layout(yaxis_title="F1-Score (Higher is Better)", yaxis=dict(range=[0, 1.0], tickformat=".2f"), margin=dict(t=30, b=40, l=10, r=10))
        st.plotly_chart(fig_cb, use_container_width=True)

        st.markdown("---")
        st.markdown("### 3. Class Imbalance Mitigation Benchmark (ADASYN vs SMOTE vs ROS)")
        imb_table = pd.DataFrame(metrics['imbalance_comparison']).T
        st.dataframe(
            imb_table.style.highlight_max(subset=['Recall', 'F1', 'ROC_AUC'], color='#BBF7D0').format({
                'Accuracy': '{:.2%}', 'Precision': '{:.2%}', 'Recall': '{:.2%}', 'F1': '{:.4f}', 'ROC_AUC': '{:.4f}'
            }),
            use_container_width=True
        )

        st.markdown("---")
        st.markdown("### 4. Time-Series Forecasting Evaluation (Daily Fleet Demand)")
        ts_table = pd.DataFrame(metrics['timeseries_metrics']['Models']).T
        st.dataframe(
            ts_table.style.highlight_min(subset=['MAE', 'RMSE'], color='#BBF7D0').format({
                'MAE': '{:,.2f} kWh', 'RMSE': '{:,.2f} kWh'
            }),
            use_container_width=True
        )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94A3B8; font-size: 0.8rem; padding: 12px 0;">
        ⚡ <b>VoltPulse EV Intelligence</b> | GUVI & HCL Sustainable Transportation AI Capstone | Developed by <b>Rajuzet</b>
    </div>
    """, unsafe_allow_html=True)
