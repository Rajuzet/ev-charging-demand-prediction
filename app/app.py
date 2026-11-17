"""
Streamlit Web Application: EV Charging Demand Prediction and Usage Pattern Analysis
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# SHAP and LIME for XAI tab
import shap
from lime.lime_tabular import LimeTabularExplainer

# Page configuration
st.set_page_config(
    page_title="EV Charging Demand & Usage Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1F2937;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_csv = os.path.join(base_dir, 'data', 'processed', 'cleaned_charging_records.csv')
    df = pd.read_csv(clean_csv)
    df['StartDatetime'] = pd.to_datetime(df['StartDatetime'])
    df['EndDatetime'] = pd.to_datetime(df['EndDatetime'])
    return df


@st.cache_resource
def load_artifacts():
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

    return reg_model, clf_model, preprocessor, metrics, daily_ts, shap_sample


# Load assets
try:
    df_raw = load_data()
    reg_model, clf_model, preprocessor, metrics, daily_ts, shap_sample = load_artifacts()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading project artifacts: {e}")
    data_loaded = False

if data_loaded:
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/ev-station.png", width=70)
    st.sidebar.title("EV Demand Analytics")
    st.sidebar.markdown("**Domain:** Sustainable Transportation & EV")

    nav_tab = st.sidebar.radio(
        "Navigation",
        [
            "📊 Executive KPIs & EDA",
            "⚡ Demand Prediction (Regression)",
            "🎯 High-Demand Classifier",
            "⏳ Time Series Forecasting",
            "🔍 Explainable AI (SHAP & LIME)",
            "🏆 Model Evaluation Benchmark"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Filters")
    all_locations = sorted(df_raw['Location'].unique())
    selected_locations = st.sidebar.multiselect("Select Location", all_locations, default=all_locations)
    
    all_types = sorted(df_raw['ChargerType'].unique())
    selected_types = st.sidebar.multiselect("Charger Type", all_types, default=all_types)

    # Filtered dataframe
    df_filtered = df_raw[
        (df_raw['Location'].isin(selected_locations)) &
        (df_raw['ChargerType'].isin(selected_types))
    ]

    # TAB 1: EXECUTIVE KPIS & EDA
    if nav_tab == "📊 Executive KPIs & EDA":
        st.markdown('<div class="main-header">⚡ EV Charging Usage Patterns & KPI Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Comprehensive exploratory analysis and real-time operational metrics across charging stations.</div>', unsafe_allow_html=True)

        # Top KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric("Total Sessions", f"{len(df_filtered):,}")
        with k2:
            st.metric("Total Energy (kWh)", f"{df_filtered['Demand'].sum():,.0f}")
        with k3:
            st.metric("Avg Demand / Session", f"{df_filtered['Demand'].mean():.2f} kWh")
        with k4:
            st.metric("Avg Session Duration", f"{df_filtered['Duration_Mins'].mean():.1f} min")
        with k5:
            high_pct = (df_filtered['High_Demand'].mean()) * 100
            st.metric("High-Demand Ratio", f"{high_pct:.1f}%")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Charging Demand by Location")
            loc_summary = df_filtered.groupby('Location')['Demand'].agg(['count', 'sum', 'mean']).reset_index()
            loc_summary = loc_summary.sort_values(by='sum', ascending=False)
            fig_loc = px.bar(
                loc_summary, x='Location', y='sum',
                title="Total Energy Delivered (kWh) by Location",
                labels={'sum': 'Total kWh', 'Location': 'Location Category'},
                color='mean', color_continuous_scale='Blues'
            )
            fig_loc.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_loc, use_container_width=True)

        with c2:
            st.subheader("Charger Type Demand Distribution")
            fig_type = px.pie(
                df_filtered, names='ChargerType', values='Demand',
                title="Energy Share by Charger Type (0 = Standard, 1 = Fast)",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4
            )
            st.plotly_chart(fig_type, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Peak Charging Hours Demand Profile")
            hourly = df_filtered.groupby('StartHour')['Demand'].agg(['count', 'mean']).reset_index()
            fig_hour = go.Figure()
            fig_hour.add_trace(go.Bar(
                x=hourly['StartHour'], y=hourly['count'], name='Session Count',
                marker_color='#93C5FD', yaxis='y1'
            ))
            fig_hour.add_trace(go.Scatter(
                x=hourly['StartHour'], y=hourly['mean'], name='Mean Demand (kWh)',
                line=dict(color='#1D4ED8', width=3), yaxis='y2'
            ))
            fig_hour.update_layout(
                title="Sessions & Mean Demand by Hour of Day",
                xaxis=dict(title="Hour of Day (0 - 23)", tickmode='linear'),
                yaxis=dict(title="Number of Sessions"),
                yaxis2=dict(title="Average Demand (kWh)", overlaying='y', side='right'),
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig_hour, use_container_width=True)

        with c4:
            st.subheader("Day of Week Usage Pattern")
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_agg = df_filtered.groupby('DayName')['Demand'].agg(['count', 'mean']).reindex(days_order).reset_index()
            fig_day = px.bar(
                day_agg, x='DayName', y='count',
                title="Total Sessions by Day of Week",
                color='mean', color_continuous_scale='Viridis',
                labels={'count': 'Sessions', 'DayName': 'Day of Week', 'mean': 'Avg kWh'}
            )
            st.plotly_chart(fig_day, use_container_width=True)

        st.subheader("Duration vs. Demand Correlation")
        fig_scatter = px.scatter(
            df_filtered.sample(min(3000, len(df_filtered)), random_state=42),
            x='Duration_Mins', y='Demand', color='Location',
            opacity=0.6, title="Demand (kWh) vs Duration (Minutes) Sample",
            labels={'Duration_Mins': 'Duration (minutes)', 'Demand': 'Demand (kWh)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # TAB 2: REGRESSION PREDICTION
    elif nav_tab == "⚡ Demand Prediction (Regression)":
        st.markdown('<div class="main-header">⚡ EV Charging Demand Predictor (Regression)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">Trained with the best model: <b>{metrics.get("best_regression_model", "XGBoost")}</b>. Predict exact session energy consumption in kWh.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Session Input Parameters")
            in_loc = st.selectbox("Location Category", all_locations)
            in_company = st.selectbox("Charger Company", [0, 1], help="Operator / Brand ID")
            in_type = st.selectbox("Charger Type", [0, 1], help="0: Standard AC, 1: Fast DC")
            in_duration = st.slider("Charging Duration (minutes)", min_value=5, max_value=720, value=90, step=5)
            
            c_h1, c_h2 = st.columns(2)
            with c_h1:
                in_hour = st.slider("Start Hour", 0, 23, 14)
                in_month = st.slider("Month", 1, 12, 6)
            with c_h2:
                in_day = st.slider("Day of Month", 1, 31, 15)
                in_dow = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])

            is_weekend = 1 if in_dow in [5, 6] else 0
            
            # Time of day
            if 5 <= in_hour < 12:
                tod = 'Morning'
            elif 12 <= in_hour < 17:
                tod = 'Afternoon'
            elif 17 <= in_hour < 22:
                tod = 'Evening'
            else:
                tod = 'Night'

            # Historical baseline
            loc_avg = df_raw[df_raw['Location'] == in_loc]['Demand'].mean()
            type_avg = df_raw[df_raw['ChargerType'] == in_type]['Demand'].mean()

            predict_btn = st.button("Predict Charging Demand", type="primary", use_container_width=True)

        with col2:
            st.subheader("Prediction Result & Insights")
            if predict_btn or True:
                # Prepare single-row dataframe matching features
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
                pred_demand = float(reg_model.predict(proc_input)[0])
                pred_demand = max(0.1, pred_demand)

                st.success(f"### Predicted Demand: **{pred_demand:.2f} kWh**")
                
                # Contextual gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pred_demand,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Predicted Demand vs Average (kWh)"},
                    delta={'reference': df_raw['Demand'].mean()},
                    gauge={
                        'axis': {'range': [0, 80]},
                        'bar': {'color': "#2563EB"},
                        'steps': [
                            {'range': [0, 14.2], 'color': "#E0F2FE"},
                            {'range': [14.2, 35], 'color': "#BAE6FD"},
                            {'range': [35, 80], 'color': "#7DD3FC"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 3},
                            'thickness': 0.75,
                            'value': metrics.get('demand_median_threshold', 14.2)
                        }
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.info(f"""
                - **Session Type:** {tod} charging ({'Weekend' if is_weekend else 'Weekday'})
                - **Location Typical Demand:** {loc_avg:.2f} kWh
                - **Effective Power Proxy:** {(pred_demand / (in_duration/60.0)):.2f} kW
                """)

    # TAB 3: CLASSIFICATION
    elif nav_tab == "🎯 High-Demand Classifier":
        st.markdown('<div class="main-header">🎯 High-Demand Session Classifier</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">Trained with: <b>{metrics.get("best_classification_model", "LightGBM")}</b>. Classifies session into <b>High Demand</b> (≥ {metrics.get("demand_median_threshold", 14.2)} kWh) or <b>Normal Demand</b>.</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Configure Session Parameters")
            cl_loc = st.selectbox("Select Station Location", all_locations, key="clf_loc")
            cl_company = st.selectbox("Charger Company", [0, 1], key="clf_comp")
            cl_type = st.selectbox("Charger Type", [0, 1], key="clf_type")
            cl_duration = st.slider("Duration (minutes)", 10, 600, 120, key="clf_dur")
            cl_hour = st.slider("Session Hour", 0, 23, 18, key="clf_hour")
            cl_dow = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], index=4, 
                                  format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x], key="clf_dow")

            is_wknd = 1 if cl_dow in [5, 6] else 0
            tod_clf = 'Morning' if 5 <= cl_hour < 12 else ('Afternoon' if 12 <= cl_hour < 17 else ('Evening' if 17 <= cl_hour < 22 else 'Night'))
            
            clf_btn = st.button("Classify Demand Category", type="primary", use_container_width=True)

        with c2:
            st.subheader("Classification Outcome")
            cl_input = pd.DataFrame([{
                'Location': cl_loc,
                'ChargerCompany': cl_company,
                'ChargerType': cl_type,
                'TimeOfDay': tod_clf,
                'IsWeekend': is_wknd,
                'Duration_Mins': cl_duration,
                'StartHour': cl_hour,
                'StartDay': 15,
                'StartMonth': 6,
                'DayOfWeek': cl_dow,
                'Hour_Sin': np.sin(2 * np.pi * cl_hour / 24.0),
                'Hour_Cos': np.cos(2 * np.pi * cl_hour / 24.0),
                'Loc_Avg_Demand': df_raw[df_raw['Location'] == cl_loc]['Demand'].mean(),
                'ChargerType_Avg_Demand': df_raw[df_raw['ChargerType'] == cl_type]['Demand'].mean()
            }])

            proc_cl = preprocessor.transform(cl_input)
            pred_class = clf_model.predict(proc_cl)[0]
            pred_prob = clf_model.predict_proba(proc_cl)[0][1]

            if pred_class == 1:
                st.error(f"### Result: **HIGH DEMAND SESSION**")
                st.write(f"Confidence (Probability of High Demand): **{pred_prob * 100:.1f}%**")
                st.progress(float(pred_prob))
                st.warning("⚠️ **Grid & Operator Recommendation:** High grid load expected. Suggest pre-allocating fast-charging priority and peak load balancing.")
            else:
                st.success(f"### Result: **NORMAL DEMAND SESSION**")
                st.write(f"Confidence (Probability of Normal Demand): **{(1 - pred_prob) * 100:.1f}%**")
                st.progress(float(1 - pred_prob))
                st.info("✅ **Operator Recommendation:** Standard capacity sufficient. Suitable for scheduled off-peak charging discounts.")

    # TAB 4: TIME SERIES FORECASTING
    elif nav_tab == "⏳ Time Series Forecasting":
        st.markdown('<div class="main-header">⏳ EV Charging Demand Time Series Forecasting</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Historical aggregate charging demand trends, Augmented Dickey-Fuller stationarity, and future forecasting using Moving Average & Exponential Smoothing.</div>', unsafe_allow_html=True)

        ts_stats = metrics.get('timeseries_metrics', {}).get('Stationarity_ADF', {})
        st.markdown(f"""
        > **Stationarity Check (Augmented Dickey-Fuller Test):**
        > - ADF Statistic: **{ts_stats.get('ADF_Statistic', 'N/A')}**
        > - p-value: **{ts_stats.get('p_value', 'N/A')}**
        > - Is Stationary: **{'Yes (p < 0.05)' if ts_stats.get('Is_Stationary') else 'No'}**
        """)

        # Calculate Moving Average
        daily_ts['7D_MA'] = daily_ts['Total_Demand_kWh'].rolling(window=7).mean()
        daily_ts['30D_MA'] = daily_ts['Total_Demand_kWh'].rolling(window=30).mean()

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=daily_ts['Date'], y=daily_ts['Total_Demand_kWh'], mode='lines', name='Actual Daily Demand', line=dict(color='#CBD5E1', width=1)))
        fig_ts.add_trace(go.Scatter(x=daily_ts['Date'], y=daily_ts['7D_MA'], mode='lines', name='7-Day Moving Average', line=dict(color='#2563EB', width=2.5)))
        fig_ts.add_trace(go.Scatter(x=daily_ts['Date'], y=daily_ts['30D_MA'], mode='lines', name='30-Day Moving Average', line=dict(color='#DC2626', width=2)))

        fig_ts.update_layout(
            title="Historical Daily Aggregate EV Charging Demand (kWh) & Moving Averages",
            xaxis_title="Date",
            yaxis_title="Total Energy Demand (kWh)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Forecast Section
        st.subheader("Future Demand Forecast (Holt-Winters Seasonal Smoothing)")
        forecast_days = st.slider("Forecast Horizon (Days)", 7, 30, 14)
        
        # Load timeseries model for forecast
        ts_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'timeseries_model.pkl')
        if os.path.exists(ts_model_path):
            hw_model = joblib.load(ts_model_path)
            last_date = daily_ts['Date'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
            future_preds = hw_model.forecast(forecast_days)

            fig_f = go.Figure()
            # Plot recent 60 days
            recent_ts = daily_ts.tail(60)
            fig_f.add_trace(go.Scatter(x=recent_ts['Date'], y=recent_ts['Total_Demand_kWh'], name='Recent Actuals', line=dict(color='#3B82F6', width=2)))
            fig_f.add_trace(go.Scatter(x=future_dates, y=future_preds, name='Holt-Winters Forecast', line=dict(color='#10B981', width=3, dash='dash')))

            fig_f.update_layout(
                title=f"Next {forecast_days}-Day Forecast vs Recent Actual Demand",
                xaxis_title="Date", yaxis_title="Total Energy Demand (kWh)"
            )
            st.plotly_chart(fig_f, use_container_width=True)

    # TAB 5: EXPLAINABLE AI
    elif nav_tab == "🔍 Explainable AI (SHAP & LIME)":
        st.markdown('<div class="main-header">🔍 Explainable AI: SHAP & LIME Interpretability</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Understand exactly which features influence predictions at global and individual session levels.</div>', unsafe_allow_html=True)

        st.subheader("Global Feature Importance (Tree Model)")
        if hasattr(reg_model, 'feature_importances_'):
            # Extract feature names from preprocessor
            cat_encoder = preprocessor.named_transformers_['cat']
            cat_features = cat_encoder.get_feature_names_out(metrics['categorical_columns'])
            all_features = list(cat_features) + metrics['numerical_columns']
            
            importances = reg_model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': all_features,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).head(15)

            fig_imp = px.bar(
                feat_imp_df, x='Importance', y='Feature', orientation='h',
                title="Top 15 Most Influential Features Driving Charging Demand",
                color='Importance', color_continuous_scale='Blues'
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("---")
        st.subheader("Local Instance Explanation (LIME)")
        st.write("Pick a test session from the dataset to explain why the model made its specific prediction:")

        sample_idx = st.slider("Select Sample Index from Evaluation Set", 0, len(shap_sample) - 1, 10)
        selected_row = shap_sample.iloc[[sample_idx]]
        st.write(selected_row[['Location', 'ChargerType', 'StartDatetime', 'Duration_Mins', 'Demand', 'High_Demand']])

        # LIME Tabular Explainer
        X_sample = shap_sample[metrics['categorical_columns'] + metrics['numerical_columns']]
        X_sample_proc = preprocessor.transform(X_sample)

        explainer = LimeTabularExplainer(
            training_data=X_sample_proc,
            feature_names=all_features,
            mode='regression',
            random_state=42
        )

        test_inst = preprocessor.transform(selected_row[metrics['categorical_columns'] + metrics['numerical_columns']])
        exp = explainer.explain_instance(test_inst[0], reg_model.predict, num_features=8)
        
        lime_list = exp.as_list()
        lime_df = pd.DataFrame(lime_list, columns=['Feature Condition', 'Contribution Weight'])
        lime_df['Direction'] = lime_df['Contribution Weight'].apply(lambda w: 'Increases Demand' if w > 0 else 'Decreases Demand')

        fig_lime = px.bar(
            lime_df, x='Contribution Weight', y='Feature Condition', orientation='h',
            color='Direction',
            color_discrete_map={'Increases Demand': '#10B981', 'Decreases Demand': '#EF4444'},
            title=f"LIME Explanation for Sample #{sample_idx} (Actual Demand: {selected_row['Demand'].values[0]} kWh)"
        )
        fig_lime.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_lime, use_container_width=True)

    # TAB 6: MODEL BENCHMARK
    elif nav_tab == "🏆 Model Evaluation Benchmark":
        st.markdown('<div class="main-header">🏆 Model Evaluation & Performance Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Complete benchmarking across Regression algorithms, Classification algorithms, and Class Imbalance mitigation strategies.</div>', unsafe_allow_html=True)

        st.subheader("1. Regression Models Comparison (Target: Demand)")
        reg_df = pd.DataFrame(metrics['regression_metrics']).T.sort_values(by='R2', ascending=False)
        st.dataframe(reg_df.style.highlight_max(axis=0, subset=['R2'], color='#BBF7D0').highlight_min(axis=0, subset=['MAE', 'RMSE'], color='#BBF7D0'), use_container_width=True)

        fig_reg_comp = px.bar(
            reg_df.reset_index().rename(columns={'index': 'Model'}),
            x='Model', y='R2', color='RMSE',
            title="Regression Model Comparison (R² Score & RMSE)",
            labels={'R2': 'R² Score (Higher is Better)', 'RMSE': 'RMSE (Lower is Better)'},
            color_continuous_scale='Teal'
        )
        st.plotly_chart(fig_reg_comp, use_container_width=True)

        st.markdown("---")
        st.subheader("2. Classification Models Comparison (Target: High Demand)")
        clf_df = pd.DataFrame(metrics['classification_metrics']).T.sort_values(by='F1', ascending=False)
        st.dataframe(clf_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC'], color='#BBF7D0'), use_container_width=True)

        fig_clf_comp = px.bar(
            clf_df.reset_index().rename(columns={'index': 'Model'}),
            x='Model', y='F1', color='ROC_AUC',
            title="Classification Model Benchmark (F1-Score & ROC-AUC)",
            labels={'F1': 'F1 Score', 'ROC_AUC': 'ROC-AUC'},
            color_continuous_scale='Purples'
        )
        st.plotly_chart(fig_clf_comp, use_container_width=True)

        st.markdown("---")
        st.subheader("3. Handling Class Imbalance Comparison (Baseline vs Resampling)")
        imb_df = pd.DataFrame(metrics['imbalance_comparison']).T
        st.dataframe(imb_df.style.highlight_max(axis=0, subset=['F1', 'ROC_AUC'], color='#BBF7D0'), use_container_width=True)
