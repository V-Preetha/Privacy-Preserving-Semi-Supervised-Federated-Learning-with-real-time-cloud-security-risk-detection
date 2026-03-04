"""Streamlit dashboard for live PP-SSFL demo with realistic telemetry and dynamic risk.

Run with:
    streamlit run streamlit_dashboard.py

This dashboard:
- Generates realistic telemetry every second (values vary continuously)
- Calculates risk dynamically: (api*0.3 + cpu*0.25 + net*0.25 + mem*0.2) / scale
- Shows a Plotly risk gauge (0-100 scale)
- Displays live telemetry table (last 20 records)
- Plots risk trend (last 100 points)
- Shows alerts and anomalies in real-time
- Displays proper evaluation metrics with class distribution
"""
import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from telemetry_generator import RealtimeTelemetryGenerator
from evaluation import train_and_evaluate

st.set_page_config(layout='wide', page_title='PP-SSFL Live Risk Monitor', initial_sidebar_state='collapsed')

# ============================================================================
# RISK CALCULATION FUNCTION
# ============================================================================
def calculate_risk(api_calls, cpu_usage, memory_usage, network_traffic):
    """Calculate risk score (0-100) from telemetry metrics.
    
    Formula: risk = (api*0.3 + cpu*0.25 + net*0.25 + mem*0.2) / scale
    
    Args:
        api_calls: int (10-200)
        cpu_usage: float (0-100)
        memory_usage: float (0-100)
        network_traffic: float (100-1200 MB)
    
    Returns:
        float: risk score (0-100)
    """
    # Normalize api_calls: 0-200 range -> map to 0-100
    api_normalized = min(api_calls / 2.0, 100)
    
    # CPU is already 0-100
    cpu_normalized = cpu_usage
    
    # Memory is already 0-100
    mem_normalized = memory_usage
    
    # Normalize network traffic: 0-1200 range -> map to 0-100
    net_normalized = min(network_traffic / 12.0, 100)
    
    # Weighted sum
    raw_risk = (
        api_normalized * 0.3 +
        cpu_normalized * 0.25 +
        net_normalized * 0.25 +
        mem_normalized * 0.2
    )
    
    # Scale to 0-100 (normalize by number of metrics)
    risk_score = min(raw_risk, 100.0)
    return risk_score


def get_risk_color(risk_score):
    """Get color based on risk level."""
    if risk_score < 25:
        return '#00cc00'  # Green
    elif risk_score < 50:
        return '#ffaa00'  # Orange
    elif risk_score < 75:
        return '#ff6600'  # Dark Orange
    else:
        return '#cc0000'  # Red


def get_risk_level(risk_score):
    """Get risk level name."""
    if risk_score < 25:
        return 'LOW'
    elif risk_score < 50:
        return 'MEDIUM'
    elif risk_score < 75:
        return 'HIGH'
    else:
        return 'CRITICAL'


# ============================================================================
# PLOTLY GAUGE CHART
# ============================================================================
def create_gauge(risk_score):
    """Create a Plotly gauge chart for risk visualization."""
    color = get_risk_color(risk_score)
    level = get_risk_level(risk_score)
    
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=risk_score,
        title={'text': f'Risk Level: {level}'},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 25], 'color': 'rgba(0, 255, 0, 0.1)'},
                {'range': [25, 50], 'color': 'rgba(255, 170, 0, 0.1)'},
                {'range': [50, 75], 'color': 'rgba(255, 102, 0, 0.1)'},
                {'range': [75, 100], 'color': 'rgba(204, 0, 0, 0.1)'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        },
        number={'suffix': ' / 100'},
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ============================================================================
# SESSION STATE AND CACHING
# ============================================================================
@st.cache_resource
def init_generator():
    """Initialize telemetry generator (no fixed seed = realistic variation)."""
    return RealtimeTelemetryGenerator(anomaly_prob=0.07, use_seed=False)

@st.cache_resource
def load_trained_model():
    """Load pre-trained model with proper evaluation metrics."""
    print("[Streamlit] Training evaluation model...")
    results = train_and_evaluate(
        n_samples=500,
        anomaly_prob=0.07,
        test_size=0.2,
        hidden_dim=64,
        epochs=10,
        verbose=False
    )
    return results['model'], results['metrics'], results['y_test']

generator = init_generator()
model, eval_metrics, y_test_for_metrics = load_trained_model()

if 'running' not in st.session_state:
    st.session_state.running = False
if 'history' not in st.session_state:
    st.session_state.history = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'telemetry' not in st.session_state:
    st.session_state.telemetry = []
if 'record_count' not in st.session_state:
    st.session_state.record_count = 0


# ============================================================================
# PAGE LAYOUT
# ============================================================================
st.title('🔒 PP-SSFL — Live Risk Monitoring Dashboard')
st.markdown('**Real-time cloud safety risk prediction with continuous telemetry analysis**')

# ============================================================================
# METRICS PANEL (WITH CLASS DISTRIBUTION)
# ============================================================================
st.markdown('### 📊 Model Evaluation Metrics')
with st.expander('Click to expand metrics and class distribution', expanded=True):
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    
    with metric_col1:
        val = eval_metrics.get('accuracy', 'N/A')
        if isinstance(val, float):
            st.metric('Accuracy', f"{val:.4f}")
        else:
            st.metric('Accuracy', val)
    
    with metric_col2:
        val = eval_metrics.get('precision', 'N/A')
        if isinstance(val, float):
            st.metric('Precision', f"{val:.4f}")
        else:
            st.metric('Precision', val)
    
    with metric_col3:
        val = eval_metrics.get('recall', 'N/A')
        if isinstance(val, float):
            st.metric('Recall', f"{val:.4f}")
        else:
            st.metric('Recall', val)
    
    with metric_col4:
        val = eval_metrics.get('f1', 'N/A')
        if isinstance(val, float):
            st.metric('F1 Score', f"{val:.4f}")
        else:
            st.metric('F1 Score', val)
    
    with metric_col5:
        val = eval_metrics.get('roc_auc', 'N/A')
        if isinstance(val, float):
            st.metric('ROC-AUC', f"{val:.4f}")
        elif val is None:
            st.metric('ROC-AUC', 'N/A', delta='Only 1 class')
        else:
            st.metric('ROC-AUC', val)
    
    # Class distribution
    st.markdown('**Test Set Class Distribution:**')
    class_col1, class_col2 = st.columns(2)
    with class_col1:
        n_normal = len([y for y in y_test_for_metrics if y == 0])
        pct_normal = 100 * n_normal / len(y_test_for_metrics)
        st.metric('Normal Samples', f'{n_normal} ({pct_normal:.1f}%)')
    with class_col2:
        n_anomaly = len([y for y in y_test_for_metrics if y == 1])
        pct_anomaly = 100 * n_anomaly / len(y_test_for_metrics)
        st.metric('Anomaly Samples', f'{n_anomaly} ({pct_anomaly:.1f}%)')

st.divider()

# Control panel
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 2, 1])
with col_ctrl1:
    if st.button('▶ START', key='btn_start', use_container_width=True):
        st.session_state.running = True
        st.rerun()
    if st.button('⏸ STOP', key='btn_stop', use_container_width=True):
        st.session_state.running = False
        st.rerun()

with col_ctrl2:
    status_symbol = '🟢 MONITORING' if st.session_state.running else '🔴 STOPPED'
    st.markdown(f'### {status_symbol}')

with col_ctrl3:
    st.metric('Records Processed', st.session_state.record_count)

st.divider()

# Main layout
gauge_col, metrics_col = st.columns([1.5, 1])

# ============================================================================
# MAIN LOOP: Generate and display telemetry
# ============================================================================
if st.session_state.running:
    # Generate new record
    record = generator.next_record()
    
    # Calculate risk
    risk = calculate_risk(
        api_calls=record['api_calls'],
        cpu_usage=record['cpu_usage'],
        memory_usage=record['memory_usage'],
        network_traffic=record['network_traffic']
    )
    
    # Store in history
    history_entry = {
        'timestamp': record['timestamp'],
        'risk': risk,
        'event_type': record['event_type'],
        'anomaly_reason': record['anomaly_reason']
    }
    st.session_state.history.append(history_entry)
    
    # Store in telemetry
    telemetry_row = {
        '⏱ TimeStamp': record['timestamp'].strftime('%H:%M:%S'),
        '🔴 Event': record['event_type'].upper(),
        '📡 API Calls': record['api_calls'],
        '💻 CPU (%)': round(record['cpu_usage'], 1),
        '🧠 Memory (%)': round(record['memory_usage'], 1),
        '📊 Net Traffic (MB)': round(record['network_traffic'], 1),
        '⚠️ Risk Score': round(risk, 1),
    }
    st.session_state.telemetry.insert(0, telemetry_row)
    
    # Create alerts for anomalies or high risk
    if record['event_type'] == 'anomaly' or risk >= 50:
        alert_entry = {
            'timestamp': record['timestamp'].strftime('%H:%M:%S'),
            'risk': risk,
            'reason': record['anomaly_reason'] or 'Risk threshold exceeded',
            'event_type': record['event_type']
        }
        st.session_state.alerts.insert(0, alert_entry)
    
    # Maintain buffer sizes
    st.session_state.record_count += 1
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[-100:]
    if len(st.session_state.telemetry) > 20:
        st.session_state.telemetry = st.session_state.telemetry[:20]
    if len(st.session_state.alerts) > 50:
        st.session_state.alerts = st.session_state.alerts[:50]
    
    # Display gauge
    with gauge_col:
        st.plotly_chart(create_gauge(risk), use_container_width=True, height=350)
    
    # Display key metrics
    with metrics_col:
        st.markdown('### 📊 Current Metrics')
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric('API Calls', record['api_calls'])
            st.metric('Memory %', f"{record['memory_usage']:.1f}")
        with col_m2:
            st.metric('CPU %', f"{record['cpu_usage']:.1f}")
            st.metric('Net MB', f"{record['network_traffic']:.1f}")
    
    st.divider()
    
    # Telemetry table
    st.markdown('### 📋 Live Telemetry Stream (Last 20 Records)')
    if st.session_state.telemetry:
        df_telemetry = pd.DataFrame(st.session_state.telemetry)
        st.dataframe(df_telemetry, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Risk trend chart
    st.markdown('### 📈 Risk Trend (Last 100 Points)')
    if len(st.session_state.history) > 1:
        df_trend = pd.DataFrame([
            {
                'point': i,
                'risk': entry['risk'],
                'event': entry['event_type']
            }
            for i, entry in enumerate(st.session_state.history)
        ])
        
        # Create line chart with anomalies highlighted
        fig_trend = go.Figure()
        
        # Normal points
        normal_data = df_trend[df_trend['event'] == 'normal']
        if not normal_data.empty:
            fig_trend.add_trace(go.Scatter(
                x=normal_data['point'],
                y=normal_data['risk'],
                mode='lines+markers',
                name='Normal',
                line=dict(color='#00cc00', width=2),
                marker=dict(size=4)
            ))
        
        # Anomaly points
        anomaly_data = df_trend[df_trend['event'] == 'anomaly']
        if not anomaly_data.empty:
            fig_trend.add_trace(go.Scatter(
                x=anomaly_data['point'],
                y=anomaly_data['risk'],
                mode='markers',
                name='Anomaly',
                marker=dict(color='#cc0000', size=10, symbol='diamond')
            ))
        
        # Add threshold line
        fig_trend.add_hline(y=50, line_dash='dash', line_color='orange', annotation_text='Threshold')
        
        fig_trend.update_layout(
            title='Risk Score Over Time',
            xaxis_title='Record #',
            yaxis_title='Risk Score',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    st.divider()
    
    # Alerts panel
    col_alerts, col_stats = st.columns([1, 1])
    
    with col_alerts:
        st.markdown(f'### 🚨 Alerts ({len(st.session_state.alerts)})')
        if st.session_state.alerts:
            for alert in st.session_state.alerts[:8]:
                risk_val = alert['risk']
                color = get_risk_color(risk_val)
                st.markdown(f"""
                <div style="padding:12px; margin:8px 0; border-left:5px solid {color}; background:#ffffff; border-radius:4px;">
                    <b style="color:{color};">{alert['reason']}</b><br/>
                    <small style="color:{color};">⏱ {alert['timestamp']} | 📊 Score: {risk_val:.1f} | Event: {alert['event_type'].upper()}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info('✓ No alerts in this session')
    
    with col_stats:
        st.markdown('### 📊 Session Stats')
        total_records = st.session_state.record_count
        total_anomalies = len([h for h in st.session_state.history if h['event_type'] == 'anomaly'])
        avg_risk = sum([h['risk'] for h in st.session_state.history]) / len(st.session_state.history) if st.session_state.history else 0
        max_risk = max([h['risk'] for h in st.session_state.history], default=0)
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric('Total Records', total_records)
            st.metric('Anomalies Detected', total_anomalies)
        with col_s2:
            st.metric('Average Risk', f"{avg_risk:.1f}")
            st.metric('Max Risk', f"{max_risk:.1f}")
    
    # Auto-rerun every 1 second
    time.sleep(1)
    st.rerun()

else:
    st.info('👈 Press **▶ START** to begin live telemetry monitoring and risk detection')
    st.markdown("""
    #### About This Dashboard
    - **Telemetry**: Generates realistic cloud metrics every second (API calls, CPU, Memory, Network)
    - **Risk Calculation**: Dynamic formula combining weighted metrics (0-100 scale)
    - **Anomalies**: Random injection of suspicious events (API spikes, CPU/Memory leaks, DDoS patterns)
    - **Alerts**: Automatic detection and flagging of high-risk events (threshold: > 50)
    - **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC (when both classes present)
    """)
