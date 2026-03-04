#!/usr/bin/env python
"""Quick test to verify dashboard dependencies work."""
import sys

try:
    from telemetry_generator import RealtimeTelemetryGenerator
    from evaluation import train_and_evaluate
    import streamlit as st
    import plotly.graph_objects as go
    
    print("✓ All imports successful")
    
    # Quick test of telemetry generator
    gen = RealtimeTelemetryGenerator(anomaly_prob=0.07, use_seed=False)
    record = gen.next_record()
    print(f"✓ Telemetry works: API={record['api_calls']}, CPU={record['cpu_usage']:.1f}%")
    
    # Verify evaluate function exists
    print(f"✓ train_and_evaluate function available: {callable(train_and_evaluate)}")
    
    print("\n✅ Dashboard ready to run!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
