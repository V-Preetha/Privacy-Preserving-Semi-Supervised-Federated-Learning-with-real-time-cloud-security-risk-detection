"""Real-time telemetry generator producing realistic varying data every second."""
from datetime import datetime, timedelta
import random
import numpy as np

class RealtimeTelemetryGenerator:
    """Generates realistic telemetry with natural variation and occasional anomalies.
    
    Normal ranges:
    - API calls: 10–40
    - CPU usage: 30–70%
    - Memory usage: 40–80%
    - Network traffic: 100–400 MB
    
    Anomalies (5-10% probability):
    - API spike: 80–200
    - CPU spike: 85–100
    - Network flood: 700–1200
    - Memory leak: 90–100
    """
    
    def __init__(self, anomaly_prob: float = 0.07, use_seed: bool = False):
        """Initialize generator.
        
        Args:
            anomaly_prob: Probability of anomaly per second (0.07 = ~7%)
            use_seed: If True, use fixed seed for reproducibility. If False, use random seed.
        """
        self.anomaly_prob = anomaly_prob
        self.current_time = datetime.utcnow()
        
        # Optional: use seed for reproducibility
        if use_seed:
            random.seed(42)
            np.random.seed(42)
        # else: random state is not fixed, values will vary

    def next_record(self):
        """Generate the next telemetry record (one-second advance).
        
        Returns:
            dict with keys:
            - timestamp: datetime object
            - event_type: "normal" or "anomaly"
            - api_calls: int (API call count)
            - cpu_usage: float (CPU percentage, 0-100)
            - memory_usage: float (Memory percentage, 0-100)
            - network_traffic: float (MB sent/received)
            - anomaly_reason: str or None
        """
        self.current_time += timedelta(seconds=1)
        is_anomaly = random.random() < self.anomaly_prob
        anomaly_reason = None
        event_type = "normal"

        if not is_anomaly:
            # Normal behavior with realistic variation
            api_calls = int(np.random.uniform(10, 40))
            cpu_usage = float(np.random.uniform(30, 70))
            memory_usage = float(np.random.uniform(40, 80))
            network_traffic = float(np.random.uniform(100, 400))
        else:
            # Inject anomaly
            event_type = "anomaly"
            anomaly_type = random.choice(['api_spike', 'cpu_spike', 'network_flood', 'memory_leak'])
            
            if anomaly_type == 'api_spike':
                api_calls = int(np.random.uniform(80, 200))
                cpu_usage = float(np.random.uniform(40, 75))
                memory_usage = float(np.random.uniform(50, 85))
                network_traffic = float(np.random.uniform(200, 600))
                anomaly_reason = 'API spike detected'
                
            elif anomaly_type == 'cpu_spike':
                api_calls = int(np.random.uniform(10, 30))
                cpu_usage = float(np.random.uniform(85, 100))
                memory_usage = float(np.random.uniform(60, 95))
                network_traffic = float(np.random.uniform(100, 300))
                anomaly_reason = 'High CPU usage detected'
                
            elif anomaly_type == 'network_flood':
                api_calls = int(np.random.uniform(50, 150))
                cpu_usage = float(np.random.uniform(50, 85))
                memory_usage = float(np.random.uniform(55, 90))
                network_traffic = float(np.random.uniform(700, 1200))
                anomaly_reason = 'Network flood detected'
                
            else:  # memory_leak
                api_calls = int(np.random.uniform(20, 50))
                cpu_usage = float(np.random.uniform(55, 80))
                memory_usage = float(np.random.uniform(90, 100))
                network_traffic = float(np.random.uniform(200, 500))
                anomaly_reason = 'Memory leak detected'

        record = {
            'timestamp': self.current_time,
            'event_type': event_type,
            'api_calls': api_calls,
            'cpu_usage': float(cpu_usage),
            'memory_usage': float(memory_usage),
            'network_traffic': float(network_traffic),
            'anomaly_reason': anomaly_reason,
        }
        return record
