import requests
import time
import random
import json

API_URL = "http://localhost:8000/predict"

def send_transaction(drift=False):
    if not drift:
        # Normal transaction
        payload = {
            "TransactionAmt": random.uniform(10, 500),
            "ProductCD": random.randint(1, 4),
            "card1": random.uniform(1000, 15000),
            "card2": random.uniform(100, 600),
            "addr1": random.uniform(100, 500),
            "P_emaildomain": random.randint(1, 20),
            "DeviceType": random.randint(1, 2),
            "DeviceInfo": random.randint(1, 100)
        }
    else:
        # DRIFTED transaction: Huge amounts and suspicious cards
        payload = {
            "TransactionAmt": random.uniform(5000, 20000), # Unusual high amounts
            "ProductCD": 9, # New product code not seen in training
            "card1": 99999.0, # Impossible card ID
            "card2": 999.0,
            "addr1": 999.0,
            "P_emaildomain": 99,
            "DeviceType": 9,
            "DeviceInfo": 999
        }
    
    try:
        response = requests.post(API_URL, json=payload)
        print(f"[{'DRIFT' if drift else 'NORMAL'}] Status: {response.status_code}, Prob: {response.json().get('fraud_probability'):.4f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Starting Drift Simulation...")
    print("Phase 1: Sending Normal Traffic (10 seconds)...")
    for _ in range(10):
        send_transaction(drift=False)
        time.sleep(1)
        
    print("\nPhase 2: INTRODUCING DATA DRIFT! (Suspicious activity spike)...")
    for _ in range(20):
        send_transaction(drift=True)
        time.sleep(0.5)
    
    print("\nDrift Simulation Complete. Check your Grafana dashboard and Prometheus alerts!")
