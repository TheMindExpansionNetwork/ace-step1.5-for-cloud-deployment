import requests
import json
import time

SERVER_URL = "http://localhost:8000"

def test_health():
    print(f"Checking health at {SERVER_URL}...")
    try:
        resp = requests.get(f"{SERVER_URL}/health")
        resp.raise_for_status()
        data = resp.json()
        print(f"✅ Health OK: {data}")
        return data.get("dit_loaded", False)
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        return False

def test_generate():
    payload = {
        "caption": "A catchy synthwave track with heavy bass",
        "duration": 5.0,  # Short duration for test
        "inference_steps": 10
    }
    
    print(f"\nTriggering generation: {payload}...")
    start_t = time.time()
    resp = requests.post(f"{SERVER_URL}/generate", json=payload)
    
    if resp.status_code == 200:
        elapsed = time.time() - start_t
        result = resp.json()
        print(f"✅ Generation SUCCESS in {elapsed:.2f}s!")
        print(f"   Audio paths: {[a['path'] for a in result.get('audios', [])]}")
    else:
        print(f"❌ Generation FAIL ({resp.status_code}): {resp.text}")

def test_lora():
    # Example: Load a LoRA adapter (update path to a real one)
    # create a dummy lora path for testing logic (won't work but tests endpoint)
    lora_path = "/tmp/dummy_lora" 
    print(f"\nTriggering LoRA load from: {lora_path}...")
    resp = requests.post(f"{SERVER_URL}/v1/models/lora/load", json={"lora_path": lora_path})
    if resp.status_code == 200:
        print(f"✅ LoRA Loaded: {resp.json()}")
    else:
        print(f"⚠️ LoRA Load Failed (expected if path invalid): {resp.text}")

def test_complex_generate():
    payload = {
        "caption": "A complex test request",
        "instrumental": True,
        "bpm": 128,
        "key_scale": "Am",
        "inference_steps": 10,
        "use_adg": True,  # Advanced DiT
        "thinking": True, # LLM
        "lm_temperature": 0.9,
        "cot_caption": "Override caption",
        "task_type": "text2music"
    }
    
    print(f"\nTriggering COMPLEX generation: {payload}...")
    try:
        resp = requests.post(f"{SERVER_URL}/generate", json=payload)
        if resp.status_code == 200:
             print("✅ Complex Request: Validated & Accepted")
        else:
             print(f"❌ Complex Request Failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    if test_health():
        print("\n--- Basic Test ---")
        # input("Press Enter to test generation...")
        # test_generate()
        
        print("\n--- Advanced Feature Test ---")
        test_complex_generate()
        
        # Uncomment to test LoRA
        # input("\nPress Enter to test LoRA loading...")
        # test_lora()
    else:
        print("\nPlease start the server first: ./run_server.sh")
