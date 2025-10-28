import os
from pyngrok import conf, ngrok
import time


def launch_mlflow_dashboard(port: int = 5000):
    conf.get_default().ngrok_path = r"C:/Users/ASUS/OneDrive/Desktop/ngrok-v3-stable-windows-amd64/ngrok.exe"


    ngrok.set_auth_token("2yDjaRyNlzdu8YXq8g6gUlAr2Gn_i4VdzCbQLZUrTkYStmVN")

    print("🚀 Starting MLflow UI...")
    os.system(f"start cmd /k mlflow ui --port {port}")

    time.sleep(3)


    public_url = ngrok.connect(port)
    print(f"✅ MLflow UI is live at: {public_url}")

    return public_url