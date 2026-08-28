"""
Télécharge le corpus de datasheets dans le dossier `data/`.

Usage : python download_data.py
"""
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# Nom local -> URL officielle du PDF
DOCUMENTS = {
    "pico.pdf": "https://datasheets.raspberrypi.com/pico/pico-datasheet.pdf",
    "esp32.pdf": "https://www.espressif.com/sites/default/files/documentation/esp32_datasheet_en.pdf",
    "arduino_uno.pdf": "https://docs.arduino.cc/resources/datasheets/A000066-datasheet.pdf",
    "bme280.pdf": "https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bme280-ds002.pdf",
    "hcsr04.pdf": "https://cdn.sparkfun.com/datasheets/Sensors/Proximity/HCSR04.pdf",
    "lm35.pdf": "https://www.ti.com/lit/ds/symlink/lm35.pdf",
}

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for name, url in DOCUMENTS.items():
        target = DATA_DIR / name
        if target.exists():
            print(f"déjà présent : {target}")
            continue
        print(f"téléchargement : {name} ...")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
        print(f"  -> {target} ({target.stat().st_size // 1024} ko)")
    print("terminé.")
