import os
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor

# --- KONFIGURATION FÜR WINDOWS ---
# Bitte passe hier deinen Laufwerksbuchstaben an (z.B. E:, D:, F:)
SEARCH_DIR = r"E:\KasankaCameras"

# Anzahl der gleichzeitigen Videos. 
# Bei einer externen Festplatte (HDD) ist 4 meistens das Limit.
MAX_WORKERS = 4

def process_single_video(file_info):
    """
    Diese Funktion wird parallel auf mehreren CPU-Kernen ausgeführt.
    """
    video_path, output_dir, output_pattern, filename = file_info

    # Zielordner erstellen
    os.makedirs(output_dir, exist_ok=True)

    # --- FFmpeg Befehl für Windows (CPU Modus) ---
    # Wir nutzen hier keine explizite Hardware-Beschleunigung, da diese unter Windows
    # stark von der Grafikkarte (NVIDIA vs AMD vs Intel) abhängt.
    # Stattdessen verlassen wir uns auf Multiprocessing (4 Videos gleichzeitig).
    
    command = [
        "ffmpeg",
        "-nostdin",      # Wichtig für Background-Prozesse
        "-y",            # Überschreiben erzwingen
        "-i", video_path,
        "-vf", "fps=1",
        "-q:v", "1",     # Beste JPEG Qualität
        "-qmin", "1",    # Verhindert Qualitätsschwankungen
        "-hide_banner",
        "-loglevel", "error",
        output_pattern
    ]

    try:
        # Starten des Prozesses
        subprocess.run(command, check=True)
        return f"Fertig: {filename}"
    except subprocess.CalledProcessError as e:
        return f"!!! FEHLER bei {filename}: {e}"
    except FileNotFoundError:
        return "!!! FEHLER: ffmpeg.exe nicht gefunden."

def main():
    print(f"High-Performance Modus für Windows gestartet.")
    print(f"Suche in: {SEARCH_DIR}")
    print(f"Gleichzeitige Prozesse: {MAX_WORKERS}")
    print("-" * 60)

    # Prüfen ob der Pfad existiert
    if not os.path.exists(SEARCH_DIR):
        print(f"CRITICAL ERROR: Der Pfad '{SEARCH_DIR}' existiert nicht.")
        print("Hast du den richtigen Laufwerksbuchstaben eingetragen?")
        return

    # 1. Alle Videos sammeln
    tasks = []
    print("Sammle Videodateien (das kann kurz dauern)...")
    
    for root, dirs, files in os.walk(SEARCH_DIR):
        for filename in files:
            if filename.lower().endswith(".mp4") and not filename.startswith("._"):
                
                video_path = os.path.join(root, filename)
                video_name_no_ext = os.path.splitext(filename)[0]
                
                # Pfade für Output
                output_dir = os.path.join(root, f"{video_name_no_ext}_frames")
                output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
                
                # Aufgabe zur Liste hinzufügen
                tasks.append((video_path, output_dir, output_pattern, filename))

    print(f"{len(tasks)} Videos gefunden. Starte parallele Verarbeitung...")
    print("-" * 60)

    # 2. Parallel abarbeiten
    # Windows benötigt diesen 'if __name__' Schutz unbedingt für Multiprocessing
    if tasks:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = executor.map(process_single_video, tasks)
            
            # Ausgabe der Ergebnisse
            for result in results:
                print(result)
    else:
        print("Keine MP4-Dateien gefunden.")

    print("-" * 60)
    print("Alle Videos verarbeitet.")

if __name__ == "__main__":
    # Check ob FFmpeg verfügbar ist
    if shutil.which("ffmpeg") is None:
        print("ACHTUNG: 'ffmpeg' wurde nicht gefunden!")
        print("Bitte stelle sicher, dass FFmpeg installiert ist und im Windows PATH liegt.")
    else:
        main()