import os
import subprocess
import shutil

# --- KONFIGURATION FÜR WINDOWS ---
# WICHTIG: Das 'r' vor den Anführungszeichen stehen lassen! 
# Das sorgt dafür, dass die Backslashes (\) von Windows richtig erkannt werden.
# Ändere "E:\" zu dem Buchstaben deiner Festplatte.
SEARCH_DIR = r"/Volumes/Kasanka21/KasankaCameras" 

def process_videos():
    print(f"Starte Verarbeitung in: {SEARCH_DIR}")
    print("Qualität: Höchste JPEG Stufe (q:v 1)")
    print("-" * 60)

    # Prüfen, ob der Pfad überhaupt existiert
    if not os.path.exists(SEARCH_DIR):
        print(f"FEHLER: Der Ordner '{SEARCH_DIR}' wurde nicht gefunden.")
        print("Bitte überprüfe den Laufwerksbuchstaben (D:, E:, F: etc.)")
        return

    # os.walk funktioniert auch auf Windows perfekt rekursiv
    for root, dirs, files in os.walk(SEARCH_DIR):
        for filename in files:
            
            # Nur MP4-Dateien. 
            # (Auf Windows gibt es meist keine ._ Dateien, aber der Check schadet nicht)
            if filename.lower().endswith(".mp4") and not filename.startswith("._"):
                
                # Pfade zusammenbauen
                video_path = os.path.join(root, filename)
                video_name_no_ext = os.path.splitext(filename)[0]
                
                # Zielordner erstellen
                output_dir = os.path.join(root, f"{video_name_no_ext}_frames")
                os.makedirs(output_dir, exist_ok=True)

                print(f"Bearbeite: {filename}")
                
                # Output Pattern
                output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

                # --- FFmpeg Befehl ---
                # Auf Windows ist 'ffmpeg' oft genug, wenn es im PATH ist.
                # Ansonsten muss hier der volle Pfad zur ffmpeg.exe stehen (z.B. r"C:\ffmpeg\bin\ffmpeg.exe")
                command = [
                    "ffmpeg",
                    "-nostdin",
                    "-y", 
                    "-i", video_path,
                    "-vf", "fps=1",
                    "-q:v", "1",
                    "-qmin", "1",
                    "-hide_banner",
                    "-loglevel", "error", 
                    output_pattern
                ]

                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"!!! Fehler bei Datei {filename}: {e}")
                except FileNotFoundError:
                    print("!!! CRITICAL ERROR: FFmpeg wurde nicht gefunden.")
                    print("Bitte stelle sicher, dass FFmpeg installiert ist und im Windows PATH liegt.")
                    return

    print("-" * 60)
    print("Alle Videos verarbeitet.")

if __name__ == "__main__":
    process_videos()