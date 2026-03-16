import os
import sys

# --- KONFIGURATION ---
# Pfad zum Hauptordner
ROOT_PATH = "/Volumes/Kasanka21/KasankaCameras"

# Sicherheits-Schalter:
# True  = Nur simulieren (zeigt in der Konsole, was passieren würde)
# False = Ernst machen (benennt Dateien wirklich um)
DRY_RUN = False 
# ---------------------

def rename_process():
    if not os.path.exists(ROOT_PATH):
        print(f"Fehler: Der Pfad {ROOT_PATH} wurde nicht gefunden.")
        return

    print(f"Starte Prozess in: {ROOT_PATH}")
    print(f"Modus: {'SIMULATION (Keine Änderungen)' if DRY_RUN else 'SCHARF (Änderungen werden angewendet)'}")
    print("-" * 60)

    # 1. Ebene: Datums-Ordner (z.B. 20211026)
    # Wir sortieren, damit die Reihenfolge logisch bleibt
    for date_folder in sorted([d for d in os.listdir(ROOT_PATH) if not d.startswith('.')]):
        date_path = os.path.join(ROOT_PATH, date_folder)
        
        if not os.path.isdir(date_path):
            continue

        # 2. Ebene: Orts-Ordner (z.B. BBC)
        for location_folder in sorted([l for l in os.listdir(date_path) if not l.startswith('.')]):
            location_path = os.path.join(date_path, location_folder)

            if not os.path.isdir(location_path):
                continue

            # 3. Ebene: Frame-Ordner suchen (z.B. GH059870_frames)
            # Wir suchen nur Ordner, die "_frames" im Namen haben
            for item in sorted(os.listdir(location_path)):
                frame_folder_path = os.path.join(location_path, item)

                # Prüfen, ob es ein Ordner ist und "_frames" enthält
                if os.path.isdir(frame_folder_path) and "_frames" in item:
                    
                    # Sicherheitscheck: Wurde der Ordner vielleicht schon umbenannt?
                    # Wenn das Datum schon im Namen steckt, überspringen wir ihn, um doppeltes Benennen zu verhindern.
                    if date_folder in item and location_folder in item:
                        print(f"[SKIP] Ordner scheint schon bearbeitet: {item}")
                        continue

                    # --- SCHRITT A: Ordner umbenennen ---
                    # Alter Name: GH059870_frames
                    # Neuer Name: GH059870_frames_20211026_BBC
                    new_folder_name = f"{item}_{date_folder}_{location_folder}"
                    new_folder_path = os.path.join(location_path, new_folder_name)

                    if DRY_RUN:
                        print(f"ORDNER RENAME: '{item}' -> '{new_folder_name}'")
                    else:
                        try:
                            os.rename(frame_folder_path, new_folder_path)
                        except OSError as e:
                            print(f"Fehler beim Umbenennen des Ordners {item}: {e}")
                            continue

                    # WICHTIG: Wenn wir nicht im Dry Run sind, müssen wir für den nächsten Schritt
                    # (Dateien umbenennen) den neuen Pfad nutzen.
                    current_working_path = frame_folder_path if DRY_RUN else new_folder_path
                    
                    # --- SCHRITT B: Dateien im Ordner umbenennen ---
                    # Wir holen alle Dateien, ignorieren versteckte (.DS_Store etc) und sortieren sie
                    try:
                        files = sorted([f for f in os.listdir(current_working_path) if not f.startswith('.') and os.path.isfile(os.path.join(current_working_path, f))])
                    except FileNotFoundError:
                        print(f"Konnte Pfad nicht lesen: {current_working_path}")
                        continue

                    for index, filename in enumerate(files, start=1):
                        # Dateiendung extrahieren (z.B. .jpg oder .png)
                        file_name_part, file_ext = os.path.splitext(filename)
                        
                        # Neuer Dateiname: frame_0001_GH059870_frames_20211026_BBC.jpg
                        # {index:04d} sorgt für die Nullen (0001, 0002...)
                        new_filename = f"frame_{index:04d}_{new_folder_name}{file_ext}"
                        
                        old_file_path = os.path.join(current_working_path, filename)
                        new_file_path = os.path.join(current_working_path, new_filename)

                        if DRY_RUN:
                            # Wir geben nur die ersten 3 und letzten Dateien aus, um die Konsole nicht zu fluten
                            if index <= 3 or index >= len(files) - 2:
                                print(f"   DATEI: '{filename}' -> '{new_filename}'")
                            elif index == 4:
                                print("   ... (weitere Dateien) ...")
                        else:
                            os.rename(old_file_path, new_file_path)

    print("-" * 60)
    print("Fertig.")

if __name__ == "__main__":
    rename_process()