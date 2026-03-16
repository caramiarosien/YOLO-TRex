"""
Analyse einer .npz-Datei aus T-Rex (Fledermaus-Tracking).
Aufgabe 1: Datei-Inspektion
Aufgabe 2: DataFrame erstellen (nur Zeitreihen-Daten, missing-Frames gefiltert)
Aufgabe 3: Trajectory Plot ("The Stupid Plot")
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend
import matplotlib.pyplot as plt

# ─── 1. Datei-Inspektion ─────────────────────────────────────────────────────
npz_path = "/Users/cara/Desktop/BA/Trex/022026_BA/data/GH060042_flipped_id0.npz"
data = np.load(npz_path, allow_pickle=True)

print("=" * 60)
print("AUFGABE 1: Datei-Inspektion")
print("=" * 60)
print(f"\nAnzahl Keys: {len(data.keys())}")
print(f"Alle Keys: {list(data.keys())}\n")

for key in data.keys():
    arr = data[key]
    print(f"  {key:30s}  shape={str(arr.shape):15s}  dtype={arr.dtype}")

# Print first 5 values for the most relevant keys
relevant_keys = ["frame", "X#wcentroid", "Y#wcentroid", "SPEED#wcentroid",
                 "missing", "time", "ANGLE", "num_pixels"]
print("\n--- Erste 5 Werte der relevanten Keys ---")
for key in relevant_keys:
    if key in data:
        print(f"  {key}: {data[key][:5]}")

# Metadata
print("\n--- Metadaten ---")
for key in ["cm_per_pixel", "id", "frame_rate", "video_size"]:
    if key in data:
        print(f"  {key}: {data[key]}")

# ─── 2. DataFrame ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("AUFGABE 2: Pandas DataFrame")
print("=" * 60)

# Only include time-series arrays with the same length (6842)
n = len(data["frame"])
ts_keys = [k for k in data.keys() if data[k].ndim == 1 and len(data[k]) == n]
print(f"\nZeitreihen-Keys (Länge={n}): {ts_keys}")

df = pd.DataFrame({k: data[k] for k in ts_keys})

# Replace inf with NaN for cleaner handling
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Mark which frames are present (missing == 0)
df["is_present"] = df["missing"] == 0

print(f"\nDataFrame Shape: {df.shape}")
print(f"Erkannte Frames (missing==0): {df['is_present'].sum()} von {len(df)}")
print(f"\nErste 10 Zeilen (Auswahl):\n")
cols_display = ["frame", "X#wcentroid", "Y#wcentroid", "SPEED#wcentroid",
                "missing", "time", "is_present"]
cols_display = [c for c in cols_display if c in df.columns]
print(df[cols_display].head(10).to_string())

# Filter to only present frames for plotting
df_valid = df[df["is_present"]].copy()
print(f"\nGültige Datenpunkte für Plot: {len(df_valid)}")
print(f"\nStatistik (nur gültige Frames):")
stats_cols = ["X#wcentroid", "Y#wcentroid", "SPEED#wcentroid"]
stats_cols = [c for c in stats_cols if c in df_valid.columns]
print(df_valid[stats_cols].describe().to_string())

# ─── 3. Trajectory Plot ("The Stupid Plot") ──────────────────────────────────
print("\n" + "=" * 60)
print("AUFGABE 3: Trajectory Plot wird erstellt...")
print("=" * 60)

x = df_valid["X#wcentroid"].values
y = df_valid["Y#wcentroid"].values

fig, ax = plt.subplots(figsize=(12, 8))

# Linie + kleine Marker
ax.plot(x, y, linewidth=0.4, color="steelblue", alpha=0.5, zorder=1)
ax.scatter(x, y, s=1.5, color="darkorange", alpha=0.7, zorder=2)

# Y-Achse invertieren (Video-Koordinaten: Nullpunkt oben links)
ax.invert_yaxis()

ax.set_xlabel("X-Position (px)", fontsize=12)
ax.set_ylabel("Y-Position (px)", fontsize=12)
ax.set_title("Fledermaus-Flugbahn – GH060042 (ID 0)\n\"The Stupid Plot\"",
             fontsize=14, fontweight="bold")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

plt.tight_layout()

out_path = "/Users/cara/Desktop/BA/YOLO/trajectory_plot_GH060042_id0.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n✅ Plot gespeichert: {out_path}")

# ─── 4. Tracklet-Extraktion (Teleportationen entfernen) ──────────────────────
print("\n" + "=" * 60)
print("AUFGABE 4: Tracklet-Extraktion")
print("=" * 60)

gap_threshold = 10  # Frames

# Berechne Frame-Differenz zwischen aufeinanderfolgenden gültigen Zeilen
df_valid["frame_diff"] = df_valid["frame"].diff()

# Tracklet_ID: startet bei 0, erhöht sich bei jedem Gap > Schwellenwert
df_valid["Tracklet_ID"] = (df_valid["frame_diff"] > gap_threshold).cumsum()

n_tracklets = df_valid["Tracklet_ID"].nunique()
print(f"\nSchwellenwert: {gap_threshold} Frames")
print(f"Erkannte Tracklets: {n_tracklets}\n")

# Zusammenfassung pro Tracklet
tracklet_summary = df_valid.groupby("Tracklet_ID").agg(
    Frames=("frame", "count"),
    Frame_Start=("frame", "first"),
    Frame_End=("frame", "last"),
    X_mean=("X#wcentroid", "mean"),
    Y_mean=("Y#wcentroid", "mean"),
    Speed_mean=("SPEED#wcentroid", "mean"),
).reset_index()
print(tracklet_summary.to_string())

# ─── 4b. Tracklet-Plot (jedes Tracklet in eigener Farbe) ─────────────────────
print("\nTracklet-Plot wird erstellt...")

fig2, ax2 = plt.subplots(figsize=(14, 9))
cmap = matplotlib.colormaps.get_cmap("tab20").resampled(n_tracklets)

for tid in range(n_tracklets):
    t = df_valid[df_valid["Tracklet_ID"] == tid]
    color = cmap(tid)
    ax2.plot(t["X#wcentroid"], t["Y#wcentroid"],
             linewidth=0.6, color=color, alpha=0.7, zorder=1)
    ax2.scatter(t["X#wcentroid"], t["Y#wcentroid"],
                s=2, color=color, alpha=0.8, zorder=2,
                label=f"T{tid} ({len(t)} F)")

ax2.invert_yaxis()
ax2.set_xlabel("X-Position (px)", fontsize=12)
ax2.set_ylabel("Y-Position (px)", fontsize=12)
ax2.set_title(f"Fledermaus-Flugbahnen – GH060042 (ID 0)\n"
              f"{n_tracklets} Tracklets (gap_threshold={gap_threshold})",
              fontsize=14, fontweight="bold")
ax2.set_aspect("equal")
ax2.grid(True, alpha=0.3)

# Legende nur wenn nicht zu viele Tracklets
if n_tracklets <= 30:
    ax2.legend(fontsize=6, ncol=3, loc="lower right",
               framealpha=0.8, markerscale=3)

plt.tight_layout()

out_path2 = "/Users/cara/Desktop/BA/Trex/022026_BA/tracklet_plot_GH060042_id0.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"\n✅ Tracklet-Plot gespeichert: {out_path2}")
