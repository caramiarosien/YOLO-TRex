import argparse
import numpy as np
from pathlib import Path

def process_file(file_path, size_threshold):
    large_objects = []
    max_w, max_h, max_area = 0, 0, 0
    obj_sizes = []
    class_indices = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 3: continue
            
            try:
                class_id = int(parts[0])
                coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            except (ValueError, IndexError):
                print(f"⚠️  [WARN] Skipping malformed line {i+1} in {file_path.name}")
                continue
            
            min_x, min_y = coords.min(axis=0)
            max_x, max_y = coords.max(axis=0)
            w = max_x - min_x
            h = max_y - min_y
            area = w * h
            
            max_w = max(max_w, w)
            max_h = max(max_h, h)
            max_area = max(max_area, area)
            obj_sizes.append((w, h))
            class_indices.append(class_id)
            
            if w > size_threshold or h > size_threshold:
                large_objects.append({
                    "index": i, "class_id": class_id, "w": w, "h": h, "area": area, "points": len(coords)
                })
                
    except Exception as e:
        print(f"⚠️  [ERR] Error reading {file_path.name}: {e}")
        return None
        
    return {
        "max_w": max_w, "max_h": max_h, "max_area": max_area,
        "large_objects": large_objects, "total_objects": len(obj_sizes),
        "obj_sizes": obj_sizes, "classes": class_indices
    }

def main():
    parser = argparse.ArgumentParser(description="Validate YOLO labels to find large anomalies or statistics.")
    parser.add_argument("--path", required=True, help="Path to a single .txt file or a directory of .txt files")
    parser.add_argument("--threshold", type=float, default=0.5, help="Size threshold (normalized 0-1) for warning (default 0.5)")
    parser.add_argument("--quiet", action="store_true", help="Only show results for files with anomalies")
    args = parser.parse_args()
    
    target_path = Path(args.path)
    if not target_path.exists():
        print(f"❌ Error: Path not found: {target_path}")
        return
        
    if target_path.is_file():
        files = [target_path]
    else:
        files = sorted(list(target_path.glob("*.txt")))
        
    if not files:
        print(f"❓ No .txt files found in {target_path}")
        return

    if not args.quiet:
        print(f"🔍 Analyzing {len(files)} file(s) with threshold {args.threshold}...")
        print("-" * 60)
    
    bad_files = []
    total_objects_global = 0
    class_counts = {}
    all_max_ws = []
    all_max_hs = []

    for file_path in files:
        res = process_file(file_path, args.threshold)
        if res is None: continue
        
        total_objects_global += res["total_objects"]
        all_max_ws.append(res["max_w"])
        all_max_hs.append(res["max_h"])
        
        for cid in res["classes"]:
            class_counts[cid] = class_counts.get(cid, 0) + 1

        if res["large_objects"]:
            bad_files.append((file_path.name, res))
            
            print(f"📄 {'ANOMALY in ' if args.quiet else 'File: '}{file_path.name}")
            print(f"   Objects found: {res['total_objects']}")
            print(f"   Max Dim:       W={res['max_w']:.3f}, H={res['max_h']:.3f}")
            print(f"   Large Objects (>{args.threshold}):")
            for obj in res["large_objects"]:
                print(f"     - Obj {obj['index']} (Class {obj['class_id']}): W={obj['w']:.3f}, H={obj['h']:.3f}, Pts={obj['points']}")
        elif not args.quiet and target_path.is_file():
             print(f"📄 File: {file_path.name}")
             print(f"   Objects found: {res['total_objects']}")
             print(f"   Max Dim:       W={res['max_w']:.3f}, H={res['max_h']:.3f}")
             print(f"   ✅ No large objects found.")

    if target_path.is_dir():
        print("-" * 60)
        print(f"✅ Scan Complete.")
        print(f"📊 Global Stats:")
        print(f"   Total Files:      {len(files)}")
        print(f"   Total Objects:    {total_objects_global}")
        if files:
            print(f"   Avg Max Width:    {np.mean(all_max_ws):.4f}")
            print(f"   Avg Max Height:   {np.mean(all_max_hs):.4f}")
            print(f"   Max Width Seen:   {np.max(all_max_ws):.4f}")
            print(f"   Max Height Seen:  {np.max(all_max_hs):.4f}")
            print(f"   Classes Found:    {', '.join(str(k) for k in sorted(class_counts.keys()))}")
            for cid in sorted(class_counts.keys()):
                print(f"     - Class {cid}: {class_counts[cid]} objects")
        
        print(f"⚠️  Anomalies: {len(bad_files)} / {len(files)} files contain large objects.")
        if bad_files and not args.quiet:
            print("📋 Top anomalies:")
            for name, res in bad_files[:10]:
                print(f"   - {name}: MaxW={res['max_w']:.3f}, MaxH={res['max_h']:.3f}")
            if len(bad_files) > 10:
                print(f"     ... and {len(bad_files) - 10} more.")

if __name__ == "__main__":
    main()

