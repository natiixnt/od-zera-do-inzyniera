import argparse
import csv
from pathlib import Path
from statistics import mean

def percentile(values, p: float):
    if not values:
        return 0.0
    v = sorted(values)
    k = (len(v) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(v) - 1)
    if f == c:
        return float(v[f])
    return float(v[f] + (v[c] - v[f]) * (k - f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="sciezka do metrics.csv")
    ap.add_argument("--zero_det_limit", type=int, default=0, help="alarm gdy liczba obrazow z 0 detekcji > ten limit")
    ap.add_argument("--avg_conf_min", type=float, default=0.0, help="alarm gdy srednie avg_conf spadnie ponizej progu")
    ap.add_argument("--p95_latency_max", type=float, default=0.0, help="alarm gdy p95 latency_ms przekroczy prog")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise ValueError(f"metrics csv not found: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # konwersje na liczby
            row["detections_count"] = int(float(row["detections_count"]))
            row["avg_conf"] = float(row["avg_conf"])
            row["latency_ms"] = float(row["latency_ms"])
            rows.append(row)

    if not rows:
        print("[WARN] empty metrics.csv")
        return

    det_counts = [x["detections_count"] for x in rows]
    confs = [x["avg_conf"] for x in rows]
    lats = [x["latency_ms"] for x in rows]

    zero_det = [x for x in rows if x["detections_count"] == 0]
    zero_det_n = len(zero_det)

    avg_conf_mean = mean(confs)
    lat_mean = mean(lats)
    lat_p50 = percentile(lats, 50)
    lat_p95 = percentile(lats, 95)

    # top 5 najwolniejszych
    slowest = sorted(rows, key=lambda x: x["latency_ms"], reverse=True)[:5]

    print("=== SUMMARY (metrics.csv) ===")
    print(f"images: {len(rows)}")
    print(f"zero detections: {zero_det_n}")
    print(f"avg_conf: mean={avg_conf_mean:.4f}")
    print(f"latency_ms: mean={lat_mean:.2f} p50={lat_p50:.2f} p95={lat_p95:.2f}")
    print("")
    print("top 5 slowest:")
    for s in slowest:
        print(f"- {s['image']}  latency_ms={s['latency_ms']:.2f}  det={s['detections_count']}  avg_conf={s['avg_conf']:.3f}")

    print("")
    print("=== BASIC ALERTS ===")
    alerts = 0

    if args.zero_det_limit > 0 and zero_det_n > args.zero_det_limit:
        print(f"[ALERT] zero detections too many: {zero_det_n} > {args.zero_det_limit}")
        alerts += 1
    else:
        print("[OK] zero detections")

    if args.avg_conf_min > 0 and avg_conf_mean < args.avg_conf_min:
        print(f"[ALERT] avg_conf mean too low: {avg_conf_mean:.4f} < {args.avg_conf_min}")
        alerts += 1
    else:
        print("[OK] avg_conf mean")

    if args.p95_latency_max > 0 and lat_p95 > args.p95_latency_max:
        print(f"[ALERT] p95 latency too high: {lat_p95:.2f} > {args.p95_latency_max}")
        alerts += 1
    else:
        print("[OK] p95 latency")

    print("")
    if alerts == 0:
        print("[INFO] no alerts")
    else:
        print(f"[INFO] alerts={alerts}")

if __name__ == "__main__":
    main()
