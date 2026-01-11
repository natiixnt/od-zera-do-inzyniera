import argparse
import csv
from pathlib import Path
from statistics import mean


def read_metrics(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # wyciagamy podstawowe metryki (reszta kolumn to class_*)
            rows.append(
                {
                    "detections_count": int(float(row.get("detections_count", 0))),
                    "avg_conf": float(row.get("avg_conf", 0.0)),
                    "latency_ms": float(row.get("latency_ms", 0.0)),
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="baseline metrics.csv")
    ap.add_argument("--current", required=True, help="current metrics.csv")
    ap.add_argument("--conf_drop_pct", type=float, default=30.0, help="alarm: spadek avg_conf mean o X procent")
    ap.add_argument("--latency_increase_pct", type=float, default=50.0, help="alarm: wzrost latency mean o X procent")
    ap.add_argument("--det_drop_pct", type=float, default=30.0, help="alarm: spadek detections_count mean o X procent")
    args = ap.parse_args()

    b = Path(args.baseline)
    c = Path(args.current)
    if not b.exists():
        raise ValueError(f"baseline not found: {b}")
    if not c.exists():
        raise ValueError(f"current not found: {c}")

    b_rows = read_metrics(b)
    c_rows = read_metrics(c)

    if not b_rows or not c_rows:
        print("[WARN] empty baseline/current")
        return

    b_conf = mean([x["avg_conf"] for x in b_rows])
    c_conf = mean([x["avg_conf"] for x in c_rows])

    b_lat = mean([x["latency_ms"] for x in b_rows])
    c_lat = mean([x["latency_ms"] for x in c_rows])

    b_det = mean([x["detections_count"] for x in b_rows])
    c_det = mean([x["detections_count"] for x in c_rows])

    def pct_change(old, new):
        if old == 0:
            return 0.0
        return ((new - old) / old) * 100.0

    conf_change = pct_change(b_conf, c_conf)     # ujemne = spadek
    lat_change = pct_change(b_lat, c_lat)        # dodatnie = wzrost
    det_change = pct_change(b_det, c_det)        # ujemne = spadek

    print("=== DRIFT COMPARE ===")
    print(f"baseline: avg_conf_mean={b_conf:.4f} latency_mean={b_lat:.2f} det_mean={b_det:.2f}")
    print(f"current : avg_conf_mean={c_conf:.4f} latency_mean={c_lat:.2f} det_mean={c_det:.2f}")
    print("")
    print(f"delta %: avg_conf={conf_change:.2f}%  latency={lat_change:.2f}%  detections={det_change:.2f}%")
    print("")
    print("=== ALERTS ===")
    alerts = 0

    # avg_conf spada -> conf_change ujemne, wiec sprawdzamy czy spadek jest ponizej -X
    if conf_change < -abs(args.conf_drop_pct):
        print(f"[ALERT] avg_conf drop too large: {conf_change:.2f}% < -{abs(args.conf_drop_pct)}%")
        alerts += 1
    else:
        print("[OK] avg_conf")

    if lat_change > abs(args.latency_increase_pct):
        print(f"[ALERT] latency increase too large: {lat_change:.2f}% > {abs(args.latency_increase_pct)}%")
        alerts += 1
    else:
        print("[OK] latency")

    if det_change < -abs(args.det_drop_pct):
        print(f"[ALERT] detections drop too large: {det_change:.2f}% < -{abs(args.det_drop_pct)}%")
        alerts += 1
    else:
        print("[OK] detections")

    print("")
    print(f"[INFO] alerts={alerts}")


if __name__ == "__main__":
    main()
