# Robot Eval Day — Execution Plan

---

## Before You Leave Home

- [ ] `.env` has valid `EDENAI_API_KEY` and `GROQ_API_KEY`
- [ ] Charge laptop
- [ ] Bring: bottle, cup (objects for grasp trials)

---

## Phase 1 — Offline Tests (no robot, no API)

```bash
cd ~/tiago_public_ws/eval
```

```bash
# Symbolic reasoning — no API, instant
python3 eval_statemanager.py --demo

# Depth pipeline — no API, instant
python3 eval_depth_refine.py --demo

# Face recognition — no VLM API, uses Kaggle dataset
python3 face_recognition_service.py &
python3 eval_face.py --kaggle_dir data/kaggle_faces
```

---

## Phase 2 — API Latency Test (5 VLM calls only)

```bash
python3 eval_latency.py --skip_yolo --skip_face
# → 5 VLM calls, ~15s total
```

---

## Phase 3 — Live Robot Tests (inside docker)

```bash
bash ~/tiago_public_ws/eval_setup.sh --docker
```

```bash
# Inside docker:
export ROS_MASTER_URI=http://10.68.0.1:11311
export ROS_IP=10.68.0.129
cd /workspace/eval

# Grasp — 8 trials (2 per distance: 0.5m, 0.8m, 1.0m, 1.2m)
python3 eval_grasp_recorder.py --object bottle --trials 8

# Search — 5 trials (hide bottle in different spots)
python3 eval_search_recorder.py --target bottle --trials 5
```

---

## Phase 4 — Report

```bash
cd ~/tiago_public_ws/eval
python3 report.py
# → summary_table.csv + PDF figures
```

---

## Checklist

| Test | API calls | Needs robot |
| --- | --- | --- |
| eval_statemanager | 0 | No |
| eval_depth_refine | 0 | No |
| eval_face (Kaggle) | 0 | No |
| eval_latency | 5 VLM | No |
| eval_grasp_recorder | 0 | Yes |
| eval_search_recorder | 0 | Yes |
