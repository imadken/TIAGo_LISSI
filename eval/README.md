# Evaluation Suite — TIAGo VLM Robot

Run each script independently. All results saved to `data/` as JSON/CSV.
Run `report.py` at the end to compile everything into figures and a summary table.

## Scripts

| Script | Needs ROS | Needs GPU | Purpose |
|--------|-----------|-----------|---------|
| eval_detection.py | No | No | YOLO vs VLM bbox IoU on saved images |
| eval_depth_refine.py | No | No | Depth refinement quality on saved RGBD |
| eval_grasp_recorder.py | Yes | No | Record live grasp trial outcomes |
| eval_search_recorder.py | Yes | No | Record head search efficiency |
| eval_face.py | No | No | Face recognition accuracy vs distance |
| eval_latency.py | No | No | VLM API + component latency |
| eval_statemanager.py | No | No | Spatial relation accuracy vs ground truth |
| report.py | No | No | Compile all results → figures + table |

## Workflow

1. **Collect images** (offline):
   - Save RGB frames as JPEGs in `data/images/`
   - Save paired depth as numpy `.npy` files in `data/depth/`
   - Annotate ground-truth bboxes in `data/annotations.json`

2. **Run offline evaluations**:
   ```bash
   cd eval/
   python3 eval_detection.py
   python3 eval_depth_refine.py
   python3 eval_face.py
   python3 eval_latency.py
   python3 eval_statemanager.py
   ```

3. **Run ROS evaluations** (robot running):
   ```bash
   python3 eval_grasp_recorder.py --object bottle --trials 15
   python3 eval_search_recorder.py --target bottle --trials 10
   ```

4. **Generate report**:
   ```bash
   python3 report.py
   ```
   Outputs: `results/summary_table.csv`, `results/fig_*.pdf`
