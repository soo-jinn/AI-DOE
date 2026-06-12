# AI DOE

AI Spec DOE is now paired with **IntelliTraffic**, a local web app for the saved
MobileNetV2 vehicle classifier in `saved_models`.

## IntelliTraffic Web App

The app lets an operator upload a JPG or PNG traffic image, runs the existing
Keras model, displays the top classification and class probability distribution,
and records each inference attempt in a local SQLite audit log.

Key files:

- `intellitraffic/app.py` - local Python web server and inference API
- `intellitraffic/static/` - professional Mapua-themed web UI
- `intellitraffic/data/inference_logs.sqlite3` - created automatically at runtime
- `saved_models/vehicle_classifier_final.keras` - trained classifier
- `saved_models/class_names.json` - class order and image size metadata

## Run Locally

From this folder:

```powershell
.\run_intellitraffic.ps1
```

Then open:

```text
http://127.0.0.1:8000
```

The web layer uses only Python's standard library. Inference still requires the
ML runtime:

```powershell
.\.venv\Scripts\pip install -r requirements.txt
```

If `.venv\Scripts\python.exe` reports that its original Python installation is
missing, recreate the virtual environment with a working Python install and then
run the install command above.

## Model Classes

The classifier outputs 11 traffic categories:

`articulated_truck`, `background`, `bicycle`, `bus`, `car`, `motorcycle`,
`non-motorized_vehicle`, `pedestrian`, `pickup_truck`, `single_unit_truck`,
`work_van`.
