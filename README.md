# PlateScanner

To build project use:
```bash
docker build -t platescanner .
```

To train yolo in bash
```bash
yolo detect train datasets=dataset\data.yaml model="yolov8n.yaml" epochs=1
```