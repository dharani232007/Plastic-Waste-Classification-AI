from ultralytics import YOLO

# 1. Load your custom trained model
model = YOLO("best.pt")

# 2. Run the detection on your test image
print("Detecting plastic waste...")
results = model.predict("test_image.jpg", show=True, save=True)