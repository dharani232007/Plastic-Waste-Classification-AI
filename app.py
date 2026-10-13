import gradio as gr
from ultralytics import YOLO
import cv2

# 1. Load your trained AI model
model = YOLO("best.pt")

# 2. Plastic Resin Mapping Dictionary
plastic_type_map = {
    "Bottle": "PET (Type 1) - Highly Recyclable",
    "Bottle cap": "HDPE (Type 2) - Recyclable",
    "Plastic container": "PP (Type 5) - Recyclable",
    "Plastic bag - wrapper": "LDPE (Type 4) - Requires special drop-off",
    "Cup": "PS (Type 6) or PP - Low Recyclability",
    "Lid": "HDPE or PP (Type 2/5)",
    "Straw": "PP (Type 5)",
    "Styrofoam piece": "PS (Type 6) - Non-Recyclable"
}

def detect_and_classify(image):
    # EDGE CASE: Handle missing image upload
    if image is None:
        return None, "⚠️ ERROR: Please upload an image first to start the analysis!"

    # Run AI prediction
    results = model.predict(image, conf=0.2)
    
    # Draw bounding boxes
    annotated_img = results[0].plot()
    rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Generate Analysis Report
    detected_items = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        
        if class_name in plastic_type_map:
            resin = plastic_type_map[class_name]
            detected_items.append(f"✅ {class_name}: {resin}")
        else:
            detected_items.append(f"♻️ {class_name}: Unknown/Other Material")
            
    # Format Text Output
    if not detected_items:
        report = "🔍 No recognized plastic waste detected. Please try another image."
    else:
        report = "\n".join(list(set(detected_items)))
        
    return rgb_img, report

# 3. Custom Dashboard UI Configuration
custom_css = """
h1 {text-align: center; color: #10b981; font-weight: bold;}
h3 {text-align: center; color: #6b7280;}
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="emerald", secondary_hue="green"), css=custom_css) as demo:
    
    # Main Headers
    gr.Markdown("# ♻️ Smart Plastic Waste Classifier")
    gr.Markdown("### Deep Neural Network Project for Automated Recycling")
    gr.Markdown("---")
    
    # Side-by-Side Application Layout
    with gr.Row():
        # Left Panel
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input Panel")
            # Fixed height to match output panel
            input_image = gr.Image(type="numpy", label="Upload Waste Image", height=400)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                submit_btn = gr.Button("🚀 Analyze Image", variant="primary")
                
        # Right Panel
        with gr.Column(scale=1):
            gr.Markdown("### 📤 AI Analysis")
            output_image = gr.Image(type="numpy", label="Detection Map", height=400)
            output_report = gr.Textbox(label="📊 Polymer Type & Recycling Report", lines=4)

    # Technical Details Accordion (Collapsed by default)
    with gr.Accordion("ℹ️ How it Works & Model Info", open=False):
        gr.Markdown(
            """
            - **Model Architecture**: YOLOv8 (Deep Neural Network)
            - **Training Dataset**: TACO (Trash Annotations in Context)
            - **Detection Pipeline**: The system first localizes waste objects using bounding boxes. It then maps the identified object class (e.g., 'Bottle') to its most likely polymer resin (e.g., 'PET') to assist automated sorting facilities.
            """
        )

    # Application Logic Routing
    submit_btn.click(
        fn=detect_and_classify,
        inputs=input_image,
        outputs=[output_image, output_report]
    )
    
    clear_btn.click(
        fn=lambda: (None, None, None),
        inputs=None,
        outputs=[input_image, output_image, output_report]
    )

if __name__ == "__main__":
    demo.launch()