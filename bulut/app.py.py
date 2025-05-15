import gradio as gr # type: ignore
import torch # type: ignore
from torchvision import models, transforms # type: ignore
from PIL import Image # type: ignore
import requests # type: ignore
import io
import numpy as np # type: ignore

# Alet ve eşya sınıfları (ImageNet sınıflarından alınanlar)
tool_labels = [
    "screwdriver", "power drill", "hand blower", "vacuum", "microwave", "refrigerator", "laptop", "notebook",
    "desktop computer", "monitor", "keyboard", "mouse", "cellular telephone", "iPod", "remote control", 
    "camera", "digital clock", "printer", "projector", "torch", "electric fan", "table lamp"
]

def get_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    labels = response.text.splitlines()
    return labels

def load_resnet_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

def preprocess_image(image):
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def classify_tools(image):
    if image is None:
        return None, None, []

    model = load_resnet_model()
    input_batch = preprocess_image(image)

    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    labels = get_imagenet_labels()
    top100_prob, top100_catid = torch.topk(probabilities, 100)

    filtered_results = []
    chart_data = []

    for i in range(100):
        label = labels[top100_catid[i]]
        if label in tool_labels:
            prob = top100_prob[i].item() * 100
            filtered_results.append(f"{label}: {prob:.2f}%")
            chart_data.append({"sınıf": label, "olasılık": prob})
            if len(filtered_results) >= 5:
                break

    if filtered_results:
        top_class = filtered_results[0].split(":")[0]
        return top_class, "\n".join(filtered_results), chart_data
    else:
        return "Alet tespit edilemedi", "Yüklenen görüntüde hedef alet/eşya bulunamadı.", []

def generate_result_image(image, label):
    return image

def classify_image(image):
    if image is None:
        return None, "Lütfen bir görsel yükleyin.", None, []

    top_class, details, chart_data = classify_tools(image)
    result_image = generate_result_image(image, top_class)
    return top_class, details, result_image, chart_data

# Gradio arayüzü
with gr.Blocks() as demo:
    gr.Markdown("# 🛠 Alet ve Eşya Tanıma Sistemi (ResNet50 + ImageNet)")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Bir görsel yükle")
    
    classify_btn = gr.Button("🔍 Sınıflandır")
    result_label = gr.Textbox(label="Tahmin")
    result_details = gr.Textbox(label="Detaylı Sonuçlar", lines=5)
    result_image = gr.Image(label="Sonuç Görseli")

    classify_btn.click(
        classify_image,
        inputs=[image_input],
        outputs=[result_label, result_details, result_image, gr.State()]
    )

demo.launch()