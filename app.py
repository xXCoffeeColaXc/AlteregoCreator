import gradio as gr
from pathlib import Path
from PIL import Image
import torch
from config.models import Config
from main import load_checkpoint, load_config
from utils import denorm, generate_valid_permutations
from network import Generator
from dataloader import get_transform

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load config and model
config: Config = load_config('config/config.yaml')
model_path = Path(config.folders.checkpoints) / 'new_run/30-G.ckpt'
G = load_checkpoint(model_path, config.model)

gender_options = ["Female", "Male"]
age_options = ["Old", "Young"]
hair_color_options = ["Black Hair", "Blond Hair", "Brown Hair"]


# Functions adapted for Gradio
def preprocess_input(image, user_attributes):
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    transform = get_transform(config.data, 'val')
    image_tensor = transform(image)
    label_org = torch.tensor(
        [1 if attr in user_attributes else 0 for attr in config.data.selected_attrs], dtype=torch.float32
    )
    valid_permutations = generate_valid_permutations(label_org, config.data.selected_attrs)
    return image_tensor, valid_permutations


def infer_and_postprocess(image_dict, gender, age, hair_color):

    user_attributes = [attr for attr in [gender, age, hair_color] if attr]

    image = image_dict['composite']
    extrema = image.convert("L").getextrema()
    if image and not user_attributes:
        return None, "Please select attributes."
    if extrema[0] == extrema[1]:
        return None, "Please upload an image."
    if not image:
        return None, "Please upload an image."

    # Preprocess input
    input_tensor, target_labels = preprocess_input(image, user_attributes)

    # Inference
    input_tensor = input_tensor.unsqueeze(0).to(device)
    generated_images = []
    for label in target_labels:
        with torch.no_grad():
            target_tensor = label.unsqueeze(0).to(device)
            generated_image = G(input_tensor, target_tensor)
        generated_images.append(generated_image.squeeze(0))

    # Postprocess
    postprocessed_images = []
    for gen_image, label in zip(generated_images, target_labels):
        gen_image = denorm(gen_image.cpu())
        gen_image = (gen_image.mul(255).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy())
        gen_image = Image.fromarray(gen_image)
        label_names = [config.data.selected_attrs[i] for i, l in enumerate(label.cpu().numpy()) if l == 1]
        postprocessed_images.append((gen_image, ", ".join(label_names)))

    return postprocessed_images, None


with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="purple")) as alterego_app:
    # Header
    gr.Markdown(
        """
        <div style="text-align: center; padding: 20px; background-color: #f0f8ff; border-radius: 10px;">
            <h1 style="color: #4a90e2;">Alterego Creator</h1>
            <p style="font-size: 18px; color: #6a6a6a;">Upload your image and select attributes to create your alterego!</p>
        </div>
        """
    )

    # Main Interface
    with gr.Row():
        # Upload Panel
        with gr.Column():
            gr.Markdown(
                """
                <div style="background-color: #e8f4fc; padding: 15px; border-radius: 10px; border: 1px solid #d4e8f8;">
                    <h2 style="color: #4a90e2;">Upload Panel</h2>
                    <p style="color: #6a6a6a;">Start by uploading an image and selecting the attributes you want.</p>
                </div>
                """
            )
            image_input = gr.ImageEditor(label=None, type="pil", elem_id="image-input", show_label=False)

            # Attribute Selection
            gr.Markdown("### Select Attributes")
            gender_input = gr.Radio(choices=gender_options, label="Gender", elem_id="gender-selection")
            age_input = gr.Radio(choices=age_options, label="Age", elem_id="age-selection")
            hair_color_input = gr.Radio(choices=hair_color_options, label="Hair Color", elem_id="hair-color-selection")

            generate_button = gr.Button("Generate Alteregos", elem_id="generate-button")

        # Output Panel
        with gr.Column():
            gr.Markdown(
                """
                <div style="background-color: #f9f0ff; padding: 15px; border-radius: 10px; border: 1px solid #e4d8f8;">
                    <h2 style="color: #8a6fc8;">Alterego View Panel</h2>
                    <p style="color: #6a6a6a;">Generated alteregos will appear here. Click to enlarge.</p>
                </div>
                """
            )
            output_gallery = gr.Gallery(
                label=None, elem_id="output-gallery", show_label=False, columns=3, height="500px"
            )
            error_message = gr.Textbox(
                label=None,
                placeholder="Errors or status messages will appear here.",
                interactive=False,
                elem_id="error-box"
            )

    # CSS Styling
    alterego_app.css = """
        #image-input { margin-top: 10px; }
        #gender-selection, #age-selection, #hair-color-selection { margin-top: 20px; }
        #generate-button { margin-top: 10px; background-color: #4a90e2; color: white; border-radius: 5px; }
        #output-gallery { margin-top: 20px; }
        #error-box { margin-top: 20px; border: 1px solid #d4e8f8; }
    """

    # Button Click Action
    def collect_attributes(gender, age, hair_color):
        return [attr for attr in [gender, age, hair_color] if attr]

    generate_button.click(
        fn=infer_and_postprocess,
        inputs=[image_input, gender_input, age_input, hair_color_input],
        outputs=[output_gallery, error_message],
        preprocess=collect_attributes
    )

alterego_app.launch()
