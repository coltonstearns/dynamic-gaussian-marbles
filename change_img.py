from PIL import Image
from src.models.ip2p import InstructPix2Pix
import torch
from torchvision import transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Load the image
img = Image.open("data/real-world/Skating/rgb/1x/0_00059.png")

img = transforms.functional.pil_to_tensor(img).unsqueeze(0)
# change to float [0, 1]
img = img.float() / 255.0
img = img.to(device)

model = InstructPix2Pix(device=device, num_train_timesteps=1000, ip2p_use_full_precision=False)
text_embedding = model.pipe._encode_prompt(
            'change the person to a clown', device=device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        )

# change img to tensor


noise_img = torch.randn_like(img).to(device)
edited_img = model.edit_image(
                        text_embedding.to(device),
                        noise_img,
                        img,
                        guidance_scale=12.5,
                        image_guidance_scale=1.5,
                        diffusion_steps=20,
                        lower_bound=0.7,
                        upper_bound=0.98,
                    )


# Save the edited image
edited_img = edited_img.squeeze(0).cpu().numpy()
edited_img = Image.fromarray((edited_img*255).astype(np.uint8).transpose(1,2,0))
edited_img.save("edited_img.png")