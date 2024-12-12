import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os
import torch
import numpy as np
import threading

from criteria.embedding_loss import EmbeddingLossBuilder
from editing.interfacegan.face_editor import FaceEditor
from models.stylegan3.model import GeneratorType
from scripts.Embedding import Embedding
from scripts.Embedding import Embedding_sg3
from scripts.sketch_proxy import SketchProxy
from scripts.text_proxy import TextProxy
from scripts.ref_proxy import RefProxy, RefProxy_sg3
from scripts.bald_proxy import BaldProxy
from scripts.color_proxy import ColorProxy, ColorProxy_sg3
from scripts.feature_blending import hairstyle_feature_blending, hairstyle_feature_blending_sg3
from scripts.refine_image import RefineProxy
from torchvision import transforms

from utils.mask_ui import painting_mask
from utils.model_utils import load_sg3_models, load_base_models
from utils.options import Options
from utils.seg_utils import vis_seg

class RedirectStdout:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

class HairStyleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hair Style Editor - StyleGAN2/3")
        self.root.geometry("705x720")
        self.root.resizable(False, False)

        self.model_type = tk.StringVar()
        self.src_image_path = ""
        self.hair_target_path = ""
        self.color_target_path = ""

        self.opts = Options().parse(jupyter=True)

        self.create_widgets()

    def create_widgets(self):
        # Model selection
        model_frame = ttk.LabelFrame(self.root, text="Step 1: Select Model Type", padding=(10, 10))
        model_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)

        ttk.Radiobutton(model_frame, text="StyleGAN2", variable=self.model_type, value="SG2").grid(row=0, column=0,
                                                                                                   padx=5, pady=5)
        ttk.Radiobutton(model_frame, text="StyleGAN3", variable=self.model_type, value="SG3").grid(row=0, column=1,
                                                                                                   padx=5, pady=5)

        # Image selection
        image_frame = ttk.LabelFrame(self.root, text="Step 2: Select Images", padding=(10, 10))
        image_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Keep image previews fixed size
        self.src_image_preview = tk.Label(image_frame, width=100, height=100, background="white")
        self.src_image_preview.grid(row=0, column=2, padx=5, pady=5)
        self.hair_target_preview = tk.Label(image_frame, width=100, height=100, background="white")
        self.hair_target_preview.grid(row=1, column=2, padx=5, pady=5)
        self.color_target_preview = tk.Label(image_frame, width=100, height=100, background="white")
        self.color_target_preview.grid(row=2, column=2, padx=5, pady=5)

        ttk.Button(image_frame, text="Select Source Image", command=self.select_src_image).grid(row=0, column=0, padx=5,
                                                                                                pady=5)
        self.src_image_label = ttk.Label(image_frame, text="No file selected")
        self.src_image_label.grid(row=0, column=0, sticky="s", pady=(0, 20))

        ttk.Button(image_frame, text="Select Hair Style Target Image", command=self.select_hair_target_image).grid(
            row=1, column=0, padx=5, pady=5)
        self.hair_target_label = ttk.Label(image_frame, text="No file selected")
        self.hair_target_label.grid(row=1, column=0, sticky="s", pady=(0, 20))

        ttk.Button(image_frame, text="Select Color Target Image", command=self.select_color_target_image).grid(row=2,
                                                                                                               column=0,
                                                                                                               padx=5,
                                                                                                               pady=5)
        self.color_target_label = ttk.Label(image_frame, text="No file selected")
        self.color_target_label.grid(row=2, column=0, sticky="s", pady=(0, 20))

        # Generate button
        ttk.Button(image_frame, text="Generate", command=self.start_generate_thread).grid(row=3, column=0, columnspan=3,
                                                                                    pady=10)

        # Display area
        self.result_frame = ttk.LabelFrame(self.root, text="Step 3: Generated Result", padding=(10, 10))
        self.result_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=10, pady=5)

        self.result_image_label = tk.Label(self.result_frame, width=300, height=300, background="white")
        self.result_image_label.grid(row=0, column=0, padx=10, pady=10)

        # Output log area
        self.log_text = tk.Text(self.result_frame, wrap='word', height=10, width=15)
        self.log_text.grid(row=1, column=0, sticky="nsew")

        log_scrollbar = ttk.Scrollbar(self.result_frame, command=self.log_text.yview)
        self.log_text['yscrollcommand'] = log_scrollbar.set
        log_scrollbar.grid(row=1, column=1, sticky='ns')

        # Redirect stdout to the log text widget
        sys.stdout = RedirectStdout(self.log_text)

        # Set default empty images
        self.set_default_images()

    def set_default_images(self):
        default_img = Image.new('RGB', (100, 100), color='white')
        default_img_tk = ImageTk.PhotoImage(default_img)
        self.src_image_preview.config(image=default_img_tk)
        self.src_image_preview.image = default_img_tk
        self.hair_target_preview.config(image=default_img_tk)
        self.hair_target_preview.image = default_img_tk
        self.color_target_preview.config(image=default_img_tk)

        default_result_img = Image.new('RGB', (300, 300), color='white')
        default_result_img_tk = ImageTk.PhotoImage(default_result_img)
        self.result_image_label.config(image=default_result_img_tk)
        self.result_image_label.image = default_result_img_tk

    def select_src_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if path:
            self.src_image_path = path
            self.display_thumbnail(path, self.src_image_preview)
            self.src_image_label.config(text=os.path.basename(path))

    def select_hair_target_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if path:
            self.hair_target_path = path
            self.display_thumbnail(path, self.hair_target_preview)
            self.hair_target_label.config(text=os.path.basename(path))

    def select_color_target_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if path:
            self.color_target_path = path
            self.display_thumbnail(path, self.color_target_preview)
            self.color_target_label.config(text=os.path.basename(path))

    def display_thumbnail(self, image_path, label):
        try:
            img = Image.open(image_path)
            img.thumbnail((100, 100))
            img_tk = ImageTk.PhotoImage(img)
            label.config(image=img_tk)
            label.image = img_tk
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image thumbnail: {e}")

    def start_generate_thread(self):
        generate_thread = threading.Thread(target=self.generate_result)
        generate_thread.start()

    def generate_result(self):
        # Check that all selections are made
        if not self.model_type.get():
            messagebox.showerror("Error", "Please select a model type.")
            return
        if not self.src_image_path or not self.hair_target_path or not self.color_target_path:
            messagebox.showerror("Error", "Please select all required images.")
            return

        # Construct output filename
        output_filename = f"{os.path.splitext(os.path.basename(self.src_image_path))[0]}_{os.path.splitext(os.path.basename(self.hair_target_path))[0]}_{os.path.splitext(os.path.basename(self.color_target_path))[0]}_{self.model_type.get().lower()}.jpg"
        output_filepath = os.path.join("/content/drive/My Drive/Hair/output_image", output_filename)

        # If output file already exists, load and display it
        if os.path.isfile(output_filepath):
            try:
                result_img = Image.open(output_filepath)
                result_img.thumbnail((300, 300))
                img_tk = ImageTk.PhotoImage(result_img)
                self.result_image_label.config(image=img_tk)
                self.result_image_label.image = img_tk
                return
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load existing result: {e}")
                return

        try:
            # Load source image
            src_image = Image.open(self.src_image_path)
            src_image_tensor = transforms.ToTensor()(src_image).unsqueeze(0).cuda()
            src_name = os.path.splitext(os.path.basename(self.src_image_path))[0]
            src_name2 = os.path.basename(self.src_image_path)
            global_cond = os.path.splitext(os.path.basename(self.src_image_path))[0]
            global_cond2 = os.path.basename(self.hair_target_path)

            color_cond = os.path.splitext(os.path.basename(self.hair_target_path))[0]
            color_cond2 = os.path.basename(self.hair_target_path)
            # Load the chosen model type
            if self.model_type.get() == "SG2":

                image_transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                g_ema, mean_latent_code, seg = load_base_models(self.opts)
                ii2s = Embedding(self.opts, g_ema, mean_latent_code[0, 0])

                if not os.path.isfile(os.path.join(self.opts.src_latent_dir, f"{src_name}.npz")):
                    inverted_latent_w_plus, inverted_latent_F = ii2s.invert_image_in_FS(
                        image_path=f'{self.opts.src_img_dir}/{src_name2}')
                    save_latent_path = os.path.join(self.opts.src_latent_dir, f'{src_name}.npz')
                    np.savez(save_latent_path, latent_in=inverted_latent_w_plus.detach().cpu().numpy(),
                             latent_F=inverted_latent_F.detach().cpu().numpy())

                src_latent = torch.from_numpy(np.load(f'{self.opts.src_latent_dir}/{src_name}.npz')["latent_in"]).cuda()
                src_feature = torch.from_numpy(np.load(f'{self.opts.src_latent_dir}/{src_name}.npz')["latent_F"]).cuda()
                src_image = image_transform(Image.open(f'{self.opts.src_img_dir}/{src_name2}').convert('RGB')).unsqueeze(
                    0).cuda()
                input_mask = torch.argmax(seg(src_image)[1], dim=1).long().clone().detach()

                bald_proxy = BaldProxy(g_ema, self.opts.bald_path)
                text_proxy = TextProxy(self.opts, g_ema, seg, mean_latent_code)
                ref_proxy = RefProxy(self.opts, g_ema, seg, ii2s)
                sketch_proxy = SketchProxy(g_ema, mean_latent_code, self.opts.sketch_path)
                color_proxy = ColorProxy(self.opts, g_ema, seg)

                def hairstyle_editing(global_cond=None, local_sketch=False, paint_the_mask=False, \
                                      src_latent=src_latent, src_feature=src_feature, input_mask=input_mask,
                                      src_image=src_image, \
                                      latent_global=None, latent_local=None, latent_bald=None, local_blending_mask=None,
                                      painted_mask=None):
                    if paint_the_mask:
                        modified_mask = painting_mask(input_mask)
                        input_mask = torch.from_numpy(modified_mask).unsqueeze(0).cuda().long().clone().detach()
                        vis_modified_mask = vis_seg(modified_mask)

                        painted_mask = input_mask

                    if local_sketch:
                        latent_local, local_blending_mask, visual_local_list = sketch_proxy(input_mask)


                    if global_cond is not None:
                        assert isinstance(global_cond, str)
                        latent_bald, visual_bald_list = bald_proxy(src_latent)


                        if global_cond.endswith('.jpg') or global_cond.endswith('.png'):
                            latent_global, visual_global_list = ref_proxy(global_cond, src_image,
                                                                          painted_mask=painted_mask)
                        else:
                            latent_global, visual_global_list = text_proxy(global_cond, src_image, from_mean=True,
                                                                           painted_mask=painted_mask)


                    src_feature, edited_hairstyle_img = hairstyle_feature_blending(g_ema, seg, src_latent, src_feature,
                                                                                   input_mask, latent_bald=latent_bald, \
                                                                                   latent_global=latent_global,
                                                                                   latent_local=latent_local,
                                                                                   local_blending_mask=local_blending_mask)
                    return src_feature, edited_hairstyle_img

                src_feature, edited_hairstyle_img = hairstyle_editing(global_cond=global_cond2, local_sketch=False,
                                                                      paint_the_mask=False)
                visual_color_list, visual_final_list = color_proxy(color_cond2, edited_hairstyle_img, src_latent,
                                                                   src_feature)

                edited_hairstyle_img = visual_final_list[-1]


            elif self.model_type.get() == "SG3":
                # 모델 로드
                image_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

                generator, opts_sg3, seg, avg_img = load_sg3_models(self.opts)

                # 임베딩 및 손실 함수
                re4e = Embedding_sg3(self.opts, generator)
                loss_builder = EmbeddingLossBuilder(self.opts)

                # 편집 도구 로드
                ref_proxy = RefProxy_sg3(self.opts, generator, seg, re4e)
                refine_proxy = RefineProxy(self.opts, generator, seg)
                color_proxy = ColorProxy_sg3(self.opts, generator, seg)

                # InterfaceGAN 초기화
                editor = FaceEditor(stylegan_generator=generator.decoder, generator_type=GeneratorType.ALIGNED)
                edit_direction = ['Bald', 'pose']

                if not os.path.isfile(os.path.join(self.opts.ref_latent_dir, f"{src_name}_sg3.npy")):
                    src_latent = re4e.invert_image_in_W(image_path=os.path.join(self.opts.src_img_dir, f'{src_name2}'),
                                                        device='cuda', avg_image=avg_img)
                else:
                    src_latent = torch.from_numpy(np.load(f'{self.opts.ref_latent_dir}/{src_name}_sg3.npy')).cuda()
                src_pil = Image.open(f'{self.opts.src_img_dir}/{src_name2}').convert('RGB')
                src_image = image_transform(src_pil).unsqueeze(0).cuda()
                input_mask = torch.argmax(seg(src_image)[1], dim=1).long().clone().detach()

                print(f"Performing edit for {edit_direction[0]}...")
                bald_feat, edit_latents = editor.edit(src_image,
                                                      latents=src_latent,
                                                      direction=edit_direction[0],
                                                      factor=5,
                                                      user_transforms=None,
                                                      apply_user_transformations=False)
                latent_bald = edit_latents[-1].unsqueeze(0)



                latent_global, visual_global_list = ref_proxy(global_cond2, src_image=src_image, m_style=6)

                blend_source, edited_hairstyle_img, edited_latent = hairstyle_feature_blending_sg3(generator, seg,
                                                                                                   src_image,
                                                                                                   input_mask,
                                                                                                   latent_bald,
                                                                                                   latent_global,
                                                                                                   avg_img)

                target_mask = seg(blend_source)[1]
                final_image, blended_latent, visual_list = refine_proxy(blended_latent=edited_latent,
                                                                        src_image=src_image,
                                                                        ref_img=visual_global_list[-1],
                                                                        target_mask=target_mask)
                visual_final_list = color_proxy(color_cond, final_image, blended_latent, blend_source)

                edited_hairstyle_img = visual_final_list[-1]



            else:
                raise ValueError("Invalid model type selected.")

            # Save the result to file
            edited_hairstyle_img = (edited_hairstyle_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            result_img = Image.fromarray(edited_hairstyle_img)
            result_img.save(output_filepath)

            # Convert the result to an image format and display it
            result_img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(result_img)
            self.result_image_label.config(image=img_tk)
            self.result_image_label.image = img_tk
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate result: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = HairStyleApp(root)
    root.mainloop()
