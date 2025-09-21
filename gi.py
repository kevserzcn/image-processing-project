import numpy as np
from tkinter import *
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Görüntü İşleme Uygulaması")
        self.root.geometry("1200x800")

        # Görüntü değişkenleri
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        self.image2_path = None

        # Parametre değişkenleri
        self.x_var = IntVar(value=0)
        self.y_var = IntVar(value=0)
        self.width_var = IntVar(value=100)
        self.height_var = IntVar(value=100)

        self.noise_type = StringVar(value='gaussian')
        self.noise_ratio = DoubleVar(value=0.1)

        self.contrast_var = DoubleVar(value=1.0)
        self.brightness_var = IntVar(value=0)

        self.kernel_var = IntVar(value=3)
        self.morph_var = StringVar(value='erosion')

        self.rotate_angle = IntVar(value=0)
        self.scale_width = DoubleVar(value=1.0)
        self.scale_height = DoubleVar(value=1.0)

        self.noise_removal_method = StringVar(value='Gaussian Blur')
        self.noise_removal_kernel = IntVar(value=3)

        self.style = ttk.Style()
        self.style.configure("TFrame", background="#ffe6f2")  # Light pink background
        self.style.configure("TLabel", background="#ffe6f2", font=("Comic Sans MS", 10, "bold"), foreground="#cc0066")
        self.style.configure("TButton", font=("Comic Sans MS", 10, "bold"), padding=5, background="#ff99cc", foreground="#660033")
        self.style.configure("TLabelFrame", background="#ffccff", font=("Comic Sans MS", 12, "bold"), foreground="#99004d")

        # Ana paneller
        self.create_main_panel()
        self.create_controls_panel()

    def create_main_panel(self):
        self.main_frame = ttk.Frame(self.root, padding="5")
        self.main_frame.grid(row=0, column=0, sticky=(N, W, E, S))

        # Görüntü seçme butonları
        self.image_button = ttk.Button(self.main_frame, text="Görüntü Seç", command=self.select_image)
        self.image_button.grid(row=0, column=0, padx=2, pady=2)

        self.image2_button = ttk.Button(self.main_frame, text="İkinci Görüntü Seç", command=self.select_second_image)
        self.image2_button.grid(row=0, column=1, padx=2, pady=2)

        # Etiketler
        self.original_label = ttk.Label(self.main_frame, text="Orijinal Görüntü")
        self.original_label.grid(row=1, column=0, padx=2, pady=2)

        self.processed_label = ttk.Label(self.main_frame, text="İşlenmiş Görüntü")
        self.processed_label.grid(row=1, column=1, padx=2, pady=2)

        # Canvaslar
        self.original_canvas = Canvas(self.main_frame, width=300, height=300, bg='#ffe6f2', highlightthickness=1, highlightbackground="#cc0066")
        self.original_canvas.grid(row=2, column=0, padx=2, pady=2)

        self.processed_canvas = Canvas(self.main_frame, width=300, height=300, bg='#ffe6f2', highlightthickness=1, highlightbackground="#cc0066")
        self.processed_canvas.grid(row=2, column=1, padx=2, pady=2)

        # Parametreler için alt çerçeve
        self.params_frame = ttk.LabelFrame(self.main_frame, text="Parametreler", padding="5")
        self.params_frame.grid(row=3, column=0, columnspan=2, sticky=(W, E), pady=5)

        # Kaydet butonu
        self.save_button = ttk.Button(self.main_frame, text="İşlenmiş Görüntüyü Kaydet", command=self.save_image)
        self.save_button.grid(row=4, column=1, padx=2, pady=2)

    def create_controls_panel(self):
        """Create the controls panel with operation buttons."""
        self.controls_frame = ttk.LabelFrame(self.root, text="İşlem Seçenekleri", padding="5")
        self.controls_frame.grid(row=0, column=1, sticky=(N, W, E, S), padx=5, pady=5)

        operations = [
            "Binary Dönüşüm", "Canny Edge Detection", "Görüntü Kırpma", "Renk Uzayı Dönüşümleri",
            "Gürültü Ekleme", "Gri Tonlama", "Kontrast Ayarları", "Morfolojik İşlemler",
            "Histogram Eşitleme", "Histogram Germe", "Histogram Görselleştirme", "Görüntü Döndürme",
            "Görüntü Ölçeklendirme", "Zoom In/Out", "Aritmetik İşlemler", "Çift Eşikleme",
            "Median Filter", "Motion Filter", "Gürültü Temizleme"
        ]

        for i, op in enumerate(operations):
            ttk.Button(self.controls_frame, text=op, command=lambda op=op: self.update_controls(op)).grid(row=i, column=0, sticky=W, pady=1)

    def select_image(self):
        self.image_path = filedialog.askopenfilename(
        title="Görüntü Seç",
        filetypes=[("Image files", "*.jpg *.png *.bmp *.jpeg")]
    )
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            self.display_image(self.image_path, self.original_canvas, image_type="original")

    def select_second_image(self):
        self.image2_path = filedialog.askopenfilename(
        title="İkinci Görüntü Seç",
        filetypes=[("Image files", "*.jpg *.png *.bmp *.jpeg")]
    )
        if self.image2_path:
            self.display_image(self.image2_path, self.processed_canvas, "processed")

    def save_image(self):
        if self.processed_image is not None:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp")]
            )
            if save_path:
                cv2.imwrite(save_path, self.processed_image)
                messagebox.showinfo("Başarılı", "Görüntü başarıyla kaydedildi!")

    def display_image(self, image_path, canvas, image_type="original"):
        img_cv = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((400, 400))
        img_tk = ImageTk.PhotoImage(img_pil)

        if image_type == "original":
            self.original_tk = img_tk  # referans tut
        else:
            self.processed_tk = img_tk  # referans tut

        canvas.delete("all")  # önceki görüntüyü temizle
        canvas.create_image(0, 0, anchor=NW, image=img_tk)

    def apply_operation(self, operation):
        """Apply the selected operation to the image."""
        if self.original_image is None:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin.")
            return

        image = self.original_image.copy()  # Use the loaded original image

        if operation == "Binary Dönüşüm":
            threshold = self.threshold_var.get()
            self.processed_image = self.binary_conversion(image, threshold)
        elif operation == "Canny Edge Detection":
            low_threshold = self.low_threshold_var.get()
            high_threshold = self.high_threshold_var.get()
            self.processed_image = self.canny_edge_detection(image, low_threshold, high_threshold)
        elif operation == "Görüntü Kırpma":
            x = self.x_var.get()
            y = self.y_var.get()
            width = self.width_var.get()
            height = self.height_var.get()
            self.processed_image = image[y:y+height, x:x+width]
        elif operation == "Renk Uzayı Dönüşümleri":
            target_space = self.color_space_var.get()
            self.processed_image = self.color_space_conversion(image, target_space)
        elif operation == "Gürültü Ekleme":
            noise_type = self.noise_type.get()
            ratio = self.noise_ratio.get()
            self.processed_image = self.add_noise(image, noise_type, ratio)
        elif operation == "Gri Tonlama":
            self.processed_image = self.gray_scale(image)
        elif operation == "Kontrast Ayarları":
            alpha = self.contrast_var.get()
            beta = self.brightness_var.get()
            self.processed_image = self.adjust_contrast(image, alpha, beta)
        elif operation == "Morfolojik İşlemler":
            kernel_size = self.kernel_var.get()
            morph_operation = self.morph_var.get()
            self.processed_image = self.apply_morphological(image, morph_operation, kernel_size)
        elif operation == "Histogram Eşitleme":
            self.processed_image = self.histogram_equalization(image)
        elif operation == "Histogram Görselleştirme":
            option = self.histogram_option.get()
            self.display_histogram(image, option)
            return  # No processed image to display
        elif operation == "Görüntü Döndürme":
            angle = self.rotate_angle.get()
            direction = self.rotation_direction.get()
            clockwise = direction == "Saat Yönü"
            self.processed_image = self.rotate_image(image, angle, clockwise)
        elif operation == "Görüntü Ölçeklendirme":
            width_scale = self.scale_width.get()
            height_scale = self.scale_height.get()
            self.processed_image = self.scale_image(image, width_scale, height_scale)
        elif operation == "Aritmetik İşlemler":
            if self.image2_path is None:
                messagebox.showerror("Hata", "Lütfen ikinci bir görüntü seçin.")
                return
            operation_type = self.arithmetic_operation_type.get()
            second_image = cv2.imread(self.image2_path)
            if second_image is None:
                messagebox.showerror("Hata", "İkinci görüntü yüklenemedi.")
                return
            self.processed_image = self.arithmetic_operations(image, second_image, operation_type)
        elif operation == "Zoom In/Out":
            zoom_factor = self.zoom_factor_var.get()
            if zoom_factor <= 0 or zoom_factor > 10:  # Validate zoom factor
                messagebox.showerror("Hata", "Zoom faktörü 0.1 ile 10 arasında olmalıdır.")
                return
            try:
                self.processed_image = self.zoom_in_out(image, zoom_factor)
            except Exception as e:
                messagebox.showerror("Hata", f"Zoom işlemi başarısız: {str(e)}")
                return
        elif operation == "Çift Eşikleme":
            low_threshold = self.low_threshold_var.get()
            high_threshold = self.high_threshold_var.get()
            self.processed_image = self.double_threshold(image, low_threshold, high_threshold)
        elif operation == "Median Filter":
            kernel_size = self.median_kernel_var.get()
            if kernel_size % 2 == 0 or kernel_size < 1:
                messagebox.showerror("Hata", "Kernel boyutu tek sayı ve 1'den büyük olmalıdır.")
                return
            self.processed_image = self.median_filter(image, kernel_size)
        elif operation == "Motion Filter":
            kernel_size = self.motion_kernel_var.get()
            if kernel_size < 1:
                messagebox.showerror("Hata", "Kernel boyutu 1 veya daha büyük olmalıdır.")
                return
            self.processed_image = self.motion_filter(image, kernel_size)
        elif operation == "Gürültü Temizleme":
            method = self.noise_removal_method.get()
            kernel_size = self.noise_removal_kernel.get()
            if kernel_size % 2 == 0 or kernel_size < 1:
                messagebox.showerror("Hata", "Kernel boyutu tek sayı ve 1'den büyük olmalıdır.")
                return
            self.processed_image = self.remove_noise(self.original_image, method, kernel_size)
        elif operation == "Histogram Germe":
            self.processed_image = self.stretch_histogram(image)

        # Display the processed image
        self.display_image_from_cv2(self.processed_image, self.processed_canvas)

    def update_controls(self, operation):
        """Update parameter controls based on the selected operation."""
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        if operation == "Binary Dönüşüm":
            ttk.Label(self.params_frame, text="Eşik Değeri:").grid(row=0, column=0, sticky=W)
            self.threshold_var = IntVar(value=127)
            Scale(self.params_frame, from_=0, to=255, orient=HORIZONTAL, variable=self.threshold_var, length=200, bg="#ffccff", fg="#99004d").grid(row=0, column=1)
        elif operation == "Canny Edge Detection":
            ttk.Label(self.params_frame, text="Alt Eşik:").grid(row=0, column=0, sticky=W)
            self.low_threshold_var = IntVar(value=100)
            Scale(self.params_frame, from_=0, to=255, orient=HORIZONTAL, variable=self.low_threshold_var, length=200, bg="#ffccff", fg="#99004d").grid(row=0, column=1)

            ttk.Label(self.params_frame, text="Üst Eşik:").grid(row=1, column=0, sticky=W)
            self.high_threshold_var = IntVar(value=200)
            Scale(self.params_frame, from_=0, to=255, orient=HORIZONTAL, variable=self.high_threshold_var, length=200, bg="#ffccff", fg="#99004d").grid(row=1, column=1)
        elif operation == "Görüntü Kırpma":
            ttk.Label(self.params_frame, text="X Başlangıç:").grid(row=0, column=0, sticky=W)
            ttk.Entry(self.params_frame, textvariable=self.x_var).grid(row=0, column=1)
            ttk.Label(self.params_frame, text="Y Başlangıç:").grid(row=1, column=0, sticky=W)
            ttk.Entry(self.params_frame, textvariable=self.y_var).grid(row=1, column=1)
            ttk.Label(self.params_frame, text="Genişlik:").grid(row=2, column=0, sticky=W)
            ttk.Entry(self.params_frame, textvariable=self.width_var).grid(row=2, column=1)
            ttk.Label(self.params_frame, text="Yükseklik:").grid(row=3, column=0, sticky=W)
            ttk.Entry(self.params_frame, textvariable=self.height_var).grid(row=3, column=1)
        elif operation == "Morfolojik İşlemler":
            ttk.Label(self.params_frame, text="İşlem Türü:").grid(row=0, column=0, sticky=W)
            self.morph_var = StringVar(value="erosion")
            morph_operations = ["erosion", "dilation", "opening", "closing", "gradient", "tophat", "blackhat"]
            ttk.Combobox(self.params_frame, textvariable=self.morph_var, values=morph_operations, state="readonly").grid(row=0, column=1)

            ttk.Label(self.params_frame, text="Kernel Boyutu:").grid(row=1, column=0, sticky=W)
            self.kernel_var = IntVar(value=3)
            ttk.Entry(self.params_frame, textvariable=self.kernel_var).grid(row=1, column=1)
        elif operation == "Zoom In/Out":
            ttk.Label(self.params_frame, text="Zoom Faktörü:").grid(row=0, column=0, sticky=W)
            self.zoom_factor_var = DoubleVar(value=1.0)
            Scale(self.params_frame, from_=0.1, to=10.0, resolution=0.1, orient=HORIZONTAL, variable=self.zoom_factor_var, length=200, bg="#ffccff", fg="#99004d").grid(row=0, column=1)
        elif operation == "Aritmetik İşlemler":
            ttk.Label(self.params_frame, text="İşlem Türü:").grid(row=0, column=0, sticky=W)
            self.arithmetic_operation_type = StringVar(value="Toplama")
            operations = ["Toplama", "Çıkartma", "Çarpma", "Bölme"]
            ttk.Combobox(self.params_frame, textvariable=self.arithmetic_operation_type, values=operations, state="readonly").grid(row=0, column=1)
        elif operation == "Renk Uzayı Dönüşümleri":
            ttk.Label(self.params_frame, text="Hedef Renk Uzayı:").grid(row=0, column=0, sticky=W)
            self.color_space_var = StringVar(value="HSV")
            color_spaces = ["HSV", "GRAY", "LAB"]
            ttk.Combobox(self.params_frame, textvariable=self.color_space_var, values=color_spaces, state="readonly").grid(row=0, column=1)
        elif operation == "Çift Eşikleme":
            ttk.Label(self.params_frame, text="Alt Eşik:").grid(row=0, column=0, sticky=W)
            self.low_threshold_var = IntVar(value=50)
            ttk.Entry(self.params_frame, textvariable=self.low_threshold_var).grid(row=0, column=1)

            ttk.Label(self.params_frame, text="Üst Eşik:").grid(row=1, column=0, sticky=W)
            self.high_threshold_var = IntVar(value=150)
            ttk.Entry(self.params_frame, textvariable=self.high_threshold_var).grid(row=1, column=1)
        elif operation == "Median Filter":
            ttk.Label(self.params_frame, text="Kernel Boyutu:").grid(row=0, column=0, sticky=W)
            self.median_kernel_var = IntVar(value=3)
            ttk.Entry(self.params_frame, textvariable=self.median_kernel_var).grid(row=0, column=1)
        elif operation == "Görüntü Döndürme":
            ttk.Label(self.params_frame, text="Dönüş Açısı:").grid(row=0, column=0, sticky=W)
            self.rotate_angle = IntVar(value=0)
            ttk.Entry(self.params_frame, textvariable=self.rotate_angle).grid(row=0, column=1)

            ttk.Label(self.params_frame, text="Yön:").grid(row=1, column=0, sticky=W)
            self.rotation_direction = StringVar(value="Saat Yönü")
            directions = ["Saat Yönü", "Saat Yönünün Tersine"]
            ttk.Combobox(self.params_frame, textvariable=self.rotation_direction, values=directions, state="readonly").grid(row=1, column=1)
        elif operation == "Gürültü Ekleme":
            ttk.Label(self.params_frame, text="Gürültü Türü:").grid(row=0, column=0, sticky=W)
            self.noise_type = StringVar(value="gaussian")
            noise_types = ["gaussian", "salt_pepper", "poisson", "speckle"]
            ttk.Combobox(self.params_frame, textvariable=self.noise_type, values=noise_types, state="readonly").grid(row=0, column=1)

            ttk.Label(self.params_frame, text="Gürültü Oranı:").grid(row=1, column=0, sticky=W)
            self.noise_ratio = DoubleVar(value=0.1)
            ttk.Entry(self.params_frame, textvariable=self.noise_ratio).grid(row=1, column=1)
        elif operation == "Motion Filter":
            ttk.Label(self.params_frame, text="Kernel Boyutu:").grid(row=0, column=0, sticky=W)
            self.motion_kernel_var = IntVar(value=5)
            ttk.Entry(self.params_frame, textvariable=self.motion_kernel_var).grid(row=0, column=1)
        elif operation == "Histogram Görselleştirme":
            ttk.Label(self.params_frame, text="Histogram Görselleştirme Seçeneği:").grid(row=0, column=0, sticky=W)
            self.histogram_option = StringVar(value="Grayscale")
            options = ["Grayscale", "Color"]
            ttk.Combobox(self.params_frame, textvariable=self.histogram_option, values=options, state="readonly").grid(row=0, column=1)
        elif operation == "Gürültü Temizleme":
            ttk.Label(self.params_frame, text="Yöntem:").grid(row=0, column=0, sticky=W)
            noise_methods = ["Gaussian Blur", "Median Blur", "Bilateral Filter"]
            ttk.Combobox(self.params_frame, textvariable=self.noise_removal_method, values=noise_methods, state="readonly").grid(row=0, column=1)

            ttk.Label(self.params_frame, text="Kernel Boyutu:").grid(row=1, column=0, sticky=W)
            Scale(self.params_frame, from_=1, to=31, resolution=2, orient=HORIZONTAL, variable=self.noise_removal_kernel, length=200, bg="#ffccff", fg="#99004d").grid(row=1, column=1)
        elif operation == "Histogram Germe":
            ttk.Label(self.params_frame, text="Histogram Germe işlemi için parametre gerekmez.").grid(row=0, column=0, sticky=W)

        # Add the "Uygula" button
        ttk.Button(self.params_frame, text="Uygula", command=lambda: self.apply_operation(operation)).grid(row=10, column=0, columnspan=2, pady=10)

    def normalize_image(self, image):
        """Normalize an image to the range [0, 255]."""
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val == 0:
            return np.zeros_like(image, dtype=np.uint8)
        normalized = (image - min_val) * (255.0 / (max_val - min_val))
        return normalized.astype(np.uint8)

    def display_image_from_cv2(self, cv2_image, canvas):
        """Display an OpenCV image on the given canvas."""
        if cv2_image is None or cv2_image.size == 0:
            messagebox.showerror("Hata", "Görüntü verisi boş!")
            return
        if cv2_image.dtype != np.uint8:
            cv2_image = self.normalize_image(cv2_image)
        if len(cv2_image.shape) == 2:  # Grayscale image
            img = Image.fromarray(cv2_image)
        else:  # Color image
            img = Image.fromarray(cv2_image)
        img = img.resize((400, 400), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.delete("all")
        canvas.create_image(0, 0, image=imgtk, anchor=NW)
        canvas.image = imgtk  # Keep a reference to avoid garbage collection

    def arithmetic_operations(self, img1, img2, operation):
        """Perform arithmetic operations on two images."""
        if img1.shape != img2.shape:
            # Resize img2 to match img1's dimensions
            img2 = resize_image(img2, img1.shape[1], img1.shape[0])

        if operation == 'Toplama':
            result = np.clip(img1 + img2, 0, 255).astype(np.uint8)
        elif operation == 'Çıkartma':
            result = np.clip(img1 - img2, 0, 255).astype(np.uint8)
        elif operation == 'Çarpma':
            result = np.clip(img1 * img2, 0, 255).astype(np.uint8)
        elif operation == 'Bölme':
            # Avoid division by zero
            img2 = np.where(img2 == 0, 1, img2)
            result = np.clip(img1 / img2, 0, 255).astype(np.uint8)
        else:
            messagebox.showerror("Hata", "Geçersiz aritmetik işlem!")
            return None
        return result

    def canny_edge_detection(self, image, low_threshold=100, high_threshold=200):
        if len(image.shape) == 3:
            image = self.gray_scale(image)
            
        # Gaussian blur
        kernel = np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ]) / 256
        
        blurred = self.apply_filter(image, kernel)
        
        # Gradient hesaplama
        sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        
        sobel_y = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])
        
        grad_x = self.apply_filter(blurred, sobel_x)
        grad_y = self.apply_filter(blurred, sobel_y)
        
        # Gradient magnitude ve direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Non-maximum suppression
        height, width = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                angle = direction[i, j] * 180 / np.pi
                angle = (angle + 180) % 180
                
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    q = magnitude[i, j+1]
                    r = magnitude[i, j-1]
                elif (22.5 <= angle < 67.5):
                    q = magnitude[i+1, j-1]
                    r = magnitude[i-1, j+1]
                elif (67.5 <= angle < 112.5):
                    q = magnitude[i+1, j]
                    r = magnitude[i-1, j]
                else:
                    q = magnitude[i-1, j-1]
                    r = magnitude[i+1, j+1]
                
                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    suppressed[i, j] = magnitude[i, j]
        
        # Double threshold
        strong_edges = np.zeros_like(suppressed)
        weak_edges = np.zeros_like(suppressed)
        
        strong_edges[suppressed >= high_threshold] = 255
        weak_edges[(suppressed >= low_threshold) & (suppressed < high_threshold)] = 255
        
        # Edge tracking
        final_edges = strong_edges.copy()
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                if weak_edges[i, j] == 255:
                    if np.any(strong_edges[i-1:i+2, j-1:j+2] == 255):
                        final_edges[i, j] = 255
        
        return final_edges

    def apply_filter(self, image, kernel):
        height, width = image.shape
        kernel_height, kernel_width = kernel.shape
        output = np.zeros_like(image)
        
        for i in range(height - kernel_height + 1):
            for j in range(width - kernel_width + 1):
                output[i + kernel_height // 2, j + kernel_width // 2] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)
        
        return output

    def gray_scale(self, image):
        if len(image.shape) == 3:
            # RGB to Grayscale conversion
            r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray.astype(np.uint8)
        return image

    def rgb_to_hsv(self, image):
        """Convert an RGB image to HSV."""
        if len(image.shape) == 3:
            r = image[:, :, 0] / 255.0
            g = image[:, :, 1] / 255.0
            b = image[:, :, 2] / 255.0

            max_val = np.maximum(np.maximum(r, g), b)
            min_val = np.minimum(np.minimum(r, g), b)
            diff = max_val - min_val

            h = np.zeros_like(r)
            mask = diff > 0
            h[mask & (max_val == r)] = (60 * ((g[mask & (max_val == r)] - b[mask & (max_val == r)]) / diff[mask & (max_val == r)]) + 360) % 360
            h[mask & (max_val == g)] = (60 * ((b[mask & (max_val == g)] - r[mask & (max_val == g)]) / diff[mask & (max_val == g)]) + 120) % 360
            h[mask & (max_val == b)] = (60 * ((r[mask & (max_val == b)] - g[mask & (max_val == b)]) / diff[mask & (max_val == b)]) + 240) % 360

            s = np.zeros_like(r)
            s[max_val > 0] = diff[max_val > 0] / max_val[max_val > 0]

            v = max_val

            h = (h / 360 * 255).astype(np.uint8)
            s = (s * 255).astype(np.uint8)
            v = (v * 255).astype(np.uint8)

            return np.stack([h, s, v], axis=-1)
        return image

    def color_space_conversion(self, image, color_space):
        """Convert an image to the specified color space."""
        if color_space == "HSV":
            return self.rgb_to_hsv(image)
        elif color_space == "GRAY":
            return self.gray_scale(image)
        elif color_space == "LAB":
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        else:
            messagebox.showerror("Hata", "Geçersiz renk uzayı seçimi!")
            return image

    def display_histogram(self, image, option="Grayscale"):
        """Display the histogram of the given image."""
        if option == "Grayscale":
            if len(image.shape) == 3:  # Convert to grayscale if the image is colored
                image = self.gray_scale(image)
            plt.figure("Histogram - Grayscale")
            plt.hist(image.ravel(), bins=256, range=[0, 256])
            plt.title("Histogram - Grayscale")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
        elif option == "Color":
            color = ('b', 'g', 'r')
            plt.figure("Histogram - Color")
            for i, col in enumerate(color):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
                plt.xlim([0, 256])
            plt.title("Histogram - Color")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
        plt.show()

    def scale_image(self, image, width_scale, height_scale):
        """Scale the image by the given width and height scale factors."""
        height, width = image.shape[:2]
        new_width = int(width * width_scale)
        new_height = int(height * height_scale)
        scaled_image = resize_image(image, new_width, new_height)
        return scaled_image

    def binary_conversion(self, image, threshold):
        """Convert an image to binary using the specified threshold."""
        gray_image = self.gray_scale(image)
        binary_image = (gray_image >= threshold) * 255
        return binary_image.astype(np.uint8)

    def add_noise(self, image, noise_type, ratio):
        """Add noise to an image."""
        row, col, ch = image.shape
        noisy_image = image.copy()
        if noise_type == "salt_pepper":
            num_salt = int(ratio * image.size * 0.5)
            num_pepper = int(ratio * image.size * 0.5)
            for _ in range(num_salt):
                x, y = np.random.randint(0, row), np.random.randint(0, col)
                noisy_image[x, y] = 255
            for _ in range(num_pepper):
                x, y = np.random.randint(0, row), np.random.randint(0, col)
                noisy_image[x, y] = 0
        elif noise_type == "gaussian":
            mean = 0
            sigma = ratio * 255
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
        return noisy_image

    def adjust_contrast(self, image, alpha, beta):
        """Adjust the contrast and brightness of an image."""
        return np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

    def apply_morphological(self, image, operation, kernel_size):
        """Apply morphological operations to an image."""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if operation == 'erosion':
            return self.erosion(image, kernel)
        elif operation == 'dilation':
            return self.dilation(image, kernel)
        elif operation == 'opening':
            return self.opening(image, kernel)
        elif operation == 'closing':
            return self.closing(image, kernel)
        elif operation == 'gradient':
            return self.gradient(image, kernel)
        elif operation == 'tophat':
            return self.tophat(image, kernel)
        elif operation == 'blackhat':
            return self.blackhat(image, kernel)
        else:
            messagebox.showerror("Hata", "Geçersiz morfolojik işlem!")
            return image

    def erosion(self, image, kernel):
        """Apply erosion to an image."""
        return self.apply_filter(image, kernel)

    def dilation(self, image, kernel):
        """Apply dilation to an image."""
        return self.apply_filter(image, kernel)

    def opening(self, image, kernel):
        """Apply opening to an image."""
        return self.apply_filter(image, kernel)

    def closing(self, image, kernel):
        """Apply closing to an image."""
        return self.apply_filter(image, kernel)

    def gradient(self, image, kernel):
        """Apply gradient to an image."""
        return self.apply_filter(image, kernel)

    def tophat(self, image, kernel):
        """Apply tophat to an image."""
        return self.apply_filter(image, kernel)

    def blackhat(self, image, kernel):
        """Apply blackhat to an image."""
        return self.apply_filter(image, kernel)

    def histogram_equalization(self, image):
        """Equalize the histogram of a grayscale image."""
        if len(image.shape) == 3:  # Convert to grayscale if the image is colored
            image = self.gray_scale(image)
        return self.equalize_hist(image)

    def equalize_hist(self, image):
        """Equalize the histogram of a grayscale image."""
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        return cdf[image]

    def rotate_image(self, image, angle, clockwise=True):
        """Rotate an image by the given angle in the specified direction."""
        if not clockwise:
            angle = -angle  # Reverse the angle for counterclockwise rotation
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated_image

    def zoom_in_out(self, image, zoom_factor):
        """Zoom in or out of the image based on the zoom factor."""
        height, width = image.shape[:2]
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)

        if new_width < 1 or new_height < 1:
            raise ValueError("Zoomed image dimensions are too small. Please use a larger zoom factor.")

        zoomed_image = resize_image(image, new_width, new_height)
        return zoomed_image

    def double_threshold(self, image, low_threshold, high_threshold):
        """Apply double thresholding to the image."""
        gray_image = self.gray_scale(image)
        binary_low = (gray_image >= low_threshold) * 255
        binary_high = (gray_image < high_threshold) * 255
        return np.bitwise_and(binary_low, binary_high).astype(np.uint8)

    def median_filter(self, image, kernel_size):
        """Apply a median filter to the image."""
        return self.apply_filter(image, np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size))

    def motion_filter(self, image, kernel_size):
        """Apply a motion filter to the image."""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return self.apply_filter(image, kernel)

    def remove_noise(self, image, method, kernel_size):
        """Remove noise from the image using the specified method."""
        if method == "Gaussian Blur":
            kernel = np.array([
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1]
            ]) / 256
            return self.apply_filter(image, kernel)
        elif method == "Median Blur":
            return self.median_filter(image, kernel_size)
        elif method == "Bilateral Filter":
            return self.apply_filter(image, np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size))
        else:
            messagebox.showerror("Hata", "Geçersiz gürültü temizleme yöntemi!")
            return image

    def stretch_histogram(self, image):
        """Perform histogram stretching on the image."""
        if len(image.shape) == 3:  # Convert to grayscale if the image is colored
            image = self.gray_scale(image)
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val == 0:
            return image  # Avoid division by zero
        stretched = (image - min_val) * (255.0 / (max_val - min_val))
        return stretched.astype(np.uint8)

if __name__ == "__main__":
    root = Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

    # Example usage
    img_path = "example_image.jpg"  # Replace with a valid image path
    if not os.path.exists(img_path):
        print(f"Error: The file '{img_path}' does not exist.")
        print("Please ensure the file exists in the current directory or provide a valid path.")
    else:
        binary = binary_conversion(img_path)
        if binary is not None:
            cv2.imshow('Binary', binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def binary_conversion(image_path):
    img = cv2.imread(image_path, 0)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary

def canny_edge_detection(image_path):
    img = cv2.imread(image_path, 0)
    edges = cv2.Canny(img, 100, 200)
    return edges

def crop_image(image_path, x, y, width, height):
    img = cv2.imread(image_path)
    cropped = img[y:y+height, x:x+width]
    return cropped

def motion_filter(image_path):
    img = cv2.imread(image_path)
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(img,-1,kernel)
    return dst

def color_space_conversion(image_path, space):
    img = cv2.imread(image_path)
    if space == 'hsv':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif space == 'gray':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif space == 'lab':
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return img

def add_noise(image_path, noise_type):
    img = cv2.imread(image_path)
    if noise_type == 'gaussian':
        row,col,ch= img.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = img + gauss
        return np.clip(noisy, 0, 255).astype('uint8')
    return img

def arithmetic_operations(img1_path, img2_path, operation):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if operation == 'add':
        result = cv2.add(img1, img2)
    elif operation == 'subtract':
        result = cv2.subtract(img1, img2)
    elif operation == 'multiply':
        result = cv2.multiply(img1, img2)
    elif operation == 'divide':
        result = cv2.divide(img1, img2)
    
    return result

def median_filter(image_path, kernel_size):
    img = cv2.imread(image_path)
    return cv2.medianBlur(img, kernel_size)

def morphological_operations(image_path, operation, kernel_size):
    img = cv2.imread(image_path, 0)
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    if operation == 'erosion':
        return cv2.erode(img, kernel, iterations = 1)
    elif operation == 'dilation':
        return cv2.dilate(img, kernel, iterations = 1)
    elif operation == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

def contrast_adjustment(image_path, alpha, beta):
    img = cv2.imread(image_path)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def histogram(image_path):
    img = cv2.imread(image_path, 0)
    return cv2.calcHist([img],[0],None,[256],[0,256])

def gray_scale(image_path):
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Örnek kullanım
if __name__ == "__main__":
    # Örnek görüntü işleme işlemleri
    img_path = "example_image.jpg"  # Replace with a valid image path
    if not os.path.exists(img_path):
        print(f"Error: The file '{img_path}' does not exist.")
    else:
        # Example usage
        binary = binary_conversion(img_path)
        if binary is not None:
            cv2.imshow('Binary', binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
