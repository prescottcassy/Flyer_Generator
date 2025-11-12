import tkinter as tk
from PIL import Image, ImageTk
import os

image_path = os.path.join(os.path.dirname(__file__), '..', 'generated_image.png')

def display_image(image_path):
    """Check if the image exists and display it in a Tkinter window."""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    tk_window = tk.Tk()
    tk_window.title("Generated Image Viewer")
    img = Image.open(image_path)
    tk_img = ImageTk.PhotoImage(img)
    label = tk.Label(tk_window, image=tk_img)
    label.pack()
    tk_window.mainloop()

if __name__ == "__main__":
    display_image(image_path)