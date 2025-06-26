import tkinter as tk
from tkinter import ttk
import math

def draw_shape(pattern, scale):
    canvas.delete("all")
    w, h = 300, 500
    cx, cy = w // 2, h // 2
    s = scale * 40

    if pattern == 'Trianglet':
        points = [cx, cy-s, cx-s, cy+s, cx+s, cy+s]
        canvas.create_polygon(points, outline='black', fill='', width=2)
    elif pattern == 'Plaquette':
        points = [cx-s, cy-s, cx+s, cy-s, cx+s, cy+s, cx-s, cy+s]
        canvas.create_polygon(points, outline='black', fill='', width=2)
    elif pattern == 'Spinnerlet':
        r = s
        for i in range(3):
            angle = i * 120
            x = cx + r * math.cos(math.radians(angle))
            y = cy + r * math.sin(math.radians(angle))
            canvas.create_line(cx, cy, x, y, fill='black', width=2)
        canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline='black', width=2)
    elif pattern == 'Tiara':
        canvas.create_arc(cx-s, cy-s, cx+s, cy+s, start=0, extent=180, style=tk.ARC, width=2)
        canvas.create_line(cx-s, cy, cx+s, cy, fill='black', width=2)
    elif pattern == 'Pyramid':
        points = [cx, cy-s, cx-s, cy+s, cx+s, cy+s]
        canvas.create_polygon(points, outline='black', fill='', width=2)
        canvas.create_line(cx-s, cy+s, cx, cy, fill='black', width=2)
        canvas.create_line(cx+s, cy+s, cx, cy, fill='black', width=2)
    elif pattern == 'Diamond':
        points = [cx, cy-s, cx-s, cy, cx, cy+s, cx+s, cy]
        canvas.create_polygon(points, outline='black', fill='', width=2)

def on_pattern_button_click(pattern):
    global selected_pattern
    selected_pattern = pattern
    draw_shape(selected_pattern, scale_var.get())

def on_scale_change(value):
    draw_shape(selected_pattern, float(value))

def toggle_options():
    if not options_frame.winfo_ismapped():
        options_frame.pack(fill='both', expand=True)
        draw_shape(selected_pattern, scale_var.get())
    else:
        options_frame.pack_forget()

root = tk.Tk()
root.title("Bonding Pattern")
root.geometry("300x300")
root.resizable(False, False)

# Bonding Pattern 버튼
toggle_btn = ttk.Button(root, text="Bonding Pattern", command=toggle_options)
toggle_btn.pack(fill='x', pady=5)

# 옵션/슬라이더/캔버스가 들어갈 프레임
options_frame = ttk.Frame(root)

pattern_options = ['Trianglet', 'Plaquette', 'Spinnerlet', 'Tiara', 'Pyramid', 'Diamond']
selected_pattern = pattern_options[0]

# 서브 버튼들
for option in pattern_options:
    btn = ttk.Button(options_frame, text=option, command=lambda opt=option: on_pattern_button_click(opt))
    btn.pack(anchor='w', padx=20, pady=2)

scale_label = ttk.Label(options_frame, text="Scale")
scale_label.pack(anchor='w', padx=20)
scale_var = tk.DoubleVar(value=1.0)
scale_slider = ttk.Scale(options_frame, from_=0.5, to=2.0, orient='horizontal', variable=scale_var, command=on_scale_change)
scale_slider.pack(fill='x', padx=20, pady=10)

canvas = tk.Canvas(options_frame, width=300, height=200, bg='white')
canvas.pack(pady=5)

# 처음에는 옵션 프레임을 숨김
options_frame.pack_forget()

root.mainloop()