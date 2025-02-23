import cv2
import numpy as np
import potrace
import gradio as gr
from fastapi import FastAPI
from gradio.routes import mount_gradio_app

app = FastAPI()


def generate_svg(path, size, fill_color, stroke_color, stroke_width):
    """Генерация SVG с настраиваемыми цветами и обводкой"""
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size[1]} {size[0]}" width="{size[1]}" height="{size[0]}">\n'

    stroke_attr = f'stroke="{stroke_color}" stroke-width="{stroke_width}"' if stroke_width > 0 else 'stroke="none"'

    for curve in path:
        svg += f'<path d="M {curve.start_point.x:.2f} {curve.start_point.y:.2f} '
        for segment in curve.segments:
            if segment.is_corner:
                svg += f'L {segment.c.x:.2f} {segment.c.y:.2f} '
                svg += f'L {segment.end_point.x:.2f} {segment.end_point.y:.2f} '
            else:
                svg += f'C {segment.c1.x:.2f} {segment.c1.y:.2f} '
                svg += f'{segment.c2.x:.2f} {segment.c2.y:.2f} '
                svg += f'{segment.end_point.x:.2f} {segment.end_point.y:.2f} '
        svg += f' Z" fill="{fill_color}" {stroke_attr}/>\n'

    return svg + '</svg>'


def process_image(
        image_path,
        blur_size,
        block_size,
        c_value,
        turdsize,
        turnpolicy,
        alphamax,
        invert_image,
        fill_color,
        stroke_color,
        stroke_width
):
    """Обработка изображения с настройками"""
    # Загрузка и предобработка
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Обработка размера размытия
    blur_size = int(blur_size)
    if blur_size % 2 == 0:
        blur_size += 1
    if blur_size < 1:
        blur_size = 1
    blurred = cv2.medianBlur(gray, blur_size)

    # Бинаризация
    block_size = int(block_size)
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value
    )

    # Инвертирование изображения
    if invert_image:
        binary = 255 - binary

    # Трассировка
    data = np.asarray(binary, dtype=np.uint8)
    bitmap = potrace.Bitmap(data)

    # Преобразование политики поворота
    policy_map = {
        "majority": potrace.POTRACE_TURNPOLICY_MAJORITY,
        "minority": potrace.POTRACE_TURNPOLICY_MINORITY,
        "black": potrace.POTRACE_TURNPOLICY_BLACK,
        "white": potrace.POTRACE_TURNPOLICY_WHITE,
        "right": potrace.POTRACE_TURNPOLICY_RIGHT,
        "left": potrace.POTRACE_TURNPOLICY_LEFT,
        "random": potrace.POTRACE_TURNPOLICY_RANDOM,
    }
    policy = policy_map.get(turnpolicy, potrace.POTRACE_TURNPOLICY_MAJORITY)

    path = bitmap.trace(
        turdsize=int(turdsize),
        turnpolicy=policy,
        alphamax=float(alphamax)
    )

    # Генерация SVG
    svg_content = generate_svg(path, binary.shape, fill_color, stroke_color, stroke_width)
    with open("result.svg", "w") as f:
        f.write(svg_content)

    return "result.svg"


# Gradio интерфейс
interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="filepath", label="Изображение"),
        gr.Slider(1, 15, value=7, step=2, label="Размер размытия (нечетное)"),
        gr.Slider(3, 31, value=15, step=2, label="Размер блока бинаризации (нечетное)"),
        gr.Slider(0, 20, value=4, label="Параметр C для бинаризации"),
        gr.Slider(0, 1000, value=50, label="Минимальный размер шума (turdsize)"),
        gr.Dropdown(
            choices=["majority", "minority", "black", "white", "right", "left", "random"],
            value="majority",
            label="Политика поворота"
        ),
        gr.Slider(0.0, 2.0, value=0.5, step=0.1, label="Максимальный угол (alphamax)"),
        gr.Checkbox(value=False, label="Инвертировать изображение"),
        gr.ColorPicker(value="#000000", label="Цвет заливки"),
        gr.ColorPicker(value="#000000", label="Цвет обводки"),
        gr.Slider(0, 10, value=0, step=1, label="Толщина обводки"),
    ],
    outputs=gr.File(label="SVG файл"),
    title="Конвертер в SVG с настройками",
    description="Настройте параметры преобразования и визуализации SVG"
)

# Интеграция с FastAPI
app = mount_gradio_app(app, interface, path="/")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)