from PIL import Image
import os

from PIL import ImageDraw, ImageFont

if __name__ == '__main__':
    # Путь к папке с PNG-изображениями
    image_folder = 'log_gan/lightning_logs/version_0/result'

    images = []
    for i, filename in enumerate(sorted(os.listdir(image_folder)), start=1):
        if filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            draw.text((350 , 730), f"Эпоха: {i}", font=ImageFont.truetype("arial.ttf", 30), fill=(0, 0, 0))

            # Добавляем изображение в список
            images.append(img)

    output_gif_path = 'output.gif'
    images[0].save(output_gif_path,
                   save_all=True,
                   append_images=images[1:] + [images[-1]]*20,
                   duration=100,
                   loop=0)

    print(f"GIF сохранен как {output_gif_path}")