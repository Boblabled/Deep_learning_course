from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/train/weights/best.pt')  # 'n' - nano версия (лёгкая)

    image = 'Test/test_image.jpg'
    image = "rock-paper-scissors-1/valid/images/0098_png.rf.7fb8d3ddf906f1a9b84809a187ac0e4d.jpg"
    results = model(image)
    for result in results:
        result.show()
        result.save('Test/test_image_detected.jpg')