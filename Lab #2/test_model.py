from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/train/weights/best.pt')  # 'n' - nano версия (лёгкая)

    results = model('Test/test_image_2.jpg')
    for result in results:
        result.show()
        result.save('Test/test_image_detected.jpg')