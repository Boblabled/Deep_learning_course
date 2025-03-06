from ultralytics import YOLO

if __name__ == '__main__':
    # Загрузка предобученной модели
    model = YOLO('yolo11n.pt')  # 'n' - nano версия (лёгкая)
    results = model.train(data='rock-paper-scissors-1/data.yaml', epochs=15, imgsz=640, batch=8)

    metrics = model.val()  # Оценка на валидационной выборке
    print("mAP50: ", metrics.box.map)  # mAP50
    print("mAP50-95: ", metrics.box.map50)  # mAP50-95

    result = model('test_image.jpg')
    result.show()  # Визуализация результата
    result.save()
