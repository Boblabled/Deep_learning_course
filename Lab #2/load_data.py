from roboflow import Roboflow

if __name__ == '__main__':
    rf = Roboflow(api_key="Jd7gvF1uzj6LCeYLLNkc")
    project = rf.workspace("test-lcktp").project("rock-paper-scissors-sxsw-5pntk")
    version = project.version(1)
    dataset = version.download("yolov11")
