from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolo26n.pt")   # 或你的权重路径
    results = model.val(
        data=r"D:\ultralytics-main\coco_2017.yaml",
        split="test",
        imgsz=640,
        device=0,
        save_json=True,
        project=r"D:\ultralytics-main\runs\detect",
        name="test"
    )
