from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26n.pt")  
    metrics = model.val(
        data=r"D:\ultralytics-main\coco_2017.yaml",
        split="val",
        imgsz=640,
        device=0,
        save_json=True,
          
    )
    print("mAP50-95 =", metrics.box.map)
    print("mAP50    =", metrics.box.map50)
    print("mAP75    =", metrics.box.map75)
