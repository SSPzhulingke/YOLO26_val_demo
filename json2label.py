from ultralytics.data.converter import convert_coco

if __name__ == "__main__":
    convert_coco(
        labels_dir=r"D:\ultralytics-main\datasets\coco\annotations",
        save_dir=r"D:\ultralytics-main\datasets",
        cls91to80=True
    )
