import json
from pathlib import Path
from typing import Union
import xml.etree.ElementTree as ET


CLASSES: list[str] = [
    "bee",
    "chicken",
    "cow",
    "creeper",
    "enderman",
    "fox",
    "frog",
    "ghast",
    "goat",
    "llama",
    "pig",
    "sheep",
    "skeleton",
    "spider",
    "turtle",
    "wolf",
    "zombie",
]


def _clip_bbox(
    xmin: int, ymin: int, xmax: int, ymax: int, width: int, height: int
) -> tuple[int, int, int, int]:
    xmin = max(0, min(xmin, width - 1))
    ymin = max(0, min(ymin, height - 1))

    xmax = max(0, min(xmax, width))
    ymax = max(0, min(ymax, height))

    return xmin, ymin, xmax, ymax


def _parse_xml(xml_path: Union[str, Path]) -> dict:
    xml_path: str = str(xml_path)

    dct = {}

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(float(size.findtext("width")))
    height = int(float(size.findtext("height")))
    dct["height"] = height
    dct["width"] = width
    dct["area"] = width * height

    boxes = []
    labels = []
    classes_names = []

    dct["file_name"] = root.findtext("filename")

    for member in root.findall("object"):
        if int((member).find("difficult").text) == 1:
            continue

        class_name: str = member.find("name").text
        if class_name not in CLASSES:
            continue

        id: int = CLASSES.index(class_name) + 1

        bndbox = member.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        xmin_upd, ymin_upd, xmax_upd, ymax_upd = _clip_bbox(
            xmin, ymin, xmax, ymax, width, height
        )
        if xmax_upd <= xmin_upd or ymax_upd <= ymin_upd:
            continue

        bw = xmax_upd - xmin_upd
        bh = ymax_upd - ymin_upd
        boxes.append([float(xmin_upd), float(ymin_upd), float(bw), float(bh)])
        labels.append(id)
        classes_names.append(class_name)

    dct["bbox"] = boxes
    dct["id"] = labels
    dct["name"] = classes_names
    dct["iscrowd"] = 0

    return dct


def voc_to_coco(dataset_split_paths: list[Path]) -> None:
    path_to_dataset_dir: Path = dataset_split_paths[0].parent.parent.parent.resolve()

    dir_annotations_name: Path = Path("annotations")
    path_to_dir_annotations: Path = path_to_dataset_dir / dir_annotations_name
    path_to_dir_annotations.mkdir(parents=True, exist_ok=True)

    classes_and_id = {class_name: id + 1 for id, class_name in enumerate(CLASSES)}
    id_and_classes = [
        {"id": id, "name": class_name, "supercategory": "mob"}
        for class_name, id in classes_and_id.items()
    ]

    coco_base = {
        "info": {"description": "VOCO-to-COCO converted dataset"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": id_and_classes,
    }

    annotation_id = 1
    img_id = 1
    split_dataset_name: Path = dataset_split_paths[0].parent.name
    for path in dataset_split_paths:
        if path.suffix.lower() == ".xml":
            parsed_xml = _parse_xml(path)
            coco_base["images"].append(
                {
                    "id": img_id,
                    "file_name": f"{split_dataset_name}/{parsed_xml['file_name']}",
                    "width": parsed_xml["width"],
                    "height": parsed_xml["height"],
                }
            )

            for bbox, category_id in zip(parsed_xml["bbox"], parsed_xml["id"]):
                x, y, w, h = bbox
                coco_base["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": int(category_id),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "area": float(w * h),
                        "iscrowd": parsed_xml["iscrowd"],
                        "segmentation": [],
                    }
                )
                annotation_id += 1

            img_id += 1

    split_dataset_annotations_path: Path = (
        path_to_dir_annotations / f"{split_dataset_name}.json"
    )

    with split_dataset_annotations_path.open("w", encoding="utf-8") as f:
        json.dump(coco_base, f, ensure_ascii=False, indent=2)

    print(
        f"{str(split_dataset_name).upper()} DATASET: The conversion has been completed"
    )
    print(
        f"{str(split_dataset_name).upper()} ANNOTAION WAS SAVED: {str(split_dataset_annotations_path)}"
    )
