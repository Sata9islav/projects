from pathlib import Path


def check_dataset(path_to_datasets_dir: str) -> list[Path]:
    path_to_datasets_dir = Path(path_to_datasets_dir)

    if not path_to_datasets_dir.exists():
        print(f"Directory {path_to_datasets_dir} does not exist")
        return []

    base_paths = [path for path in path_to_datasets_dir.iterdir() if path.is_file()]

    jpg_paths = {path for path in base_paths if path.suffix.lower() == ".jpg"}
    xml_paths = {path for path in base_paths if path.suffix.lower() == ".xml"}

    print()
    print(f"JPG NUMS: {len(jpg_paths)}")
    print(f"XML NUMS: {len(xml_paths)}")
    print()

    if len(jpg_paths) != len(xml_paths):
        print("There is not enough markup or image.")
        print("The following incomplete pairs will not be considered:\n")
    else:
        print(f"=== DATASET {str(path_to_datasets_dir.name).upper()} IS CORRECT ===")

    exists_files = []
    cnt = 1
    for file_path in sorted(jpg_paths | xml_paths):
        has_jpg = file_path.with_suffix(".jpg").exists()
        has_xml = file_path.with_suffix(".xml").exists()

        if not (has_jpg and has_xml):
            missing = []
            if not has_jpg:
                missing.append("jpg")
            if not has_xml:
                missing.append("xml")

            print(f"{cnt}. {file_path} -> missing: {', '.join(missing)}")
            cnt += 1
        else:
            exists_files.append(file_path)
    print("\nThe total number of files that will be used: ", len(exists_files))
    return exists_files
