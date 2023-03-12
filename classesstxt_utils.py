import os


def load_classess(directory):
    classes = []
    path_to_classes = os.path.join(directory, "classes.txt")
    if(os.path.exists(path_to_classes)):

        with open(os.path.join(directory, "classes.txt")) as classes_file:
            for line in classes_file.readlines():
                classes.append(line.strip())
        return classes
    return None


def save_classes_to_file(path, classes):
    with open(path+"/classes.txt", "w") as file_handle:
        for class_element in classes:
            file_handle.write(class_element)
            file_handle.write("\n")


def load_model_type_from_directory(directory):
    target_path = os.path.join(directory, "model_type.txt")
    try:
        with open(target_path) as model_type_file:
            for line in model_type_file:
                return line.strip()
    except Exception as e:
        print(f"Model file {target_path} not found - using resnet as default model_type")
        print(e)
        return "resnet"
    print("Empty file - using resnet as default model_type")
    return "resnet"


def save_model_type_to_file(directory, model_type):
    with open(directory+"/model_type.txt", "w") as file_handle:
        file_handle.write(model_type)
        file_handle.write("\n")
