import os
import shutil
import sys
import imageio
from imgaug import augmenters as iaa


def augment(non_augmented_folder: str, augmented_folder: str):
    """ augments the images in non_augmented_folder

    will generate 3 training images (fog, snow, rain) in the augmented_folder for each image in non_augmented_folder

    :param non_augmented_folder: folder containing non augmented training images
    :param augmented_folder: output folder, will contain the augmented images
    :return: nothing
    """

    # always restart from scratch
    if os.path.exists(augmented_folder):
        shutil.rmtree(augmented_folder)
    os.mkdir(augmented_folder)

    fog = iaa.Fog()
    snow = iaa.Snowflakes(flake_size=(0.8, 1.0), speed=(0.01, 0.05))
    rain = iaa.Rain(speed=(0.3, 0.8), drop_size=(0.4, 0.8))

    i = 0
    for dirpath, _, files in os.walk(non_augmented_folder):
        for filename in sorted(files):
            if filename.lower()[-4:] == ".jpg":
                fname = os.path.join(dirpath, filename)
                annotation_file = os.path.join(dirpath, f"{filename.lower()[:-4]}.txt")
                print(f"read train file {fname}")
                image = imageio.v2.imread(fname)
                print("read image")
                i += 1
                print("augment with fog")
                augment_image(augmented_folder, i, fog(image=image), annotation_file)
                i += 1
                print("augment with snow")
                augment_image(augmented_folder, i, snow(image=image), annotation_file)
                i += 1
                print("augment with rain")
                augment_image(augmented_folder, i, rain(image=image), annotation_file)


def augment_image(dirpath: str, i: int, image, non_augmented_annotation_file: str):
    image_file = generate_image_name(dirpath, i)
    annotation_file = generate_annotation_name(dirpath, i)
    imageio.imwrite(image_file, image)
    shutil.copy(non_augmented_annotation_file, annotation_file)


def generate_image_name(dirpath: str, i: int) -> str:
    return os.path.join(dirpath, f"image{str(i).zfill(4)}.jpg")


def generate_annotation_name(dirpath: str, i: int) -> str:
    return os.path.join(dirpath, f"image{str(i).zfill(4)}.txt")


if __name__ == '__main__':
    if (len(sys.argv) < 3):
        raise Exception("please specify non augmented folder and augmented folder")

    augment(sys.argv[1], sys.argv[2])
    print("done")
