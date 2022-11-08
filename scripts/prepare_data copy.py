import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import image_to_latex.data.utils as utils


PROJECT_DIRNAME = Path(__file__).resolve().parents[1]
DATA_DIRNAME = PROJECT_DIRNAME / "data"
RAW_IMAGES_DIRNAME = DATA_DIRNAME / "formula_images"
RAW_LABELS_DIRNAME = DATA_DIRNAME / "labels"
PROCESSED_IMAGES_DIRNAME = DATA_DIRNAME / "formula_images_processed"
PROCESSED_LABELS_DIRNAME =DATA_DIRNAME / "labels_processed"
VOCAB_FILE = PROJECT_DIRNAME / "image_to_latex" / "data" / "vocab.json"
all_formular_lst=DATA_DIRNAME/ "all_formular_lst"

def main():
#    DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    print(DATA_DIRNAME,PROCESSED_IMAGES_DIRNAME)
    cur_dir = os.getcwd()
    os.chdir(DATA_DIRNAME)

    # Download images and grouth truth files
#    for filename, url in METADATA.items():
#        if not Path(filename).is_file():
#            utils.download_url(url, filename)

    # Unzip
#    if not RAW_IMAGES_DIRNAME.exists():
#        RAW_IMAGES_DIRNAME.mkdir(parents=True, exist_ok=True)
#        utils.extract_tar_file("formula_images.tar.gz")

    # Extract regions of interest
    if not PROCESSED_IMAGES_DIRNAME.exists():
        PROCESSED_IMAGES_DIRNAME.mkdir(parents=True, exist_ok=True)
        print("Cropping images...")
        for image_filename in RAW_IMAGES_DIRNAME.glob("*.png"):
            cropped_image = utils.crop(image_filename, padding=8)
            if not cropped_image:
                continue
            cropped_image.save(PROCESSED_IMAGES_DIRNAME / image_filename.name)

   # Clean the ground truth file
    if not PROCESSED_LABELS_DIRNAME.exists():
        PROCESSED_LABELS_DIRNAME.mkdir(parents=True, exist_ok=True)
        print("replace labels...")
        script = Path(__file__).resolve().parent / "find_and_replace.sh "
        for label_filename in tqdm(RAW_LABELS_DIRNAME.glob("*.txt")):
#            subprocess.call(["sh", f"{str(script)}", label_filename, PROCESSED_LABELS_DIRNAME/label_filename.name])
            os.system("".join(["sh ", f"{str(script)}", label_filename.__str__()+' ', (PROCESSED_LABELS_DIRNAME/label_filename.name).__str__()]))

    # Build vocabulary
    if not VOCAB_FILE.is_file():
        print("Building vocabulary...")
        all_formulas = utils.get_all_formulas(all_formular_lst)
        _, train_formulas = utils.get_split(all_formulas, "im2latex_train_filter.lst")
        tokenizer = utils.Tokenizer()
        tokenizer.train(train_formulas)
        tokenizer.save(VOCAB_FILE)
    os.chdir(cur_dir)


if __name__ == "__main__":
    main()
