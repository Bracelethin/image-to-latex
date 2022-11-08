from pathlib import Path

data_path=Path(__file__).resolve().parents[1]/"data"
train_path=data_path+"/train/"
test_path=data_path+"/test/"
dev_path=data_path+"/dev/"
output_path=data_path
def convert():
    index=0
    def writeLst(lstPath, imgDir):
        nonlocal index
        nonlocal f0
        with open(lstPath, 'w') as f:
            for label_filename in (Path(imgDir)/"labels").glob("*.txt"):
                img_filename=label_filename.stem+".png"
                f.write(img_filename + ' ' + str(index) + '\n')
                with open(label_filename, 'r', encoding='utf-8') as f2:
                    line = f2.read().replace('\n',"")
#                    incre_number=linstripe.count('\n')
                    f0.write(line + '\n')
                    index += 1
    def writeTestLst(lstPath, imgDir):
        nonlocal index
        with open(lstPath, 'w') as f:
            for img_filename in (Path(imgDir)/"images").glob("*.png"):
                f.write(img_filename.name + ' ' + str(img_filename.stem) + '\n')
    with open(output_path+"im2latex_formulas.norm.lst","w") as f0:
        writeLst(output_path+"im2latex_train_filter.lst", train_path)
        writeLst(output_path+"im2latex_validate_filter.lst", dev_path)
        writeTestLst(output_path+"im2latex_test_filter.lst", test_path)
        

if (__name__=="__main__"):
    convert()