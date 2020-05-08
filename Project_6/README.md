Download the zip file given and extract test1 and train.

Create a folder called train_renamed.
Run rename.py to populate this folder with the corresponding cat and dog images.

Run create_sheet.py to create an excel sheet containing the training labels for all the images.

Finally, run main.py using python3 to train the model.
Note: Reduce the batch size if CUDA runs out of memory.

You will also require the cnn_finetune library to run the scripts here, along with torch and torchvision.
This can be downloaded at https://github.com/creafz/pytorch-cnn-finetune.
