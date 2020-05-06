Download the zip file given and extract test1 and train.

Create two empty folders called train_cats and train_dogs. Then run rename.py to populate these folders with the corresponding
cat and dog images.

Create a folder called train_renamed and copy all the images in train_cats and train_dogs to it.

Run create_sheet.py to create an excel sheet containing the training labels for all the images.

Finally, run main.py using python3 to train the model.
Note: Reduce the batch size if CUDA runs out of memory.

You will also require the cnn_finetune library to run the scripts here, along with torch and torchvision.
This can be downloaded at https://github.com/creafz/pytorch-cnn-finetune.
