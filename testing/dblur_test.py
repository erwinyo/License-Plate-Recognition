from dblur.testers.mscnn import MSCNNTester
from dblur.testers.stack_dmphn import StackDMPHNTester
from dblur.default.restormer import train, test, deblur_imgs, deblur_single_img

mscnn_tester = MSCNNTester()
model1 = mscnn_tester.get_model()

dmphn_tester = StackDMPHNTester()
model2 = dmphn_tester.get_model(num_of_stacks=1)

train_img_dir = "path_to_training_dataset"
val_img_dir = "path_to_validation_dataset"
test_img_dir = "path_to_test_dataset"
model_path = "path_for_model"
#
# # Train model
# train(model_path, train_img_dir)
#
# # Test pretrained model
# train(model_path, test_img_dir)

# # Deblur images in a directory using pretrained model
# deblur_imgs(model_path, "blurred_imgs_path", "sharp_imgs_path")

# Deblur single image using pretrained model
deblur_single_img(model1, "../asset/image/blur#1.jpg", "../asset/image/deblur#1.jpg")
