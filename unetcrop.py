from unet import crop as unet

MODEL = unet.unet_model.unet()
MODEL.load_weights("unet/unet_people.hdf5")

fname = "sample-img"
unet.crop_img(fname, MODEL)