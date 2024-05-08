import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import os

def load_and_prep_image(filename, img_shape=512):
  """
  Reads an image from filename, turns it into a tensor and reshapes it
  to (img_shape, img_shape, colour_channels).
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode the read file into a tensor
  img = tf.image.decode_image(img)
  # Resize the image
  img = tf.image.resize(img, size=[img_shape, img_shape])
  # Rescale the image (get all values between 0 and 1)
  img = img/255.
  return img


class_names=['Alovera','Kuppai_meni','Manathakkali','Mudakathan','Neem','Nochi','Omavalli','Semparuthi','Thoothuvalai','Thulasi']


def pred_and_plot(model, filename, class_names=class_names):
  img = load_and_prep_image(filename)

  
  pred = model.predict(tf.expand_dims(img, axis=0))
  if pred.max()<0.7:
    pred_class="Unknown"
 
  else:
    if len(pred[0]) > 1:
      pred_class = class_names[tf.argmax(pred[0])]
    else:
      pred_class = class_names[int(tf.round(pred[0]))]

    if pred_class=='Alovera':
      str1="Aloe vera is used for skin care, soothing sunburn and acne, and for digestive health, easing symptoms like IBS and acid reflux with its juice."
      str2="Aloe vera"
    elif pred_class=='Kuppai_meni':
      str1="kuppaimeni, is utilized in traditional medicine for wound healing, anti-inflammatory effects, and respiratory health."
      str2="Acalypha indica"
    elif pred_class=='Manathakkali':
      str1="Manathakkali, or Solanum nigrum, is used in traditional medicine for its potential anti-inflammatory, antioxidant, and hepatoprotective properties."
      str2="Solanum nigrum"
    elif pred_class=='Mudakathan':
      str1="Is traditionally used for its anti-inflammatory and analgesic properties in managing joint pain and inflammatory conditions."
      str2="Cardiospermum halicacabum"
    elif pred_class=='Neem':
      str1="Neem (Azadirachta indica) is utilized for its various medicinal properties, including antibacterial, antiviral, and antifungal effects, as well as for skincare and pest control."
      str2="Azadirachta indica"
    elif pred_class=='Nochi':
      str1="Nochi (Vitex negundo) is employed in traditional medicine for its potential anti-inflammatory, analgesic, and antipyretic properties."
      str2="Vitex negundo"
    elif pred_class=='Omavalli':
      str1="Omavalli (Ocimum tenuiflorum) is utilized in traditional medicine for its anti-inflammatory, antioxidant, and adaptogenic properties."
      str2="Ocimum tenuiflorum"
    elif pred_class=='Semparuthi':
      str1="Semparuthi (Hibiscus rosa-sinensis) is traditionally used for its potential in promoting hair growth and enhancing skin health."
      str2="Hibiscus rosa-sinensis."
    elif pred_class=='Thoothuvalai':
      str1="Thoothuvalai (Solanum trilobatum) is utilized in traditional medicine for its expectorant and bronchodilator properties in treating respiratory ailments."
      str2="Solanum trilobatum  "
    elif pred_class=='Thulasi':
      str1="The scientific name for Thulasi is Ocimum sanctum, commonly known as Holy Basil."
      str2="Ocimum sanctum"
    else:
      str1=" "
      str2=" "
    print(f"Scientific Name:{str2} Uses:{str1}")
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class} Prob:{pred.max()}")
  plt.axis(False)
  plt.show()

effnet_model=tf.keras.models.load_model(('effherbs.h5'),custom_objects={'KerasLayer':hub.KerasLayer})
path = 'test'
files = []

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
   for file in f:
     files.append(os.path.join(r, file))
for f in files:
    pred_and_plot(model=effnet_model,filename=f)