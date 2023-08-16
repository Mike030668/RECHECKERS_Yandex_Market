import cv2
from PIL import ImageOps
import matplotlib.pyplot as plt

def show_predictions(type_mask, images, y_pred, y_true,
                     name_cls, color='green'):
    '''
    Показывае на картинках y_pred, y_true метки
    и окрашивает рамку 
    color='green' в зеленый
    color == 'red  в красный
    '''
    if color == 'red': R,G,B = [238,59,5]
    if color == 'green': R,G,B = [102,205,0]

    columns = int(len(type_mask)**0.5)
    if columns> 3: columns = 3
    if columns**2 < len(type_mask): rows = len(type_mask)//columns +1
    else: rows = columns

    # Plot the sample images now
    fig = plt.figure(figsize=(12, rows*4))
    ax = []
    for i in range(columns*rows):
      try:
        indx = type_mask[i]
        # making border around image using copyMakeBorder
        image =images[indx]
        image = np.array(image)
        r_, g_, b_ = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        rb = np.pad(array=r_, pad_width=20, mode='constant', constant_values=R)
        gb = np.pad(array=g_, pad_width=20, mode='constant', constant_values=G)
        bb = np.pad(array=b_, pad_width=20, mode='constant', constant_values=B)
        image_b = np.dstack(tup=(rb, gb, bb))

        ax.append(fig.add_subplot(rows, columns, i+1))
        ax[-1].set_title("True: {}\nPred: {}".format(name_cls[y_true[indx]],
                                                      name_cls[y_pred[indx]]) )
        plt.imshow(image_b)
        ax[-1].axis('off')
      except: pass

    plt.show() 

import PIL
from PIL import Image
import cv2
import numpy as np

import requests
from io import BytesIO

def data2predict(urls, labels=[], resize2 = ()):

      imgs = []
      labels_ = []
      links  = []
      for i, l in zip(urls, labels):
          try:
              url = f'https:{i}'
              response = requests.get(url)
              img = PIL.Image.open(BytesIO(response.content)).convert('RGBA')
              if resize2:
                 img = img.resize(resize2, Image.Resampling.LANCZOS)

              imgs.append((img))
              links.append((i))
              labels_.append((l))

          except Exception as ex:

                 print(f'not load {i}')
                 print(ex)

      return np.array(imgs), np.array(links), np.array(labels_)

def show_images(images, y_true, name_cls):
    '''
    Показывает на картинки таблицей
    '''
    rows = int(len(images)**0.5)
    if rows> 3: rows = 3
    if rows**2 < len(images): columns = len(images)//rows  +1
    else: columns = rows

    # Plot the sample images now
    fig = plt.figure(figsize=( columns*4, rows*4))
    ax = []
    for i in range(columns*rows):
      try:
        image = np.array(images[i])

        ax.append(fig.add_subplot(rows, columns, i+1))
        ax[-1].set_title("{}".format(name_cls[y_true[i]]) )
        plt.imshow(image)
        ax[-1].axis('off')
      except: pass

    plt.show()


def prompts2seg_classes(prompt,
                        addiins = [
                                   'infographic',
                                   'product',
                                   'packaging'
                                   ]):
  qtytakes_p = len(prompt)
  for p in range(qtytakes_p):
    if not p:
      classses = [t for t in prompt.split(',')[p].split(' ') if t not in ('', '.', '!')]
    else:
      try:
        classses+=[t for t in prompt.split(',')[p].split(' ') if t not in ('', '.', '!')]
      except: pass

  return  [t for t in list(set(classses)) if len(t) > 2] + addiins


from typing import List
def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]