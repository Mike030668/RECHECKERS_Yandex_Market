import tensorflow as tf
import numpy as np
from numpy.core.fromnumeric import resize
from scipy.ndimage import rotate
import cv2

######################
# augment image #
######################

def agumager(img, simple_agum = True):
    """
    Функция создания аугментированных наборов
    """
    shape = img.shape
    aug_imgs = [img]
    aug_imgs.extend(list(map(lambda x: np.flip(x,axis=1), aug_imgs)))
    #if not simple_agum:
    aug_imgs.extend(list(map(lambda x: np.flip(x,axis=0), aug_imgs)))
    aug_imgs.extend(list(map(lambda x: np.rot90(x), aug_imgs)))
    id = np.random.randint(len(aug_imgs))
    out_img = aug_imgs[id]

    # поворты малые
    if np.random.random()>0.5:
      rad = 10 if not simple_agum else 3
      ang = np.random.randint(1, rad)
      if np.random.random()>0.5: ang*=-1
      out_img = rotate(out_img, angle=ang)

    # обрезаем и выравниваем
    if np.random.random()<0.3: out_img = out_img[:shape[0],:shape[1],:]
    elif 0.6>np.random.random()>=0.3: out_img = out_img[-shape[0]:,-shape[1]:,:]
    else:
      out_img = cv2.resize(img, dsize=shape[:2], interpolation=cv2.INTER_CUBIC)

    return out_img


######################
# generatot to model #
######################

class Generator2Сlassificator(tf.keras.utils.Sequence):
    def __init__(self, urls, take_data, text_reader, text_preprocess,
                 mask_predictor, good_links, agument = None, labels =[], 
                 recize2=(), batch_size = 16):

        self.urls = urls
        self.labels = labels
        self.recize2 = recize2
        self.batch_size = batch_size
        self.good_links = good_links
        self.num_batches = np.ceil(len(self.urls) / batch_size)
        self.batch_idx = np.array_split(range(len(self.urls)), self.num_batches)
        self.take_data = take_data
        self.text_reader = text_reader
        self.text_detect = text_preprocess
        self.mask_predictor = mask_predictor
        self.agument = agument


    def __len__(self):
        return len(self.batch_idx)

    def __getitem__(self, idx):
        batch_urls = self.urls[self.batch_idx[idx]]
        batch_labels = self.labels[self.batch_idx[idx]]
        images, labels = self.take_data(urls = batch_urls,
                                        labels = batch_labels,
                                        resize2 = self.recize2,
                                        good_links = self.good_links
                                        )
        batch_img = []
        batch_logits = []
        batch_intsept = []
        batch_y = []
        for img, label  in zip(images, labels):
            img_bgr = np.array(img)[:, :, 2::-1]
            if self.agument:
              img_bgr = self.agument(img_bgr)
            
            # detection
            self.mask_predictor.set_image(img_bgr)
            masks, scores, logits = self.mask_predictor.predict(
                multimask_output=True
            )
            logits = np.moveaxis(logits, [0], [-1])

            # texts
            text_box_xy = self.text_detect(img_bgr, self.text_reader)

            # intesections masks
            mask_text = text_box_xy !=0
            masks_intsept = [masks[i]*scores[i]*text_box_xy for i in range(masks.shape[0])]
            masks_intsept = np.stack(masks_intsept, axis=-1)

            # norm image
            img_norm = cv2.normalize(img_bgr, None,
                                 alpha=0,
                                 beta=1,
                                 norm_type=cv2.NORM_MINMAX,
                                 dtype=cv2.CV_32F)

            batch_img.append(img_norm)
            batch_intsept.append(masks_intsept)
            batch_logits.append(logits)
            batch_y.append(label)

        batch_img = np.stack(batch_img)
        batch_logits = np.stack(batch_logits)
        batch_intsept = np.stack(batch_intsept)
        batch_y = np.stack(batch_y)
        # собираем входные данные для регрессора
        return [batch_img, batch_intsept, batch_logits], batch_y

######################
# text preprocessing #
######################

def text_boxes(img, reader):
    img_BGR = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    bounds = reader.readtext(img_BGR)
    if len(bounds):
      boxes = []
      conf = []
      for bound in bounds:
          p0, p1, p2, p3 = bound[0]
          boxes.append([int(p0[0]), int(p1[1]), int(p2[0]), int(p3[1]), bound[2]])
      return  boxes
    else: return  []

def temp_texts(results, img_shape):
    temp_texts = np.zeros(img_shape[:2])
    qty = len(results)
    if qty:
      for i in range(qty):
        roi = np.zeros(img_shape[:2])
        x1, y1, x2, y2, conf = results[i]
        roi[y1:y2, x1:x2] = conf
        temp_texts+=roi
    return temp_texts

def text_detect(img, reader):
    text_info = text_boxes(img, reader)
    text_map = temp_texts(text_info, img.shape)
    return text_map
