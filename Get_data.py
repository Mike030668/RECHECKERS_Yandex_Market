import PIL
from PIL import ImageDraw, Image
import cv2
import numpy as np

import requests
from io import BytesIO

def urls_to_images(urls, resize2 = ()):
    imgs = []
    for i in urls:
        try:
            url = f'https:{i}'
            response = requests.get(url)
            img = PIL.Image.open(BytesIO(response.content)).convert('RGBA')
            if resize2:
              img = img.resize(resize2, Image.Resampling.LANCZOS)
            imgs.append((img))

        except Exception as ex:
            print(f'not load {i}')
            print(ex)
    return imgs

def get_data(urls, resize2 = (),  labels=[], good_links=()):

    if len(labels):
      imgs = []
      labels_ = []
      for i, l in zip(urls, labels):
          try:
              url = f'https:{i}'
              response = requests.get(url)
              img = PIL.Image.open(BytesIO(response.content)).convert('RGBA')


          except Exception as ex:
              if len(good_links):
                 l = np.random.choice(list(good_links.keys()))
                 url = np.random.choice(good_links[l])
                 response = requests.get(url)
                 img = PIL.Image.open(BytesIO(response.content)).convert('RGBA')

                 print(f'not load {i} take random GOOD_URL')
              else:
                 print(f'not load {i}')
                 print(ex)

          if resize2:
              img = img.resize(resize2, Image.Resampling.LANCZOS)

          imgs.append((img))
          labels_.append((l))

      return imgs, labels_
    else:
      return urls_to_images(urls, resize2), []


# Go through Each Unique Link to Identify Broken Ones
def identifyBrokenLinks(uniqueExternalLinks):
    count = 0
    length_uniqueExternalLinks = len(uniqueExternalLinks)
    user_agent = {'User-Agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36'}
    brokenLinksList = []

    for link in uniqueExternalLinks:
        print("Checking external link #", count," out of ",length_uniqueExternalLinks,".")

        l = f'https:{link}'
        try:
            statusCode = requests.get(l, headers=user_agent).status_code

            if statusCode == 404:
                brokenLinksList.append(link)
            else:
                pass
        except:
            brokenLinksList.append(link)
        count = count + 1
    return brokenLinksList


# Go through Each Unique Link to Identify Broken Ones
def take_goodLinks(uniqueExternalLinks, bad_links,  qty = 10):
    count = 0
    goodLinksList = []

    for link in uniqueExternalLinks:
        if link not in bad_links:
          try:
            url = f'https:{link}'
            response = requests.get(url)
            _ = PIL.Image.open(BytesIO(response.content)).convert('RGBA')

            goodLinksList.append(url)
            count = count + 1
          except: pass
        if count == qty: break

    return goodLinksList