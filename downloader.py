# -*- coding: utf-8 -*-

# !/usr/bin/python

# Note: requires the tqdm package (pip install tqdm)

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

# Thanks to @maxwell: https://www.kaggle.com/maxwell110/python3-version-image-downloader

import sys, os, multiprocessing, csv
from urllib import request, error
from PIL import Image
from io import BytesIO
import tqdm as tqdm
import re

def parse_data(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    return key_url_list[1:]  # Chop off header


def download_image(key_url):
    out_dir = sys.argv[2]
    (key, url) = key_url
    filename = os.path.join(out_dir, '{}.jpg'.format(key))

    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return 1

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return 1

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return 1
    
    return 0


def loader():
    if len(sys.argv) != 3:
        print('Syntax: {} <data_file.csv> <output_dir/>'.format(sys.argv[0]))
        sys.exit(0)
    (data_file, out_dir) = sys.argv[1:]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = parse_data(data_file)
    pool = multiprocessing.Pool(processes=64)  # Num of CPUs
    failures = sum(tqdm.tqdm(pool.imap_unordered(download_image, key_url_list), total=len(key_url_list)))
    print('Total number of download failures:', failures)
    pool.close()
    pool.terminate()


# arg1 : data_file.csv
# arg2 : output_dir

def download_sgf(url):
    try:
        print('Donwloading ', url[0])
        filename = url[0].split("/")[-1]
        response = request.urlopen(url[0])
        with open('kgs_data2/' + filename, 'bw') as f:
            f.write(response.read())
    except:
        print('Warning: Could not download file {} from {}'.format(url[0]))        

def download_from_url(url):
    pool = multiprocessing.Pool(processes=64)
    try:
        print('DOWNLOADING MAIN URL', url)
        response = request.urlopen(url)
    except:
        print('WARNING: COULD NOT DOWNLOAD FROM MAIN URL ', url)
        return 0
    _data = response.read()
    m = re.findall(r"((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)", str(_data))
    print('Found {} games'.format(len(m)))
    for item in m:
        pool.apply_async(download_sgf, args=(item,))
    pool.close()
    pool.join()
    pool.terminate()
    return len(m) #  total game

if __name__ == '__main__':
    month = 1
    total_game = 0
    games = []
    for year in [2010, 2011]:
        for month in range(12):
            url = "https://orb.at/top100/{}{num:02d}.html".format(year, num=month+1)
            total_game += download_from_url(url)
    print("TOTAL GAME:", total_game)

