#!/usr/bin/env python
""" This script sets up the trained data model which achieved the best submission
for us on Kaggle.
It requires python 3.5, internet connection and the `requests` module for
getting the data, as it is 1.6 Gb

If, for some reason, this doesn't work, then please download the trained model
from this link: goo.gl/x526hG
We recommend you download each file by itself, otherwise Google takes forever to
prepare a ZIP archive.
Then put the files under the `..data/1482182487/checkpoints` folder

"""

# From here: http://stackoverflow.com/a/39225039/786559

import requests, sys, os
from os import path


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def main():
    # check correct directory:
    if path.basename(os.getcwd()) != "scripts":
        print("You appear to be in the wrong directory. Please run this file from 'scripts'. Nothing done.")
        exit(-1)

    # create required directory
    base_path = '../data/1482182487/checkpoints'
    os.makedirs(base_path, exist_ok=True)

    # download required files
    base_name = 'model-9900'
    files = {'0B089tpx89mdXblZBOUpnTFZKME0':'data-00000-of-00001', # google_id:name
             '0B089tpx89mdXMEl1MUF3Z041MXc':'index',
             '0B089tpx89mdXMDVMM1BqX3JmcE0':'meta' }

    for fid, name in files.items():
        full_name = '{}/{}.{}'.format(base_path, base_name, name)
        if path.isfile(full_name):
            print('%s already exists... skipping' % full_name)
        else:
            print('Downloading %s ...' % full_name)
            download_file_from_google_drive(fid, full_name)

    # set up the checkpoint file:
    checkpoint_contents =  '''
        model_checkpoint_path: "./model-9900"
        all_model_checkpoint_paths: "./model-9900"
    '''
    with open('%s/checkpoint' % base_path, 'w') as f:
        f.write(checkpoint_contents)


if __name__ == "__main__":
    main()
