#!/usr/bin/env python
'''Take a photo.

Take a photo using a USB or Raspberry Pi camera.
'''

import os
from time import time, sleep
import json
import requests
import numpy as np
import cv2

try:
    from farmware_tools.env import Env
except ImportError:
    IMAGES_DIR = os.getenv('IMAGES_DIR')
else:
    IMAGES_DIR = Env().images_dir

def _farmware_api_url():
    major_version = int(os.getenv('FARMBOT_OS_VERSION', '0.0.0')[0])
    base_url = os.environ['FARMWARE_URL']
    return base_url + 'api/v1/' if major_version > 5 else base_url

def legacy_log(message, message_type):
    'Send a message to the log.'
    try:
        os.environ['FARMWARE_URL']
    except KeyError:
        print(message)
    else:
        log_message = '[take-photo] ' + str(message)
        headers = {
            'Authorization': 'bearer {}'.format(os.environ['FARMWARE_TOKEN']),
            'content-type': 'application/json'}
        payload = json.dumps(
            {'kind': 'send_message',
             'args': {'message': log_message, 'message_type': message_type}})
        requests.post(_farmware_api_url() + 'celery_script',
                      data=payload, headers=headers)

try:
    from farmware_tools import device
except ImportError:
    log = legacy_log
else:
    def log(message, message_type):
        'Send a log message.'
        device.log('[take-photo] {}'.format(message), message_type)


def undistort(img):
    log('Undistorting image.', 'debug')
    
    mtx = np.load('Take-Photo-master/calibration/mtx.npy')
    dist = np.load('Take-Photo-master/calibration/dist.npy')
    roi = np.load('Take-Photo-master/calibration/roi.npy')
    newcameramtx = np.load('Take-Photo-master/calibration/newcameramtx.npy')
    
    # h,  w = img.shape[:2]

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]

    return dst
def adjust_gamma(image, gamma=1.0):
    log('Adjusting gamma.', 'debug')

	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    invGamma = 1.0 / gamma
    
    table = np.array([ ((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
    
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def crop(image):
    log('Cropping center to 640px.', 'debug')
    cropSize = 640

    height, width, _ = image.shape
    
    if cropSize > height and cropSize > width:
        offsetH = (height - cropSize) / 2
        offsetW = (width - cropSize) / 2

        cropped_image = image[offsetH:offsetH+cropSize, offsetW:offsetW+cropSize]
        return cropped_image
    else:
        return image

def rotate(image):
    log('Rotate image if calibration data exists.', 'debug')

    angle = float(os.environ['CAMERA_CALIBRATION_total_rotation_angle'])
    sign = -1 if angle < 0 else 1
    turns, remainder = -int(angle / 90.), abs(angle) % 90  # 165 --> -1, 75
    if remainder > 45: turns -= 1 * sign  # 75 --> -1 more turn (-2 turns total)
    angle += 90 * turns                   #        -15 degrees
    image = np.rot90(image, k=turns)
    height, width, _ = image.shape
    matrix = cv2.getRotationMatrix2D((int(width / 2), int(height / 2)), angle, 1)

    return cv2.warpAffine(image, matrix, (width, height))

def image_filename():
    'Prepare filename with timestamp.'
    epoch = int(time())
    filename = '{timestamp}.jpg'.format(timestamp=epoch)
    return filename

def upload_path(filename):
    'Filename with path for uploading an image.'
    images_dir = IMAGES_DIR or '/tmp/images'
    if not os.path.isdir(images_dir):
        log('{} directory does not exist.'.format(images_dir), 'error')
    path = images_dir + os.sep + filename
    return path

def save_image(image):
    'Save an image to file after attempting rotation.'
    filename = image_filename()
    # Try to process image
    try:

        # Undistort image
        h, w, _ = image.shape
        log('Image w: {} h:{}'.format(w, h), 'debug')
        final_image = undistort(image)

        # Rotate image
        h, w, _ = final_image.shape
        log('Image w: {} h:{}'.format(w, h), 'debug')
        final_image = rotate(final_image)
        
        # Adjust gamma
        h, w, _ = final_image.shape
        log('Image w: {} h:{}'.format(w, h), 'debug')
        final_image = adjust_gamma(final_image, gamma=0.55) 
        
        # Crop image
        h, w, _ = final_image.shape
        log('Image w: {} h:{}'.format(w, h), 'debug')
        final_image = crop(final_image) 

    except:
        final_image = image
    else:
        filename = 'rotated_' + filename
    # Save the image to file
    filename_path = upload_path(filename)
    cv2.imwrite(filename_path, final_image)
    print('Image saved: {}'.format(filename_path))

def usb_camera_photo():
    'Take a photo using a USB camera.'
    # Settings
    camera_port = 0      # default USB camera port
    discard_frames = 25  # number of frames to discard for auto-adjust

    # Check for camera
    if not os.path.exists('/dev/video' + str(camera_port)):
        print('No camera detected at video{}.'.format(camera_port))
        camera_port += 1
        print('Trying video{}...'.format(camera_port))
        if not os.path.exists('/dev/video' + str(camera_port)):
            print('No camera detected at video{}.'.format(camera_port))
            log('USB Camera not detected.', 'error')

    # Open the camera
    camera = cv2.VideoCapture(camera_port)

    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)


    sleep(0.1)

    # Let camera adjust
    for _ in range(discard_frames):
        camera.grab()

    # Take a photo
    ret, image = camera.read()

    # Close the camera
    camera.release()

    # Output
    if ret:  # an image has been returned by the camera
        save_image(image)
    else:  # no image has been returned by the camera
        log('Problem getting image.', 'error')

def rpi_camera_photo():
    'Take a photo using the Raspberry Pi Camera.'
    from subprocess import call
    try:
        tempfile = upload_path('temporary')
        retcode = call(
            ['raspistill', '-w', '640', '-h', '480', '-o', tempfile])
        if retcode == 0:
            image = cv2.imread(tempfile)
            os.remove(tempfile)
            save_image(image)
        else:
            log('Problem getting image.', 'error')
    except OSError:
        log('Raspberry Pi Camera not detected.', 'error')

if __name__ == '__main__':
    log("OpenCV version: %s" % cv2.__version__, 'debug')
    
    

    try:
        CAMERA = os.environ['camera']
    except (KeyError, ValueError):
        CAMERA = 'USB'  # default camera

    if 'RPI' in CAMERA:
        rpi_camera_photo()
    else:
        usb_camera_photo()
