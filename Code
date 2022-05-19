from pynput.mouse import Controller,Button,Listener
from numpy import asarray
from mss import mss
import cv2
import time

# def on_click(x, y, button, pressed):
#     if pressed:
#         print('X:{0}, Y:{1}'.format(x, y))
#
# with Listener(on_click=on_click) as listener:
#     listener.join()

def bot_vision():
    screenshot_coordinates = (1292, 418, 2555, 1000)
    day_template = cv2.imread(
    "TEMPLATE FILE PATH")
    night_template = cv2.imread(
    "TEMPLATE FILE PATH")

    with mss() as sct:
        #SCREEN SIZE AND CONVERT TO NUMPY ARRAY
        screenshot = sct.grab(screenshot_coordinates)
        image = asarray(screenshot)

        #PUT IMAGE INTO GREYSCALE
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grey_day = cv2.cvtColor(day_template, cv2.COLOR_RGB2GRAY)
        grey_night = cv2.cvtColor(night_template, cv2.COLOR_RGB2GRAY)
        night_w, night_h = grey_night.shape[::-1]
        day_w, day_h = grey_day.shape[::-1]

        #COMPARE DAY SCREENSHOT TO TEMPLATE
        result_day = cv2.matchTemplate(
            image=image_gray,
            templ=grey_day,
            method=cv2.TM_CCOEFF_NORMED)

        caught_min_val, caught_max_val, caught_min_loc, caught_max_loc = cv2.minMaxLoc(result_day)
        print(str(caught_max_val) + " MATCH ATTEMPT")
        if caught_max_val >= 0.8:
            image = cv2.rectangle(
                img=image_gray,
                pt1=caught_max_loc,
                pt2=(caught_max_loc[0] + day_w,
                     caught_max_loc[1] + day_h),
                color=(0, 0, 255),
                thickness=2)
            mouse = Controller()
            mouse.position = (2429, 886)
            mouse.click(Button.left, 1)
            time.sleep(7)
            mouse.position = (2277, 876)
            mouse.click(Button.left, 1)
            time.sleep(4)
            mouse.position = (2359, 786)
            mouse.click(Button.left, 1)

        #COMPARE NIGHT SCREENSHOT TO TEMPLATE
        result_night = cv2.matchTemplate(
            image=image_gray,
            templ=grey_night,
            method=cv2.TM_CCOEFF_NORMED)

        caught_min_val, caught_max_val, caught_min_loc, caught_max_loc = cv2.minMaxLoc(result_night)
        print(str(caught_max_val) + " MATCH ATTEMPT")
        if caught_max_val >= 0.8:
            image = cv2.rectangle(
                img=image_gray,
                pt1=caught_max_loc,
                pt2=(caught_max_loc[0] + night_w,
                     caught_max_loc[1] + night_h),
                color=(0, 0, 255),
                thickness=2)
            mouse = Controller()
            mouse.position = (2429, 886)
            mouse.click(Button.left, 1)
            time.sleep(7)
            mouse.position = (2277, 876)
            mouse.click(Button.left, 1)
            time.sleep(4)
            mouse.position = (2359, 786)
            mouse.click(Button.left, 1)

        #CREATE MINI DISPLAY FOR DEBUGGING
        image_mini = cv2.resize(src=image_gray, dsize=(650, 450))
        cv2.imshow("Bot Vision", image_mini)
        cv2.waitKey(10)  # Refresh Rate, Lower More CPU USE

while True:
    bot_vision()
