import numpy as np
import matplotlib.pyplot as plt
import cv2


class image_helper:
    def __init__(self, img):
        self.resized_img = img
        self.original_img = img
        self.thresh_img = img
        self.display_img = img

    @staticmethod
    def show_img_cv2(img):
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def show_imgs_cv2(imgs):
        for i in range(len(imgs)):
            cv2.imshow('image ' + str(i), imgs[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def show_img_plt(img):
        plt.imshow(img, cmap='gray')
        plt.show()

    def show_imgs_plt(self, imgs):
        ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=3, colspan=1)
        ax1.set_size_inches(18.5, 10.5)
        ax1.imshow(imgs)
        plt.show()

    def play_with_hsv_and_blur(self):
        def scale_image(slider_val):
            if slider_val > 0:
                self.resized_img = cv2.resize(self.original_img, (0, 0), fx=slider_val / 50,
                                              fy=slider_val / 50)
            else:
                self.resized_img = self.original_img

            cv2.imshow('result', self.resized_img)

        def process_median_blur(slider_val):
            b_val = slider_val if slider_val % 2 != 0 else slider_val + 1
            self.display_img = cv2.medianBlur(self.thresh_img, b_val)
            cv2.imshow('result', self.display_img)

        def process_hsv(slider_val):
            hsv = cv2.cvtColor(self.resized_img, cv2.COLOR_BGR2HSV)

            # считываем значения бегунков
            h1 = cv2.getTrackbarPos('h_min', 'settings')
            s1 = cv2.getTrackbarPos('s_min', 'settings')
            v1 = cv2.getTrackbarPos('v_min', 'settings')
            h2 = cv2.getTrackbarPos('h_max', 'settings')
            s2 = cv2.getTrackbarPos('s_max', 'settings')
            v2 = cv2.getTrackbarPos('v_max', 'settings')

            # формируем начальный и конечный цвет фильтра
            h_min = np.array((h1, s1, v1), np.uint8)
            h_max = np.array((h2, s2, v2), np.uint8)

            # накладываем фильтр на кадр в модели HSV
            self.thresh_img = cv2.inRange(hsv, h_min, h_max)

            cv2.imshow('result', self.thresh_img)

        def process_thresh(slider_val):
            _, thresh1 = cv2.threshold(self.thresh_img, slider_val, 255, cv2.THRESH_BINARY)
            # _, thresh2 = cv2.threshold(self.resized_img, b_val, 255, cv2.THRESH_BINARY_INV)
            # _, thresh3 = cv2.threshold(self.resized_img, b_val, 255, cv2.THRESH_TRUNC)
            # _, thresh4 = cv2.threshold(self.resized_img, b_val, 255, cv2.THRESH_TOZERO)
            # _, thresh5 = cv2.threshold(self.resized_img, b_val, 255, cv2.THRESH_TOZERO_INV)
            self.display_img = thresh1
            cv2.imshow('result', self.display_img)

        def process_gauss_blur(slider_val):
            b_val = slider_val if slider_val % 2 != 0 else slider_val + 1
            gauss = cv2.GaussianBlur(self.thresh_img, (b_val,b_val),0,0)
            self.display_img = gauss
            cv2.imshow('result', self.display_img)

        cv2.namedWindow("result")  # создаем главное окно
        cv2.namedWindow("settings", flags=cv2.WINDOW_AUTOSIZE)  # создаем окно настроек

        # создаем 6 бегунков для настройки начального и конечного цвета фильтра
        cv2.createTrackbar('h_min', 'settings', 0, 255, process_hsv)
        cv2.createTrackbar('s_min', 'settings', 0, 255, process_hsv)
        cv2.createTrackbar('v_min', 'settings', 0, 255, process_hsv)
        cv2.createTrackbar('h_max', 'settings', 255, 255, process_hsv)
        cv2.createTrackbar('s_max', 'settings', 255, 255, process_hsv)
        cv2.createTrackbar('v_max', 'settings', 255, 255, process_hsv)

        cv2.createTrackbar('img_scale', 'settings', 1, 100, scale_image)

        cv2.createTrackbar('thresh', 'settings', 1, 255, process_thresh)
        cv2.createTrackbar('med_blur', 'settings', 1, 100, process_median_blur)
        cv2.createTrackbar('blur', 'settings', 1, 100, nothing)
        cv2.createTrackbar('gauss_blur', 'settings', 1, 100, process_gauss_blur)
        cv2.createTrackbar('bilateral', 'settings', 1, 100, nothing)
        cv2.createTrackbar('nlMeans', 'settings', 1, 100, nothing)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def play_with_blur(img):
        def process_blur(slider_position):
            b_val = slider_position if slider_position % 2 != 0 else slider_position + 1
            blur = cv2.medianBlur(img, b_val)
            cv2.imshow('result', blur)

        cv2.namedWindow("result")  # создаем главное окно

        cv2.createTrackbar('blur', 'blur_settings', 1, 500, process_blur)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def nothing(*arg):
    pass
