import cv2
import matplotlib.pyplot as plt

lic_data = cv2.CascadeClassifier("haarcascades/haarcascade_russian_plate_number.xml")

def plt_show(img, title="", gray = False, size=(100,100)):
    temp = img
    if gray == False:
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        plt.title(title)
        plt.imshow(temp, cmap="gray")
        plt.show()


def detect_number(img):
    temp = img
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    number = lic_data.detectMultiScale(gray)

    print("number plate detected: "+ str(len(number)))
    for num in number:
        (x, y, w, h) = num
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.rectangle(temp, (x, y), (x+w, y+h), (255, 0, 0), 3)
    plt_show(temp)

def main():
    img = cv2.imread("data/test/car.jpg")
    plt_show(img)
    detect_number(img)
    print("Llegue")

if __name__ == "__main__":
    main()