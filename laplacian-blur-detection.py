
import cv2
import equi_to_cube as etc
import matplotlib.pyplot as plt
import sys

def main():

    cap = cv2.VideoCapture(sys.argv[1])

    var_list = []
    frame_number = 1
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        print('Frame ' + str(frame_number) + ' Variance ' + str(lap_var) )

        var_list.append(lap_var)

        # Umwndlung zu
        # if lap_var > 90:
        #     converted = etc.equi_to_cube(frame)
        #     edge = 1440 # Breite und Höhe der sechs Bildteile
        #
        #     #cv2.imwrite('./cube_images/' + str(frame_number) + '.png', converted)
        #     cv2.imwrite('./cube_images/' + str(frame_number) + 'top' + '.png', converted[0:edge, 2*edge:3*edge])
        #     cv2.imwrite('./cube_images/' + str(frame_number) + 'bottom' + '.png', converted[2*edge:3*edge, 2*edge:3*edge])
        #     cv2.imwrite('./cube_images/' + str(frame_number) + 'back' + '.png', converted[1*edge:2*edge, 0*edge:1*edge])
        #     cv2.imwrite('./cube_images/' + str(frame_number) + 'left' + '.png', converted[1*edge:2*edge, 1*edge:2*edge])
        #     cv2.imwrite('./cube_images/' + str(frame_number) + 'front' + '.png', converted[1*edge:2*edge, 2*edge:3*edge])
        #     cv2.imwrite('./cube_images/' + str(frame_number) + 'right' + '.png', converted[1*edge:2*edge, 3*edge:4*edge])

        frame_number += 1



    plt.plot(var_list)
    plt.ylabel('variance')
    plt.show()

    # When everything done, release the capture
    cap.release()

if __name__ == "__main__":
    # execute only if run as a script
    main()
