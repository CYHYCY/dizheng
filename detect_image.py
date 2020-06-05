from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from models.mtcnn_model import P_Net, R_Net, O_Net
import cv2
import numpy as np


class face_detect(object):
    def __init__(self):
        self.load_model()

    def load_model(self):
        thresh = [0.6, 0.7, 0.7]
        min_face_size = 20
        stride = 2
        slide_window = False
        detectors = [None, None, None]
        prefix = ['./weight/PNet_landmark/PNet', './weight/RNet_landmark/RNet',
                  './weight/ONet_landmark/ONet']
        epoch = [18, 14, 16]
        model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
        PNet, RNet, ONet = FcnDetector(P_Net, model_path[0]), Detector(R_Net, 24, 1, model_path[1]), \
                           Detector(O_Net, 48, 1, model_path[2])
        detectors[0], detectors[1], detectors[2] = PNet, RNet, ONet
        self.mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                            stride=stride, threshold=thresh, slide_window=slide_window)

    def detect(self, img, show_pic=False):
        """
        :param img: BGR格式的图片
        :param show_pic: 是否展示图片
        :return:boxes_c type is ndarray, shape is (M, 5)
        """
        boxes_c, landmarks = self.mtcnn_detector.detect(img)
        if show_pic:
            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                score = boxes_c[i, 4]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255), 3)

            for i in range(landmarks.shape[0]):
                for j in range(int(len(landmarks[i]) / 2)):
                    cv2.circle(frame, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))
                cv2.putText(frame, 'handsome men', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (127, 255, 212), 2)
                cv2.imshow("", frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        return boxes_c


if __name__ == "__main__":
    model = face_detect()
    path = "./suda.jpg"
    frame = cv2.imread(path)
    test_data = np.array(frame)
    boxes_c = model.detect(test_data)
