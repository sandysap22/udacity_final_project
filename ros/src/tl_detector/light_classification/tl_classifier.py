import os
print("cwd -------->",os.getcwd())
from styx_msgs.msg import TrafficLight
#sys.path.append("..")
from object_detection import traffic_light_carnd
from PIL import Image




class TLClassifier(object):
    def __init__(self):        
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        sign = traffic_light_carnd.traffic_light_detector(image)
        
        print(sign)
        if sign == "unkown" :
            #return "unkown"
            return TrafficLight.UNKNOWN
        if sign == "red" :
            #return "red"
            return TrafficLight.RED
        if sign == "green" :
            #return "unkown"
	    return TrafficLight.UNKNOWN
        return TrafficLight.UNKNOWN


if __name__ == "__main__" :
  
  PATH_TO_TEST_IMAGES_DIR = 'object_detection/test_images'
  TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 11) ]
   
  classifier = TLClassifier()

  for image_path in TEST_IMAGE_PATHS:  

    image = Image.open(image_path)
    signal = classifier.get_classification(image)
    print(image_path + " : " + str(signal))
