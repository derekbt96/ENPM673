import numpy as np 
import cv2
from matplotlib import pyplot as plt
from functions import LK_tracker, get_frames


# CHANGE THE VARIABLE BELOW TO THE DESIRED OUTPUT PROBLEM
# PROBLEM 1 = Car
# PROBLEM 2 = Bolt
# PROBLEM 3 = Dragon Baby
problem = 2


cap = get_frames(problem)

tracker = LK_tracker()

# out = cv2.VideoWriter('tracker.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (800,600))
while True:
	
	frame = cap.get_next_frame()
	

	if frame is None:
		break

	result = tracker.apply(frame)

	# result = cv2.resize(result, (800, 600), interpolation = cv2.INTER_AREA)
	# out.write(result2)
	cv2.imshow('result',result)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break

# out.release()
cv2.destroyAllWindows()



# cd Documents\Documents\Aerospace\ENPM673\ENPM673\Project_4\Code