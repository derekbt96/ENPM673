import numpy as np 
import cv2
from matplotlib import pyplot as plt
from functions import LK_tracker, get_frames


# CHANGE THE VARIABLE BELOW TO THE DESIRED OUTPUT PROBLEM
# PROBLEM 1 = Car
# PROBLEM 2 = Bolt
# PROBLEM 3 = Dragon Baby
problem = 1


cap = get_frames(problem)
tracker = LK_tracker(problem)

start_frame = cap.get_next_frame()
result = tracker.apply(start_frame)

# out = cv2.VideoWriter('tracker3.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (800,600))

while True:
	
	frame = cap.get_next_frame()

	if frame is None:
		break

	
	result = tracker.apply(frame)
	
	# result = cv2.resize(result, (800, 600), interpolation = cv2.INTER_AREA)
	cv2.imshow('result',result)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# out.release()
print('Done')
cv2.destroyAllWindows()


