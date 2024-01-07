# To Capture Frame
import cv2
import numpy as np 
# import the tensorflow modules and load the mode
import tensorflow as tf 
model=tf.keras.models.load.model("keras_model.h5")

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()
	image=cv2.resize(status,(224,224))

	test_image=np.array(image,dtype=np.float32)
	test_image=np.expand_dims(test_image,axis=0)

	Normalize_img=test_image/255.0

	prediction=model.predict(Normalize_img)
	print(f"Prediction,{prediction}")

	cv2.imshow('frame',frame)

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)

		

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
