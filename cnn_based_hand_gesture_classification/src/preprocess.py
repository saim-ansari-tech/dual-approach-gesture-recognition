input_img = cv2.resize(hand_crop, (128, 128))
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) 
input_img = input_img.astype('float32') / 255.0
input_img = np.expand_dims(input_img, axis=0)
