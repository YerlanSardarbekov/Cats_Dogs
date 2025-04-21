# Cats_Dogs
Classification of Cats vs Dogs

# Provide the correct path to your model file
model_path = '/content/catvsdog.h5'
model = load_model(model_path)

test_image = image.load_img('/content/drive/MyDrive/archive (1)/test_set/test_set/dogs/dog.4037.jpg',target_size=(224,224))

#For show image
plt.imshow(test_image)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)

# Result array
result = model.predict(test_image)

#Mapping result array with the main name list
i=0
if(result>=0.5):
  print("Dog")
else:
  print("Cat")

result #will show the result whether cat or dog
