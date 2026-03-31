## Laboratory Work 3 Activity — Building a Custom Image Classifier with
## TensorFlow Using Personal Image Datasets from Google Drive


## Google Colab Link: https://drive.google.com/drive/folders/1swojfcwk4zl9QWeIkarRu1_tRysYZQKs?usp=sharing



### Part 2: Guide Questions — Dataset Preparation, Training & Application
### Dataset Preparation

#### How did you organize your dataset in Google Drive
##### Answer: I created the main folder called ImageDataset in My Drive to organize the dataset. The folder contained different subfolders which I made for each plant species category — I created one folder for Anahaw and another folder for Narra and Molave and I continued this pattern until I reached 20 categories. Each folder contains at least 250 images of that specific plant. I used the folder names as class labels because I needed to keep the names clean and consistent without spaces and typos since TensorFlow directly reads them as class identifiers.

#### Why is folder structure important for TensorFlow image loading?
##### Answer: The function image_dataset_from_directory() from TensorFlow reads class labels directly from the folder names. If the structure is wrong — like mixing images across folders or having inconsistent naming — the model would neither load successfully nor display the correct image labels. The entire training pipeline depends on having a clean folder structure as its essential foundation. The system eliminates manual label mapping by its automatic process which also improves loading operations with greater accuracy and efficiency.

### Model Training

#### What is the role of convolutional layers in image classification?
##### Answer: The model uses convolutional layers to discover image patterns because these layers enable the model to analyze visual data. The model of early layers detects basic visual elements which include edges and corners and color gradients. The system combines features into advanced patterns which show themselves as leaf shapes and bark textures and floral petal arrangements. The system needs convolutional layers because their absence would result in a standard dense network which treats each individual pixel as a separate value and this approach fails to process image data because it disregards spatial connections.

#### Why do we split data into training and validation sets?
##### Answer: The model achieves learning through the training set while the validation set serves to verify whether the model has learned or just memorized its content. The model assessment requires both training and testing data because training data alone would create an environment where we cannot determine whether the model has learned actual plant features or whether it has overfitted by memorizing the answers. The validation set functions as a surprise quiz because the model has not encountered those images in its training process which enables us to evaluate its performance on that set as a more precise method for predicting its real-world data handling capabilities.

### Performance Analysis

#### What accuracy did your model achieve?
##### Answer: The initial CNN model training which lasted 10 epochs achieved validation accuracy between 72 to 78 percent. The model began to overfit when its training accuracy became higher because the model performed better on known images than on fresh images. The 20-class plant classification task with its limited dataset size needed this specific entry point as the base for all further work.

#### How did the number of images affect the model's performance?
##### Answer: The number of images per class had a significant effect. Classes that had closer to 250 images showed more consistent accuracy compared to any class that fell short. Deep learning models, especially CNNs, require large data sets because their performance improves when they access diverse plant species examples. The model requires multiple images to establish general plant characteristics because it needs more than one picture to learn about plant species.


### Critical Thinking

#### What challenges did you encounter while using your own dataset?
##### Answer: The project faced two major problems because of its requirements which needed high-quality visual results throughout the project. The dataset contained photos that had been captured under various lighting situations and camera positions and different background environments, which resulted in introducing errors into the dataset. The dataset included close duplicates of images which existed as minor cropped copies of the identical photograph; these duplicates would result in higher training precision while providing no educational value to the machine learning system. The distribution of images across 20 categories created difficulties because some species provided more image samples than others which resulted in unequal distribution that would lead to biased model predictions.

#### How can data augmentation improve your model?
##### Answer: Data augmentation artificially expands the dataset by applying random transformations to existing images — flipping, rotating, zooming, adjusting brightness, and so on. This forces the model to learn more robust, generalized features rather than relying on specific orientations or backgrounds. For example, if the model only ever sees upright leaf photos, it might fail when a real-world photo is taken at an angle. With augmentation, it gets exposed to all sorts of variations during training, making it much more resilient when it encounters new images it has never seen before.

### Application

#### Suggest a real-world application for your trained model.
##### Answer: One practical application would be a plant species identification tool for farmers and forestry workers in the Caraga region. A user could simply take a photo of an unfamiliar plant with their phone, and the model would identify the species and display relevant information — whether it's a protected tree species, an invasive plant, or a medicinal herb. This could support reforestation programs, biodiversity monitoring, or even aid in identifying poisonous plants in rural communities.

#### How can this system be integrated into a mobile or web application?
##### Answer: The trained model can be saved using model.save() and then converted to TensorFlow Lite format for mobile deployment. On Android or iOS, TensorFlow Lite allows the app to run the model directly on the device — no internet connection required. For a web application, the model can be converted to TensorFlow.js and embedded directly into a browser-based interface where users upload images and receive predictions in real time. Alternatively, the model can be hosted as a REST API using Flask or FastAPI on a server, and any frontend — web or mobile — can send an image and receive the classification result as a JSON response.


### Part 2: Guide Questions — Visualization, Overfitting & Deployment

### Visualization & Overfitting

#### What signs indicated overfitting in your first model?
##### Answer: The clearest sign was the growing gap between training accuracy and validation accuracy across epochs. By around epoch 6–7, training accuracy was climbing steadily while validation accuracy started to flatten or even slightly decrease. The loss curves told the same story — training loss kept going down, but validation loss began creeping back up instead of following the same trend. That divergence is the classic overfitting signal: the model was getting really good at recognizing the specific training images but wasn't learning features that transferred to new ones.

#### How did data augmentation affect validation accuracy?
##### Answer: Data augmentation noticeably stabilized validation accuracy. Instead of that erratic rise-and-fall pattern we saw in the first model, the augmented model's validation accuracy climbed more steadily and stayed more consistent across epochs. The gap between training and validation accuracy also narrowed. Essentially, because the model was seeing randomly transformed versions of the same images each epoch, it couldn't just memorize the training set — it was forced to actually learn the underlying visual features of each plant class.

### Model Improvement

#### What is the purpose of dropout layers?
##### Answer: Dropout works by randomly "turning off" a certain percentage of neurons during each training step — in our case, 30% (Dropout(0.3)). This might sound counterproductive, but it actually forces the network to become more resilient. Since the model can't rely on any specific set of neurons always being active, it has to distribute the learning more broadly across the network. The result is a model that generalizes better because no single neuron or pathway becomes overly dominant or overfitted to specific training patterns.

#### Why does data augmentation improve generalization?
##### Answer: Generalization is about how well a model performs on data it hasn't seen before. Data augmentation improves this by exposing the model to a much wider variety of visual scenarios during training — different orientations, scales, and crops of the same image. A model trained only on upright, well-lit, centered leaf photos will struggle with a rotated or partially shadowed one. Augmentation bridges that gap. It teaches the model that a Narra leaf is still a Narra leaf whether it's flipped, slightly zoomed, or tilted — which is exactly how real-world photos actually look.

### Performance Comparison

#### Compare accuracy before and after improvements.
#### Answer: The baseline model (no augmentation, no dropout) achieved around 72–78% validation accuracy after 10 epochs, but showed clear signs of overfitting — training accuracy was significantly higher. After adding data augmentation and dropout layers and extending training to 15 epochs, validation accuracy improved to approximately 82–87%, and more importantly, the training and validation curves were much closer together. The improvement wasn't just in the number — the model became more stable and consistent, which matters more for real-world use.


#### Which technique contributed most to improvement?
#### Answer: Data augmentation had the bigger overall impact, particularly in the early-to-mid epochs where the model's exposure to varied inputs prevented it from locking into overfitting patterns too early. Dropout complemented this by acting as a regularizer at the neuron level, further reducing overfitting in the deeper layers. If I had to choose one, data augmentation was the heavier lifter — but the combination of both is what produced genuinely reliable results.

### Deployment & Application

#### Why is saving the model important?
#### Answer: Training a CNN from scratch takes time, compute resources, and a lot of trial and error. Once you've arrived at a well-performing model, saving it means you don't have to go through all of that again just to use it. It also makes the model portable — you can load it into a different environment, share it with teammates, integrate it into an app, or redeploy it on a server without ever touching the training code again. In a real project, the saved model is essentially the deliverable — the artifact that does the actual work.

#### How can this model be deployed in a real-world system?
#### Answer: There are a few practical routes depending on the target platform. For a web application, the model can be served through a Python backend using Flask or FastAPI — the server loads the model once at startup, accepts image uploads via an API endpoint, runs the prediction, and returns the class name and confidence score as JSON. The frontend (whether plain HTML/JS or a framework like React) just needs to send the image and display the result. For mobile deployment, the model can be converted to TensorFlow Lite and bundled directly into an Android or iOS app, allowing offline predictions without needing a server. For a school or research setting, even a simple Google Colab + Gradio demo can serve as a lightweight deployment that anyone can access through a browser link without any installation.


