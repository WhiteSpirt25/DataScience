import pandas as pd
import matplotlib.pyplot as plt


#reading and splitting in groups of 1 image
data = open("semeion_data")
data = data.read()
list_data = data.split('\n')

#parsing(making image into 256 size vector)
image_list = []
images = [None] * 1593
listNumb = []
for image in range(1593):
    image_list = list_data[image].split(' ')
    temp = [None] * 256

    for elem in range(256):
        temp[elem] = float(image_list[elem])

    images[image]=temp
    listNumb.append(0)
    for i in range(10):
        if float(image_list[256 + i]) > 0:
            listNumb[image] = i

#graphics
from sklearn.decomposition import PCA as sklearnPCA
pca = sklearnPCA(n_components=2)
transformed = pd.DataFrame(pca.fit_transform(images))

for i in range(5):
    plt.scatter(transformed[i*200:i*200+20][0], transformed[i*200:i*200+20][1], label='Class 0', c='red')
    plt.scatter(transformed[i*200+20:i*200+40][0], transformed[i*200+20:i*200+40][1], label='Class 1', c='blue')
    plt.scatter(transformed[i*200+40:i*200+60][0], transformed[i*200+40:i*200+60][1], label='Class 2', c='cyan')
    plt.scatter(transformed[i*200+60:i*200+80][0], transformed[i*200+60:i*200+80][1], label='Class 3', c='green')
    plt.scatter(transformed[i*200+80:i*200+100][0], transformed[i*200+80:i*200+100][1], label='Class 4', c='black')
    plt.scatter(transformed[i*200+100:i*200+120][0], transformed[i*200+100:i*200+120][1], label='Class 5', c='yellow')
    plt.scatter(transformed[i*200+120:i*200+140][0], transformed[i*200+120:i*200+140][1], label='Class 6', c='magenta')
    plt.scatter(transformed[i*200+140:i*200+160][0], transformed[i*200+140:i*200+160][1], label='Class 7', c='orange')
    plt.scatter(transformed[i*200+160:i*200+180][0], transformed[i*200+160:i*200+180][1], label='Class 8', c='violet')
    plt.scatter(transformed[i*200+180:i*200+200][0], transformed[i*200+180:i*200+200][1], label='Class 9', c='#58d68d')
plt.show()

#train split
from sklearn.model_selection import train_test_split

y = listNumb
X = images

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=0)

#knn in action
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=6)

model.fit(X_train,y_train)

print(model.score(X_test, y_test))



