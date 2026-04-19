#A klasszikus neurális hálózat működésének órai magyarázatához tartozó segédkód
#Használd a  network_architect.png képen lévő szituációt: jelen példa arra folyamatra épül

input_vector = [1.66, 1.56]
weights_1 = [1.45, -0.66]
bias = [0.0]

#1. mutassuk be, hogy keletkezik egy  neuron az első rejtett rétegbe: a bemeneti vektor szorozva a súly vektor

first_indexes_mult = input_vector[0] * weights_1[0]
second_indexes_mult = input_vector[1] * weights_1[1]
dot_product_1 = first_indexes_mult + second_indexes_mult

print(f"The dot product is: {dot_product_1}")

#numpy könyvtárral még könnyebb a szorzat számítás:
import numpy as np
dot_product_1 = np.dot(input_vector, weights_1)
print(f"The dot product is: {dot_product_1}")


#A 0-1 közötti kiemenet előállításához szükség van egy nem lineráis transzformációra: pl a logisztikus függvény

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# A predikció előállítása: az első rejtett rétegbeli neuron esetén vesszük - a fent deifinált - lienáris transzformáciját
def make_prediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    return layer_2

#
# Kipróbáljuk a predikciót az input vektorra

prediction = make_prediction(input_vector, weights_1, bias)

print(f"The prediction result is: {prediction}")



 # most egy másik input vektorra is megnézzük
input_vector = np.array([2, 1.5])
prediction = make_prediction(input_vector, weights_1, bias)
print(f"The prediction result is: {prediction}")



# Vizsgáljuk meg a predikció jóságát: tegyük fel hogy a fenti [2, 1.5] input vektorhoz az elvárt predikció a: 0
target = 0
# nézzük meg az eltérést(hibát): "négyzetes eltérés" a prediktált érték és az elvárt érték között, láthatjuk, hogy jó nagy az eltérés, tehát nagy hibával prediktál a háló

mse = np.square(prediction - target)
print(f"Prediction: {prediction}; Error: {mse}")


#Hogyan lehet a hibát minimalizálni? Gradiens süllyedés módszere: deriválás szerepe: szóbeli magyarázat
#Mivel a hiba függvényem jelen példában egy másodfokú függvény (x^2), a deriváltja 2*x
# használd a gradient_descent.png képet a magyarázathoz (derivált szerepe a szélsőértékek meghatározásásoz)
derivative = 2 * (prediction - target)
print(f"The derivative is {derivative}")

# Frissítjük a súlyokat a derivált segítségéve, láthatjuk, hogy a hiba sokkal kissebb lett:
weights_1 = weights_1 - derivative
prediction = make_prediction(input_vector, weights_1, bias)
error = (prediction - target) ** 2
print(f"Prediction: {prediction}; Error: {error}")


#Tanítási folyamat, backpropagation: szóbeli magyarázat csak()


