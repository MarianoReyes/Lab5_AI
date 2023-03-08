# Laboratorio 5

## Task 1

### Repita los pasos para entrenar su modelo, pero ahora usando librerías y compare los resultados. ¿Cuál implementación fue mejor? ¿Por qué?

- Es mejor la que usa librerias ya que estas son más exactas porque ya están diseñadas para implementar este tipo de algoritmos y hacer que sea más fácil su uso.


## Task 2

### Repita los pasos para entrenar su modelo, pero ahora usando librerías y compare los resultados. ¿Cuál implementación fue mejor? ¿Por qué? (Responda como parte del readme de su repositorio)

- La implementación que usa librerias es mejor por diferentes razones

  Primordialmente se puede ver la diferencia por los resultados, ya que el modelo con las librerias implementadas fue mucho mas eficiente y con un mejor resultado de accuracy. Los factores que afectan son:

  Eficiencia: Sklearn está optimizado para la velocidad y la eficiencia, utilizando algoritmos avanzados y estructuras de datos para realizar cálculos complejos en una fracción del tiempo requerido por las implementaciones personalizadas.

  Escalabilidad: Sklearn está diseñado para escalar a grandes conjuntos de datos y manejar una amplia gama de tareas, desde el preprocesamiento de datos y la extracción de características hasta la selección y evaluación de modelos. Ya que con el realizado por nostros tuvimos que modificar aun mas la data para que se pudiera implementar. 

  Flexibilidad: Sklearn proporciona una amplia gama de modelos y algoritmos que se pueden personalizar y combinar fácilmente para satisfacer necesidades específicas, así como herramientas para la selección de características, validación cruzada, ajuste de hiperparámetros y más.

## Al final de los tasks 1.1 y 1.2, responda como parte del readme de su repositorio (del task 1.1 use aquella implementación que lo haya hecho mejor según su criterio):
##¿Cómo difirieron los grupos creados por ambos modelos?
  - 
##¿Cuál de los modelos fue más rápido?
   El KNN con libreias fue un poco mas rapido que el SVM con librerias, y sin librerias el KNN fue mucho mas rapido y tuve mejor accuracy significativamente comparado con el SVM sin librerias.
  
##¿Qué modelo usarían?
  Basandonos puramente en accuracy vemos como el algoritmos tiene un desempeño un poco mas alto. sin embargo dado que los datos parecen tener un número moderado de características (8 en total) y una cantidad razonable de datos de entrenamiento, KNN podría ser una buena opción.

  SVM también podría ser una buena opción, especialmente si hay una clara separación lineal entre los correos phishing y legítimos. En ese caso, se podría utilizar un kernel lineal para separar las dos clases.

  Tanto KNN como SVM podrían ser algoritmos adecuados para clasificar los correos electrónicos como phishing o legítimos.
