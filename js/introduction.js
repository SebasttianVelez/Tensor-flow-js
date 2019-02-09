async function learnLinear(){

  //Creacion de una red neuronal con un unico nodo
  const model = tf.sequential();
  //"units" e "imputShape" manejan en numero de nodos
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));
  
  //Compilar el modelo
  model.compile({
   loss: 'meanSquaredError', //tipo de perdida
   optimizer: 'sgd' //optimizador descenso de gradiente estocastico estandar
  });
  
  //Datos para entrenar el modelo

  //Datos de entrada
  const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]); // shape
  
  //Datos de salida
  const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]); //shape
 
  //metodo fit para entrenar el modelo
  await model.fit(xs, ys, {epochs: 500});

  //epochs son el numero de loops, o numero de iteraciones sobre todos los datos de entrenamiento.
  

  //impresion del tensor con el metodo predict
  document.getElementById('output_field').innerText =
   model.predict(tf.tensor2d([24], [1, 1]));
 }
 learnLinear();