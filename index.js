import tf, { sequential } from "@tensorflow/tfjs";
import DATA from "./D.json" assert { type: "json" };


const oneHot = (value, categories) => {
    return Array.from(tf.oneHot(value, categories).dataSync()); //dataSync() is for extracting the data
  };


  const normalize = (tensor) => {
    return tf.div(
      tf.sub(tensor, tf.min(tensor)),
      tf.sub(tf.max(tensor), tf.min(tensor))
    );
  };

function CATEGO (){
    let wilayas = []
    let marques = []
    let models = []


    DATA.forEach((e)=>{
       wilayas.push(e.Wilaya)
       marques.push(e.Marque)
       models.push(e.Modele)
    })
    return {WILAYAS:[...new Set(wilayas)], MARQUES: [...new Set(marques)], MODELS:[...new Set(models)]}
}


const cat = CATEGO()
console.log(cat)
let xs = []
let ys = []

DATA.forEach((e)=>{
    xs.push(
        [
            e.Quantite, // quantity
            oneHot(cat.WILAYAS.indexOf(e.Wilaya),cat.WILAYAS.length), // wilaya
            oneHot(cat.MARQUES.indexOf(e.Marque),cat.MARQUES.length), //marque
            oneHot(cat.MODELS.indexOf(e.Modele),cat.MODELS.length) //model
        ].flat()
    )
    ys.push(
        e.Benefice/10000
    )
})
console.log(ys)
// length of xs is 129
const model = sequential()

model.add(tf.layers.dense({inputShape:[xs[0].length], units:120, activation:"relu"}))
model.add(tf.layers.dense({inputShape:[120], units:90, activation:"relu"}))
model.add(tf.layers.dense({inputShape:[90], units:60, activation:"relu"}))
model.add(tf.layers.dense({inputShape:[60], units:40, activation:"relu"}))
model.add(tf.layers.dense({inputShape:[40], units:1, activation:"relu"}))

model.summary()


model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.adam(0.01),
    metrics: ["accuracy"],
  });


const features = tf.tensor(xs,[xs.length,xs[0].length]) 
const labels = tf.tensor(ys,[ys.length,1])




await model
  .fit(features, labels, {
    epochs: 400,
    shuffle:true,
    batchSize:128,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log("epoch: " + epoch + " loss: " + logs.loss);
      },
    },
  })
  .then((info) => {
    console.log("Accuracy: " + info.history.acc);
  });





// data to test {"Wilaya":"Oran",","Marque":"Equiplique","Modele":"Over Size",Quantite":35,"Benefice":12250}
const newData1 = {
    Wilaya:"Oran",
    Marque:"Equiplique",
    Modele:"Over Size",
    Quantite:35,
    Benefice:12250
}


// "Wilaya":"Oran","Marque":"dior","Modele":"semelle bleu","Quantite":1,"Gratuits":0,"PT":5200,"Benefice":1200}
const newData2 = {
  Wilaya:"Oran",
  Marque:"dior",
  Modele:"semelle bleu",
  Quantite:1,
  Benefice:1200
}


//"Wilaya":"Oran","Marque":"K","Modele":"divers","Quantite":1,"Gratuits":0,"PT":4000,"Benefice":600}
const newData3 = {
  Wilaya:"Oran",
  Marque:"K",
  Modele:"divers",
  Quantite:1,
  Benefice:600
}

//"Wilaya":"Oran","Marque":"ShootPower","Modele":"lecra","Quantite":1,"Gratuits":0,"PT":2100,"Benefice":700}
const newData4 = {
  Wilaya:"Oran",
  Marque:"ShootPower",
  Modele:"lecra",
  Quantite:1,
  Benefice:700
}


//"Wilaya":"Oran","Marque":"foot","Modele":1,"Quantite":1,"Gratuits":0,"PT":1200,"Benefice":300}
const newData5 = {
  Wilaya:"Oran",
  Marque:"foot",
  Modele:1,
  Quantite:1,
  Benefice:300
}


const arrr1 = [
    newData1.Quantite,
    oneHot(cat.WILAYAS.indexOf(newData1.Wilaya),cat.WILAYAS.length), // wilaya
    oneHot(cat.MARQUES.indexOf(newData1.Marque),cat.MARQUES.length), //marque
    oneHot(cat.MODELS.indexOf(newData1.Modele),cat.MODELS.length) //model
] .flat()

  console.log(
    `\nprediction for Wilaya:${newData1.Wilaya}, Marque:${newData1.Marque}, Modele:${newData1.Modele}, Quantite:${newData1.Quantite} ` +
      model.predict(tf.tensor(arrr1, [1, xs[0].length])).arraySync()*10000
  );

  const arrr2 = [
    newData2.Quantite,
    oneHot(cat.WILAYAS.indexOf(newData2.Wilaya),cat.WILAYAS.length), // wilaya
    oneHot(cat.MARQUES.indexOf(newData2.Marque),cat.MARQUES.length), //marque
    oneHot(cat.MODELS.indexOf(newData2.Modele),cat.MODELS.length) //model
] .flat()

  console.log(
    `\nprediction for Wilaya:${newData2.Wilaya}, Marque:${newData2.Marque}, Modele:${newData2.Modele}, Quantite:${newData2.Quantite} ` +
      model.predict(tf.tensor(arrr2, [1, xs[0].length])).arraySync()*10000
  );

  const arrr3 = [
    newData3.Quantite,
    oneHot(cat.WILAYAS.indexOf(newData3.Wilaya),cat.WILAYAS.length), // wilaya
    oneHot(cat.MARQUES.indexOf(newData3.Marque),cat.MARQUES.length), //marque
    oneHot(cat.MODELS.indexOf(newData3.Modele),cat.MODELS.length) //model
] .flat()

  console.log(
    `\nprediction for Wilaya:${newData3.Wilaya}, Marque:${newData3.Marque}, Modele:${newData3.Modele}, Quantite:${newData3.Quantite} ` +
      model.predict(tf.tensor(arrr3, [1, xs[0].length])).arraySync()*10000
  );

  const arrr4 = [
    newData4.Quantite,
    oneHot(cat.WILAYAS.indexOf(newData4.Wilaya),cat.WILAYAS.length), // wilaya
    oneHot(cat.MARQUES.indexOf(newData4.Marque),cat.MARQUES.length), //marque
    oneHot(cat.MODELS.indexOf(newData4.Modele),cat.MODELS.length) //model
] .flat()

  console.log(
    `\nprediction for Wilaya:${newData4.Wilaya}, Marque:${newData4.Marque}, Modele:${newData4.Modele}, Quantite:${newData4.Quantite} ` +
      model.predict(tf.tensor(arrr4, [1, xs[0].length])).arraySync()*10000
  );

  const arrr5 = [
    newData5.Quantite,
    oneHot(cat.WILAYAS.indexOf(newData5.Wilaya),cat.WILAYAS.length), // wilaya
    oneHot(cat.MARQUES.indexOf(newData5.Marque),cat.MARQUES.length), //marque
    oneHot(cat.MODELS.indexOf(newData5.Modele),cat.MODELS.length) //model
] .flat()

  console.log(
    `\nprediction for Wilaya:${newData5.Wilaya}, Marque:${newData5.Marque}, Modele:${newData5.Modele}, Quantite:${newData5.Quantite} ` +
      model.predict(tf.tensor(arrr5, [1, xs[0].length])).arraySync()*10000
  );
