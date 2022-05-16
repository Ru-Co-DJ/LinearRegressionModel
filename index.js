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
model.add(tf.layers.dense({inputShape:[120], units:100, activation:"relu"}))
model.add(tf.layers.dense({inputShape:[100], units:100, activation:"relu"}))
model.add(tf.layers.dense({inputShape:[100], units:80, activation:"relu"}))
model.add(tf.layers.dense({inputShape:[80], units:1, activation:"relu"}))

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





//data to test {"Wilaya":"Oran",","Marque":"Equiplique","Modele":"Over Size",Quantite":35,"Benefice":12250}




const newData1 = {
    Wilaya:"Oran",
    Marque:"Equiplique",
    Modele:"Over Size",
    Quantite:35,
    Benefice:12250
}
const arrr1 = [
    newData1.Quantite,
    oneHot(cat.WILAYAS.indexOf(newData1.Wilaya),cat.WILAYAS.length), // wilaya
    oneHot(cat.MARQUES.indexOf(newData1.Marque),cat.MARQUES.length), //marque
    oneHot(cat.MODELS.indexOf(newData1.Modele),cat.MODELS.length) //model
] .flat()



console.log(arrr1)

  console.log(
    `prediction for Wilaya:"Oran", Marque:"Equiplique", Modele:"Over Size", Quantite:35 ` +
      model.predict(tf.tensor(arrr1, [1, xs[0].length])).arraySync()*10000
  );
/*
const newData2 = {
    Wilaya:"Oran",
    Marque:"Lefties",
    Modele:"ORG",
    Quantite:1,
    Benefice:600
}

const arrr2 = [
    newData1.Quantite,
    oneHot(cat.WILAYAS.indexOf(newData2.Wilaya),cat.WILAYAS.length), // wilaya
    oneHot(cat.MARQUES.indexOf(newData2.Marque),cat.MARQUES.length), //marque
    oneHot(cat.MODELS.indexOf(newData2.Modele),cat.MODELS.length) //model
] .flat()

console.log(arrr1)

  console.log(
    `prediction for Wilaya:"Oran", Marque:"ShootPower", Modele:"lecra", Quantite:1 ` +
      model.predict(tf.tensor(arrr1, [1, 59])).arraySync()*10000
  );

  console.log(
    `prediction for Wilaya:"Tiaret", Marque:"Lefties", Modele:"ORG", Quantite:1 ` +
      model.predict(tf.tensor(arrr2, [1, 59])).arraySync()*10000
  );
  


*/
