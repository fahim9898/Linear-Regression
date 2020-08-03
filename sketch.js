/*
      Linear Line equation:   y = m * x + b
        x: X-axis point
        m: slope value
        b: Off set of y called as bias
*/


let m;                         
let b;
var lr=0.2;                                                           // Learning rate
var model;                                                            // tf model
const x_values=[];                                                    // Points position  
const y_values=[];

function setup() {
  
  // Setup canvas
  // canvas = createCanvas(windowWidth / 2, windowHeight / 1.5);
  // canvas.position(windowWidth / 4, windowHeight / 6)
  canvas = createCanvas(windowWidth, windowHeight);
  // canvas.position(windowWidth , windowHeight  )
  body = document.getElementById("body");
  body.style.backgroundColor = "#7e90ab"
  
  // Setup variables
	m=tf.variable(tf.scalar(random(1)));                               // Defined tf variabe m
  b = tf.variable(tf.scalar(random(1)));                             // Defined tf variabe b
	// m.print();
	model=tf.train.sgd(lr);
}

function windowResized(){
  canvas = resizeCanvas(windowWidth, windowHeight);
  // canvas.position(windowWidth , windowHeight )
}

function draw() {
  background("#303a52");
  stroke(255);
  strokeWeight(6);
  // Redraw all points selected points
  for(var i=0;i<x_values.length;i++){
      var map_x=map(x_values[i],0,1,0,width);
      var map_y=map(y_values[i],1,0,0,height);
      point(map_x,map_y);
  }

  if(x_values.length>0){      
    tf.tidy(()=>{
    model.minimize(function(){
          return( loss(x_values,y_values));
    });
  });
  }

  // Draw line using predict function
  strokeWeight(1);
  if (x_values.length == 0) return;
   
  tf.tidy(()=>{
    var l_x1,l_x2,l_y1,l_y2;          // line point
    
    l_x1=[0];
    l_x2=[1];
    l_y1=predict(l_x1);
    l_y2=predict(l_x2);
    
    map_x1=map(l_x1[0],0,1,0,width);
    map_y1=map(l_y1.get(0),1,0,0,height); 
    map_x2=map(l_x2[0],0,1,0,width);
    map_y2=map(l_y2.get(0),1,0,0,height);
    
    line(map_x1,map_y1,map_x2,map_y2);
  });
   
 }

// When Mose is pressed then add dots on screeen 
function mousePressed(){
	var x=map(mouseX,0,width,0,1);
  var y=map(mouseY,0,height,1,0);
  var IsInRect = IsInRectWraper(0, 0, width, height);
  if( IsInRect(mouseX, mouseY) ){
    x_values.push(x);
    y_values.push(y);
  }else{
    console.log("out")
  }
}

function IsInRectWraper(x, y, w, h){
  return (tx, ty)=>{
    if(tx >= x  && tx <= x+w  && ty >= y && ty <= y+h ){
      return true;
    }else{
      false
    }
  }
}

// Predict point according x-axis point   ( Using predict(X-axis Point)=> Return Y-axis Point)
function predict(x){
  var t_x=tf.tensor1d(x); 
  
  // Return value using y = m*x + b
  return (t_x.mul(m).add(b));
}

// Loss Function to update neural network
function loss(x,y){;
  var pre_t_y=predict(x);
  var t_y=tf.tensor1d(y);
  return (tf.losses.meanSquaredError(t_y,pre_t_y));
  // return (tf.losses.logLoss(t_y,pre_t_y));
}