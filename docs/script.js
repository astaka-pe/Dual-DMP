// fandisk
const inputElem = document.getElementById('step');
const currentValueElem = document.getElementById('current-value');
const currentImageElem = document.getElementById('current-image');
const img = document.getElementById("image-place");

const setCurrentValue = (val) => {
    var num = ("000" + (Number(val)/10-1)).slice(-2)
    
    currentValueElem.innerText = val;
    currentImageElem.innerText = "images/fandisk_anim/snapshot00_L" + String(num) + ".png";
    img.src = "images/fandisk_anim/snapshot00_L" + String(num) + ".png";
}

const rangeOnChange = (e) =>{
  setCurrentValue(e.target.value);
}

// window.onload = () => {
//   inputElem.addEventListener('input', rangeOnChange);
//   setCurrentValue(inputElem.value);
// }

// ankylosaurus
const inputElem2 = document.getElementById('step2');
const currentValueElem2 = document.getElementById('current-value2');
const currentImageElem2 = document.getElementById('current-image2');
const img2 = document.getElementById("image-place2");

const setCurrentValue2 = (val) => {
    var num2 = ("000" + (Number(val)/10-1)).slice(-2)
    
    currentValueElem2.innerText = val;
    currentImageElem2.innerText = "images/ankylosaurus_anim/snapshot00_L" + String(num2) + ".png";
    img2.src = "images/ankylosaurus_anim/snapshot00_L" + String(num2) + ".png";
}

const rangeOnChange2 = (e) =>{
  setCurrentValue2(e.target.value);
}

window.onload = () => {
    inputElem.addEventListener('input', rangeOnChange);
    setCurrentValue(inputElem.value);
    inputElem2.addEventListener('input', rangeOnChange2);
    setCurrentValue2(inputElem2.value);
}