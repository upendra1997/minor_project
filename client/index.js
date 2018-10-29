'use strict';

function clicked(event){
    var text = document.getElementById('text').value.trim();
    if(text.length == 0){
        return;
    }
    console.log(text);
    var req = new XMLHttpRequest();
    req.onreadystatechange = function(){
        if(req.readyState == 4 && req.status == 200){
            var result = parseFloat(req.responseText);
            var color = "lightgray";
            var element = document.getElementById('result');
            if(result > 0.5+1e-9){
                result = result - 0.5;
                color = "lightgreen";
            }
            else if(result < 0.5-1e-9){
                result = 0.5 - result;
                color="lightsalmon";
            }
            else{
                result = 0.00;
            }
            element.innerText = result.toFixed(2);
            element.style.backgroundColor = color;
        }
    }
    req.open("GET",'/text/'+text);
    req.send();

    
}