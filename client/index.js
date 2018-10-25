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
            if(result > 0){
                color = "lightgreen";
            }
            else if(result < 0){
                result*=-1;
                color="lightsalmon";
            }
            element.innerText = result.toFixed(2);
            element.style.backgroundColor = color;
        }
    }
    console.log("asdasd");
    req.open("GET",'/text/'+text);
    req.send();

    
}