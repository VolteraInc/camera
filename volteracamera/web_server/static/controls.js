"use strict"

//Grab the needed controls
var laserIntensityValue = document.getElementById("laser-intensity-value");
var sensorExposureValue = document.getElementById("sensor-exposure-value");
var laserStateCheckbox = document.getElementById("laser-toggle");
var laserIntensitySlider = document.getElementById("laser-power-adjust");
var sensorExposureSlider = document.getElementById("sensor-exposure-adjust");

const laserURL = '/controls/laser_state';
const sensorURL = '/controls/sensor_state';

//Tie together the slider controls.
function setupSliderControl ( sliderElement, textElement, requestBuilder, responseHandler ) {

  var slider = sliderElement;
  var text = textElement;
  var requestBuild = requestBuilder;
  var responseFunc = responseHandler

  //update the value whenever the slider changes.
  slider.onmouseup = function() {
     fetchState( requestBuild(), responseFunc ); 
  }

  slider.oninput = function() {
     text.innerHTML = slider.value; 
  }

  function setValue ( value ) {
    if ((value > slider.max) || (slider.min < slider.min)) {
      return;
    }
    text.innerHTML = value;
    slider.value = value;
  };

  return setValue;
}

//Sets up the callbacks and state updates needed for the laser slider (including the fetch logic)
var setLaserValue = setupSliderControl (laserIntensitySlider, laserIntensityValue, getLaserRequest, function(data) {
  laserIntensitySlider.value = data.intensity;
  laserIntensityValue.innerHTML = data.intensity;
  laserStateCheckbox.checked = data.on_off; 
});

laserStateCheckbox.oninput = function () {
  fetchState ( getLaserRequest(), function(data) {
    laserIntensitySlider.value = data.intensity;
    laserIntensityValue.innerHTML = data.intensity;
    laserStateCheckbox.checked = data.on_off;  
  });
};

//Get the laser json data to send in requests to the server.
function getLaserRequest () {
  let data = {
    on_off : laserStateCheckbox.checked,
    intensity : laserIntensitySlider.value
  };
  let request = new Request(
    laserURL,
    {method: 'POST',
     headers: new Headers({'Content-Type':'application/json'}),
     body: JSON.stringify(data),
     cache: 'no-store'
    });
  return request;
}

//This function fetches the various states, setting a new state if the request if build properly.
//Called on page load to get the initial state and every time the state changes.
function fetchState ( request, valueChangeHandler ) {
  fetch(request)
    .then( function (response) {
      response.json()
        .then( function (data) {valueChangeHandler(data)} )
      })
    .catch(function (error) {
      console.log('Request failed', error);
    });
};


//load initial laser values with an empty request.
fetchState (new Request(laserURL, {method: 'POST', cache: 'no-store'}), function(data) {
  laserStateCheckbox.checked = data.on_off;
  setLaserValue (data.intensity);
});

