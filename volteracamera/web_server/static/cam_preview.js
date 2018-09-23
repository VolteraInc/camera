"use strict";

function captureImage() {
  loader.pause();
  
  fetch ("/controls/capture_proper_image").then(
    function(response) {
      if(response.ok) {
  

      } else {
        console.log('Network request to save image fialed, respone ' + response.status + ': ' + response.statusText);
      }
    }
  );
}

function image_loader() {
 
  const url = '/controls/cam_image'
  var timeout; 

  function reload_image() {
    fetch(url, {cache: "no-store"}).then(function(response) {
      if(response.ok) {
        response.blob().then( function(blob) {
          var objectURL = URL.createObjectURL(blob);
          var img = document.getElementById('preview_image');
          img.src = objectURL;
          timeout = setTimeout (reload_image, 1);
        });
      } else {
        console.log('Network request for camera image failed with response ' + response.status + ': ' + response.statusText);
      }
    });
  };

  function pause () {
    clearTimeout(timeout);
  };

  reload_image(); 
  return {
    pause: pause,
    start: reload_image
  };

};

var loader = image_loader();


