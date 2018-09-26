import {loadSidebar} from '/static/image_sidebar.js';
"use strict";

//Expose capture image to namespace outside module.
window.captureImage = captureImage
window.loadSidebar = loadSidebar

function captureImage() {
  loader.pause();
  
  fetch ("/controls/capture_proper_image").then(
    function(response) {
      if(response.ok) {
        let sidebar_element = document.getElementById('image-sidebar-contents');
        loadSidebar(sidebar_element)
        loader.start()
      } else {
        console.log('Network request to save image fialed, respone ' + response.status + ': ' + response.statusText);
        loader.start()
      }
    }
  );
}


//Function for reloading images automatically
function image_loader() {
 
  const url = '/controls/preview_cam_image'
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

//instance of the reloader
var loader = image_loader();


