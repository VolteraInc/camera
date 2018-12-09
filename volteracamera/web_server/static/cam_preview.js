"use strict";

var CamPreview = CamPreview || {};

CamPreview = (function () {

  var captureImage = function (callback_action) {
    loader.pause();

    fetch("/controls/capture_proper_image").then(
      function (response) {
        if (response.ok) {
          let sidebar_element = document.getElementById('image-sidebar-contents');
          loader.start()
          if (typeof(callback_action) === "function") {
            callback_action();
          }
        } else {
          console.log('Network request to save image failed, response ' + response.status + ': ' + response.statusText);
          loader.start()
        }
      }
    );
  }


  //Class for reloading images automatically
  var image_loader = function () {

    const url = '/controls/preview_cam_image'
    var timeout;

    function reload_image() {
      fetch(url, { cache: "no-store" }).then(function (response) {
        if (response.ok) {
          response.blob().then(function (blob) {
            var objectURL = URL.createObjectURL(blob);
            var img = document.getElementById('preview_image');
            img.src = objectURL;
            timeout = setTimeout(reload_image, 1);
          });
        } else {
          console.log('Network request for camera image failed with response ' + response.status + ': ' + response.statusText);
        }
      });
    };

    function pause() {
      clearTimeout(timeout);
    };

    reload_image();
    return {
      pause: pause,
      start: reload_image
    };

  };

  var loader = image_loader()
  return {
    loader: loader,
    captureImage: captureImage
  };
})();


