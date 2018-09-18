
function reload_image() {
var url = '/controls/cam_image'
fetch(url, {cache: "no-store"}).then(function(response) {
  if(response.ok) {
    response.blob().then( function(blob) {
      var objectURL = URL.createObjectURL(blob);
      var img = document.getElementById('preview_image');
      img.src = objectURL;
      setTimeout (reload_image, 500);
    });
  } else {
    console.log('Network request for camera image failed with response ' + response.status + ': ' + response.statusText);
  }
});
};

reload_image();
