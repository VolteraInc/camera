
function reload_image() {
var url = '/controls/cam_image'
fetch(url, {cache: "no-store"}).then(function(response) {
  if(response.ok) {
    response.blob().then( function(blob) {
      var objectURL = URL.createObjectURL(blob);
      var old_img = document.getElementById('preview_image');
      var new_img = new Image();
      new_img.id = old_img.id;
      new_img.src = objectURL;
      old_img.parentNode.insertBefore(new_img,old_img);
      old_img.parentNode.removeChild(old_img);
      console.log('Recieved ' + objectURL);
      setTimeout (reload_image, 500);
    });
  } else {
    console.log('Network request for camera image failed with response ' + response.status + ': ' + response.statusText);
  }
});
};

reload_image();
