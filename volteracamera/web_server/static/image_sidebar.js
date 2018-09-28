"use strict"

//expose to external namespace 
window.openNav = openNav;
function openNav() {
    document.getElementById("split-content").style.gridTemplateColumns = "20% 80%";
    document.getElementById("toggle-sidebar").onclick=closeNav;
    document.getElementById("toggle-sidebar").innerHTML="\<";
    document.getElementById("hideable-images").style.display = "block";
}

//expose to external namespace
window.closeNav = closeNav;
function closeNav() {
   document.getElementById("split-content").style.gridTemplateColumns = "4% 96%";
   document.getElementById("toggle-sidebar").onclick=openNav;
   document.getElementById("toggle-sidebar").innerHTML="\>";
   document.getElementById("hideable-images").style.display = "none";
}


window.loadSidebar = loadSidebar;
export function loadSidebar () {

  var url_sidebar = "/load_sidebar";
  var sidebar = document.getElementById('image-sidebar-contents');
  //Clear out the existing elements.
  while (sidebar.firstChild) {
    sidebar.removeChild(sidebar.firstChild);
  }

  //fetch the latest contents.
  fetch (url_sidebar).then(
    function (response ) {
      response.json().then(
        function (data) {

          
          for (let i = 0; i < data["images"].length; ++i) {
            var divElement = document.createElement("div");
            divElement.setAttribute("id", "sidebar_image");
            var removeImage = document.createElement("a");
            removeImage.setAttribute ("href", "#");
            removeImage.onclick = function () {deleteImage(i)};
            removeImage.innerHTML = "&#10005";
            var aElement = document.createElement("a");
            aElement.setAttribute("href", data["images"][i].image);
            var imageElement = document.createElement("img")
            imageElement.setAttribute ("src", data["images"][i].thumbnail);
            imageElement.setAttribute ("style", "width: 100%");

            divElement.appendChild( removeImage );
            aElement.appendChild( imageElement );
            divElement.appendChild( aElement );
            sidebar.appendChild( divElement );
    
          }
          openNav();
        });
    }).catch (  
    function (error) {  
      console.log ("Request failed", error);
    });

}

function deleteImage (num) {
  var url = "/delete_image/" + num;
  fetch (url); //don't need to do anything with the response.
  loadSidebar();
};


loadSidebar(); 


