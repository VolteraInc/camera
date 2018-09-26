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
export function loadSidebar (sidebar_element) {

  var url_sidebar = "/load_sidebar";
  var sidebar = sidebar_element;
  //Clear out the existing elements.
  while (sidebar.firstChild) {
    console.log ("Clearing sidebar")
    sidebar.removeChild(sidebar.firstChild);
  }

  //fetch the latest contents.
  fetch (url_sidebar).then(
    function (response ) {
      response.json().then(
        function (data) {

          console.log(data);
          data["images"].forEach(function(image) {
            console.log (image.image)
            var divElement = new document.createElement("div");
            divElement.setAttribute("id", "sidebar_image")    
            var aElement = new document.createElement("a");
            aElement.setAttribute("href", image.image);
            var imageElement = new document.createElement("img")
            imageElement.setAttribute ("src", image.thumbnail);

            aElement.appendChild( imageElement );
            divElement.appendChild( aElement );
            sidebar.appendChild( divElement );
    
          })
          openNav();
        });
    }).catch (  
    function (error) {  
      console.log ("Request failed", error);
    });

}


var sidebar_element = document.getElementById('image-sidebar-contents');
loadSidebar(sidebar_element); 


