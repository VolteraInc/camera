"use strict"

function openNav() {
    document.getElementById("split-content").style.gridTemplateColumns = "20% 80%";
    document.getElementById("toggle-sidebar").onclick=closeNav;
    document.getElementById("toggle-sidebar").innerHTML="\<";
    document.getElementById("hideable-images").style.display = "block";
}

function closeNav() {
   document.getElementById("split-content").style.gridTemplateColumns = "4% 96%";
   document.getElementById("toggle-sidebar").onclick=openNav;
   document.getElementById("toggle-sidebar").innerHTML="\>";
   document.getElementById("hideable-images").style.display = "none";
}

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
          data["images"].forEach((image) => {
          
            console.log ("in sidebar")
            var divElement = new document.createElement("div");
            divElement.setAttribute("id", "sidebar_image")    
            var aElement = new document.createElement("a");
            aElement.setAttribute("href", image.image);
            var imageElement = new document.createElement("img")
            imageElement.setAttribute ("src", image.thumbnail);

            aElement.appendChild( imageElement );
            divElement.appendChild( aElement );
            sidebar.appendChild( aElement );
    
            openNav();
          })
        });
    }).catch (  
    function (error) {  
      console.log ("Request failed", error);
    });

}

loadSidebar(); 


